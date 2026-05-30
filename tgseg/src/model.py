import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ResNetBackbone, FPNProjection
from text_encoder import TextEncoder
from fusion import CrossModalFusion
from deformable_transformer import DeformableTransformer
from mask_head import PaperDynamicMaskHead
from positional_encoding import PositionEmbeddingSine
from memory_read import MaskMemoryRead
from matcher import HungarianMatcher
from criterion import SetCriterion, build_targets_from_gt_masks
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from losses import sigmoid_focal_loss, mask_focal_dice_loss_per_query


class TPPMini(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_queries=5,
        use_propagation=True,
        lambda_presence: float = 1.0,
        presence_tau: float = 0.5,
        lambda_empty: float = 0.0,   
    ):
        super().__init__()


        self.backbone = ResNetBackbone(pretrained=True)
        self.fpn = FPNProjection(self.backbone.out_channels, hidden_dim)

        
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            normalize=True
        )

        
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim, freeze=True)
        self.fusion = CrossModalFusion(hidden_dim=hidden_dim)

        
        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            num_queries=num_queries,
            n_levels=3,          
            n_heads=8,
            n_points=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_ffn=hidden_dim * 4,
            dropout=0.1
        )

        
        self.mask_head = PaperDynamicMaskHead(hidden_dim=hidden_dim, dyn_channels=8, num_layers=3)

        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        
        self.presence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.lambda_presence = float(lambda_presence)
        self.presence_tau = float(presence_tau)
        self.lambda_empty = float(lambda_empty)

        self.track_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.memory_read = MaskMemoryRead(hidden_dim)

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.use_propagation = use_propagation

        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, cost_mask=1.0)
        self.criterion = SetCriterion(
            self.matcher,
            lambda_cls=2.0, lambda_box=5.0, lambda_giou=2.0,
            lambda_mask=1.0, lambda_dice=1.0
        )

    def select_best_query(self, class_logits):
        if class_logits.dim() == 3:
            class_logits = class_logits.squeeze(-1)
        return class_logits.argmax(dim=1)

    def _pad_to_num_queries(self, x, pad_value=0.0):
        B, Q = x.shape[:2]
        if Q == self.num_queries:
            return x
        pad_q = self.num_queries - Q
        pad_shape = list(x.shape)
        pad_shape[1] = pad_q
        pad = x.new_full(pad_shape, pad_value)
        return torch.cat([x, pad], dim=1)

    def forward_frame(
        self,
        img,
        captions,
        prev_query=None,
        prev_mask=None,          
        prev_enc_lvl0=None,      
        prev_box=None
    ):
        
        B = img.size(0)
        device = img.device

       
        feats = self.backbone(img)        
        feats = self.fpn(feats)          
       

        feat_shapes = []
        vision_tokens_by_level = []

        for f in feats:
            _, D, H, W = f.shape
            feat_shapes.append((H, W))
            pos = self.position_embedding(f)                     
            tokens = f.flatten(2).permute(0, 2, 1)                  
            pos_t  = pos.flatten(2).permute(0, 2, 1)               
            vision_tokens_by_level.append(tokens + pos_t)          

        if isinstance(captions[0], list):
            if len(captions) == B:
                prompt_lists = captions
            else:
                prompt_lists = [captions[0] for _ in range(B)]
        else:
            prompt_lists = [captions for _ in range(B)]

        K = len(prompt_lists[0])
        assert K >= 1

        word_feats_list = []
        sent_feats_list = []
        attn_mask_list  = []

        for k in range(K):
            batch_prompt_k = [prompt_lists[b][k] for b in range(B)]
            wf, sf, am = self.text_encoder(batch_prompt_k, device)
            word_feats_list.append(wf)
            sent_feats_list.append(sf)
            attn_mask_list.append(am)

        
        fused_tokens_by_level, prompt_weights = self.fusion(
            vision_tokens_by_level,
            word_feats_list,
            attn_mask_list
        )

        w_lvl0 = prompt_weights[:, 0]     
        best_p = w_lvl0.argmax(dim=1)     

        selected_sent_feats = []
        for b in range(B):
            selected_sent_feats.append(sent_feats_list[best_p[b].item()][b])
        selected_sent_feats = torch.stack(selected_sent_feats, dim=0)  

        
        tx_tokens = [fused_tokens_by_level[1], fused_tokens_by_level[2], fused_tokens_by_level[3]]
        tx_shapes = [feat_shapes[1], feat_shapes[2], feat_shapes[3]]   

       
        memory, spatial_shapes, level_start_index = self.transformer.encode(tx_tokens, tx_shapes)

      
        H3, W3 = tx_shapes[0]
        HW3 = H3 * W3
        start0 = int(level_start_index[0].item())
        end0   = start0 + HW3


        enc0_tokens = memory[:, start0:end0]  
        enc0_map = enc0_tokens.transpose(1, 2).reshape(B, self.hidden_dim, H3, W3)  

        enc0_used = enc0_map

        if (prev_mask is not None) and (prev_enc_lvl0 is not None):
            prev_mask_ds = F.interpolate(prev_mask.unsqueeze(1), size=(H3, W3), mode="nearest").squeeze(1)
            mem_read_map = self.memory_read(
                curr_feat=enc0_map,
                prev_feat=prev_enc_lvl0,
                prev_mask=prev_mask_ds
            )
            mem_read_tokens = mem_read_map.flatten(2).transpose(1, 2)   

            memory = memory.clone()
            memory[:, start0:end0] = mem_read_tokens
            enc0_used = mem_read_map


        queries = self.transformer.decode(
            memory,
            spatial_shapes,
            level_start_index,
            prev_query=prev_query,
            text_query=selected_sent_feats,
            use_propagation=self.use_propagation
        )

        masks = self.mask_head(queries, feats)     
        box_offsets = self.box_head(queries)       
        class_logits = self.class_head(queries)    


        pooled = queries.max(dim=1).values  
        presence_logit = self.presence_head(pooled).squeeze(-1)  

        if prev_box is not None:
            boxes = prev_box.unsqueeze(1) + box_offsets
        else:
            boxes = box_offsets

        boxes = boxes.sigmoid()

        return queries, masks, boxes, class_logits, presence_logit, enc0_used




    def forward_sequence(self, imgs, captions):
        B, T = imgs.shape[:2]
        device = imgs.device

        prev_query = None
        prev_box   = None
        prev_mask  = None
        prev_enc0  = None

        all_masks = []
        all_boxes = []
        all_scores = []
        all_presence = []

        for t in range(T):
            frame = imgs[:, t]

            queries, masks, boxes, class_logits, presence_logit, enc0_used = self.forward_frame(
                frame,
                captions,
                prev_query=prev_query,
                prev_mask=prev_mask,
                prev_enc_lvl0=prev_enc0,
                prev_box=prev_box
            )

            all_presence.append(presence_logit)  

            
            presence_prob = torch.sigmoid(presence_logit)
            pred_present = (presence_prob >= self.presence_tau).float()  
            mask_gate = pred_present.view(B, 1, 1)
            box_gate  = pred_present.view(B, 1)
            query_gate = pred_present.view(B, 1)
            enc_gate  = pred_present.view(B, 1, 1, 1)

            best_idx = self.select_best_query(class_logits)  

            best_query = torch.stack([queries[b, best_idx[b]] for b in range(B)], dim=0)  
            best_box   = torch.stack([boxes[b,  best_idx[b]] for b in range(B)], dim=0)  
            best_mask  = torch.stack([masks[b,  best_idx[b]] for b in range(B)], dim=0)  

            if self.use_propagation:
                prev_query = (self.track_mlp(best_query) * query_gate).detach()
                prev_mask  = (best_mask * mask_gate).detach()
                prev_box   = (best_box  * box_gate).detach()
                prev_enc0  = (enc0_used * enc_gate).detach()
            else:
                prev_query = None
                prev_mask  = None
                prev_box   = None
                prev_enc0  = None

            masks_store  = self._pad_to_num_queries(masks, pad_value=0.0)
            boxes_store  = self._pad_to_num_queries(boxes, pad_value=0.0)
            scores_store = self._pad_to_num_queries(class_logits.squeeze(-1), pad_value=-1e9)

            absent = (pred_present < 0.5)
            if absent.any():
                masks_store[absent] = 0.0
                boxes_store[absent] = 0.0
                scores_store[absent] = -1e9

            all_masks.append(masks_store)
            all_boxes.append(boxes_store)
            all_scores.append(scores_store)

        all_masks  = torch.stack(all_masks,  dim=1)   
        all_boxes  = torch.stack(all_boxes,  dim=1)   
        all_scores = torch.stack(all_scores, dim=1)   
        all_presence = torch.stack(all_presence, dim=1)  

        return all_masks, all_boxes, all_scores, all_presence

  
    def forward_sequence_train(self, imgs, captions, gt_masks):
        
        B, T = imgs.shape[:2]
        device = imgs.device

        lambda_cls  = 2.0
        lambda_box  = 5.0
        lambda_giou = 2.0
        lambda_mask = 1.0

        prev_query = None
        prev_box   = None
        prev_mask  = None
        prev_enc0  = None

        total_loss = 0.0

        all_masks = []
        all_boxes = []
        all_scores = []
        all_presence = []

        batch_idx = torch.arange(B, device=device)

        for t in range(T):
            frame = imgs[:, t]
            gt    = gt_masks[:, t]  

            queries, masks, boxes, class_logits, presence_logit, enc0_used = self.forward_frame(
                frame,
                captions,
                prev_query=prev_query,
                prev_mask=prev_mask,
                prev_enc_lvl0=prev_enc0,
                prev_box=prev_box
            )

            all_presence.append(presence_logit) 

            class_logits_bq = class_logits.squeeze(-1)  
            _, Q, Hf, Wf = masks.shape

            gt_present = (gt.flatten(1).sum(dim=1) > 0).float() 

            gt_ds = F.interpolate(
                gt.unsqueeze(1).float(),
                size=(Hf, Wf),
                mode="nearest"
            ).squeeze(1) 

            gt_boxes = []
            for b in range(B):
                m = gt[b]
                ys, xs = torch.where(m > 0.5)
                if ys.numel() == 0:
                    gt_boxes.append(torch.tensor([0.5, 0.5, 0.0, 0.0], device=device))
                    continue
                H0, W0 = m.shape
                x0 = xs.min().float() / W0
                x1 = (xs.max().float() + 1.0) / W0
                y0 = ys.min().float() / H0
                y1 = (ys.max().float() + 1.0) / H0
                cx = (x0 + x1) * 0.5
                cy = (y0 + y1) * 0.5
                w  = (x1 - x0).clamp(min=0)
                h  = (y1 - y0).clamp(min=0)
                gt_boxes.append(torch.stack([cx, cy, w, h], dim=0))
            gt_boxes = torch.stack(gt_boxes, dim=0) 

            best_idx = torch.zeros(B, dtype=torch.long, device=device)

            loss_t = 0.0

            for b in range(B):
                loss_presence_b = F.binary_cross_entropy_with_logits(
                    presence_logit[b:b+1],
                    gt_present[b:b+1]
                )

              
                if gt_present[b].item() == 0.0:
                    cls_target = torch.zeros_like(class_logits_bq[b:b+1])  
                    loss_cls_b = sigmoid_focal_loss(class_logits_bq[b:b+1], cls_target)

                    empty_penalty = masks[b].sigmoid().mean()  

                    loss_b = (
                        self.lambda_presence * loss_presence_b +
                        lambda_cls * loss_cls_b +
                        self.lambda_empty * empty_penalty
                    )
                    loss_t = loss_t + loss_b
                    best_idx[b] = 0
                    continue




                gt_box_b = gt_boxes[b]  


                l1_cost = (boxes[b] - gt_box_b.unsqueeze(0)).abs().mean(dim=-1) 

                pred_xyxy = box_cxcywh_to_xyxy(boxes[b])  
                tgt_xyxy  = box_cxcywh_to_xyxy(gt_box_b.unsqueeze(0))  
                giou = generalized_box_iou(pred_xyxy, tgt_xyxy).squeeze(1)  
                giou_cost = 1.0 - giou

                mask_cost = mask_focal_dice_loss_per_query(
                    masks[b:b+1],
                    gt_ds[b:b+1]
                ).squeeze(0) 

                cls_pos_cost = sigmoid_focal_loss(
                    class_logits_bq[b:b+1],
                    torch.ones_like(class_logits_bq[b:b+1]),
                    reduction="none"
                ).squeeze(0)  

                match_cost = (
                    lambda_cls  * cls_pos_cost +
                    lambda_box  * l1_cost +
                    lambda_giou * giou_cost +
                    lambda_mask * mask_cost
                )

                qbest = int(match_cost.argmin().item())
                best_idx[b] = qbest


                cls_target = torch.zeros_like(class_logits_bq[b:b+1]) 
                cls_target[0, qbest] = 1.0
                loss_cls_b = sigmoid_focal_loss(class_logits_bq[b:b+1], cls_target)

                pred_box_best = boxes[b, qbest].unsqueeze(0)  
                loss_l1_b = F.l1_loss(pred_box_best, gt_box_b.unsqueeze(0), reduction="mean")

                pred_xyxy_best = box_cxcywh_to_xyxy(pred_box_best)
                tgt_xyxy_best  = box_cxcywh_to_xyxy(gt_box_b.unsqueeze(0))
                giou_best = generalized_box_iou(pred_xyxy_best, tgt_xyxy_best).squeeze()
                loss_giou_b = (1.0 - giou_best)

                pred_mask_best = masks[b:b+1, qbest:qbest+1]  
                loss_mask_b = mask_focal_dice_loss_per_query(
                    pred_mask_best,
                    gt_ds[b:b+1]
                ).mean()

                loss_b = (
                    self.lambda_presence * loss_presence_b +
                    lambda_cls  * loss_cls_b +
                    lambda_box  * loss_l1_b +
                    lambda_giou * loss_giou_b +
                    lambda_mask * loss_mask_b
                )

                loss_t = loss_t + loss_b

            loss_t = loss_t / B
            total_loss = total_loss + loss_t




            best_query = queries[batch_idx, best_idx]   
            best_box   = boxes[batch_idx, best_idx]     
            best_mask  = masks[batch_idx, best_idx]     

            gt_gate_q = gt_present.view(B, 1)
            gt_gate_m = gt_present.view(B, 1, 1)
            gt_gate_b = gt_present.view(B, 1)
            gt_gate_e = gt_present.view(B, 1, 1, 1)

            if self.use_propagation:
                prev_query = self.track_mlp(best_query) * gt_gate_q
                prev_mask  = best_mask * gt_gate_m
                prev_box   = best_box  * gt_gate_b
                prev_enc0  = enc0_used * gt_gate_e
            else:
                prev_query = None
                prev_mask  = None
                prev_box   = None
                prev_enc0  = None

            masks_store  = self._pad_to_num_queries(masks, pad_value=0.0)
            boxes_store  = self._pad_to_num_queries(boxes, pad_value=0.0)
            scores_store = self._pad_to_num_queries(class_logits_bq, pad_value=-1e9)

            all_masks.append(masks_store)
            all_boxes.append(boxes_store)
            all_scores.append(scores_store)

        all_masks  = torch.stack(all_masks,  dim=1)
        all_boxes  = torch.stack(all_boxes,  dim=1)
        all_scores = torch.stack(all_scores, dim=1)
        all_presence = torch.stack(all_presence, dim=1)  

        total_loss = total_loss / T
        return total_loss, all_masks, all_boxes, all_scores, all_presence


