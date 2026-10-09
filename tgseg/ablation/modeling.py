from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import ExperimentConfig
from .encoders import FPNProjection, build_text_encoder, build_vision_encoder


# The original architecture modules are reused without editing them. Their
# imports are flat (for example, deformable_transformer imports ms_deform_attn),
# so the existing src directory must be importable as a top-level module path.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou  # noqa: E402
from deformable_transformer import DeformableTransformer  # noqa: E402
from fusion import CrossModalFusion  # noqa: E402
from losses import mask_focal_dice_loss_per_query, sigmoid_focal_loss  # noqa: E402
from mask_head import PaperDynamicMaskHead  # noqa: E402
from memory_read import MaskMemoryRead  # noqa: E402
from positional_encoding import PositionEmbeddingSine  # noqa: E402


class TGSEGAblationModel(nn.Module):
    """TG-SEG with independently selectable encoders and presence modeling."""

    def __init__(
        self,
        config: ExperimentConfig,
        pretrained_vision: bool = True,
        pretrained_text: bool = True,
        image_size: int = 512,
    ):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.backbone = build_vision_encoder(
            config.vision_encoder,
            pretrained=pretrained_vision,
            image_size=image_size,
        )
        self.fpn = FPNProjection(self.backbone.out_channels, hidden_dim)
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            normalize=True,
        )
        self.text_encoder = build_text_encoder(
            config.text_encoder,
            hidden_dim=hidden_dim,
            freeze=config.freeze_text_encoder,
            pretrained=pretrained_text,
        )
        self.fusion = CrossModalFusion(hidden_dim=hidden_dim)
        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            num_queries=config.num_queries,
            n_levels=3,
            n_heads=8,
            n_points=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_ffn=hidden_dim * 4,
            dropout=0.1,
        )
        self.mask_head = PaperDynamicMaskHead(
            hidden_dim=hidden_dim,
            dyn_channels=8,
            num_layers=3,
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.presence_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            if config.use_presence
            else None
        )
        self.track_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_read = MaskMemoryRead(hidden_dim)

        self.hidden_dim = hidden_dim
        self.num_queries = config.num_queries
        self.use_presence = config.use_presence
        self.use_propagation = config.use_propagation
        self.presence_tau = config.presence_tau
        self.lambda_presence = config.lambda_presence
        self.lambda_empty = config.lambda_empty

    @staticmethod
    def _normalize_prompts(
        captions: Sequence[Sequence[str]] | Sequence[str], batch_size: int
    ) -> list[list[str]]:
        if len(captions) == batch_size and all(
            isinstance(item, str) for item in captions
        ):
            return [[str(item)] for item in captions]
        if len(captions) == batch_size and all(
            isinstance(item, (list, tuple)) for item in captions
        ):
            prompts = [list(item) for item in captions]
        elif batch_size == 1 and all(isinstance(item, str) for item in captions):
            prompts = [list(captions)]
        else:
            raise ValueError(
                "captions must contain one prompt list per batch element."
            )
        if not prompts[0] or any(len(item) != len(prompts[0]) for item in prompts):
            raise ValueError("Every sample must have the same non-zero prompt count.")
        return prompts

    def _pad_queries(self, value: torch.Tensor, pad_value: float) -> torch.Tensor:
        query_count = value.shape[1]
        if query_count == self.num_queries:
            return value
        if query_count > self.num_queries:
            raise RuntimeError(
                f"Received {query_count} queries, configured for {self.num_queries}."
            )
        pad_shape = list(value.shape)
        pad_shape[1] = self.num_queries - query_count
        padding = value.new_full(pad_shape, pad_value)
        return torch.cat([value, padding], dim=1)

    @staticmethod
    def _best_query(class_logits: torch.Tensor) -> torch.Tensor:
        return class_logits.squeeze(-1).argmax(dim=1)

    def forward_frame(
        self,
        image: torch.Tensor,
        captions: Sequence[Sequence[str]] | Sequence[str],
        prev_query: torch.Tensor | None = None,
        prev_mask: torch.Tensor | None = None,
        prev_encoder_feature: torch.Tensor | None = None,
        prev_box: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = image.shape[0]
        prompt_lists = self._normalize_prompts(captions, batch_size)

        features = self.fpn(self.backbone(image))
        feature_shapes = []
        vision_tokens = []
        for feature in features:
            _, _, height, width = feature.shape
            feature_shapes.append((height, width))
            position = self.position_embedding(feature)
            tokens = feature.flatten(2).transpose(1, 2)
            position_tokens = position.flatten(2).transpose(1, 2)
            vision_tokens.append(tokens + position_tokens)

        word_features = []
        sentence_features = []
        padding_masks = []
        for prompt_index in range(len(prompt_lists[0])):
            prompt_batch = [prompts[prompt_index] for prompts in prompt_lists]
            words, sentence, padding = self.text_encoder(
                prompt_batch, image.device
            )
            word_features.append(words)
            sentence_features.append(sentence)
            padding_masks.append(padding)

        fused_tokens, prompt_weights = self.fusion(
            vision_tokens,
            word_features,
            padding_masks,
        )
        selected_prompt = prompt_weights[:, 0].argmax(dim=1)
        selected_sentence = torch.stack(
            [
                sentence_features[selected_prompt[b].item()][b]
                for b in range(batch_size)
            ]
        )

        transformer_tokens = [fused_tokens[1], fused_tokens[2], fused_tokens[3]]
        transformer_shapes = [feature_shapes[1], feature_shapes[2], feature_shapes[3]]
        memory, spatial_shapes, level_start_index = self.transformer.encode(
            transformer_tokens,
            transformer_shapes,
        )

        height, width = transformer_shapes[0]
        start = int(level_start_index[0].item())
        end = start + height * width
        encoder_map = memory[:, start:end].transpose(1, 2).reshape(
            batch_size, self.hidden_dim, height, width
        )
        encoder_map_used = encoder_map

        if prev_mask is not None and prev_encoder_feature is not None:
            resized_mask = F.interpolate(
                prev_mask.unsqueeze(1),
                size=(height, width),
                mode="nearest",
            ).squeeze(1)
            encoder_map_used = self.memory_read(
                curr_feat=encoder_map,
                prev_feat=prev_encoder_feature,
                prev_mask=resized_mask,
            )
            memory = memory.clone()
            memory[:, start:end] = encoder_map_used.flatten(2).transpose(1, 2)

        queries = self.transformer.decode(
            memory,
            spatial_shapes,
            level_start_index,
            prev_query=prev_query,
            text_query=selected_sentence,
            use_propagation=self.use_propagation,
        )
        masks = self.mask_head(queries, features)
        box_offsets = self.box_head(queries)
        class_logits = self.class_head(queries)
        if prev_box is None:
            boxes = box_offsets.sigmoid()
        else:
            boxes = (prev_box.unsqueeze(1) + box_offsets).sigmoid()

        if self.presence_head is None:
            presence_logits = image.new_full((batch_size,), 20.0)
        else:
            pooled_queries = queries.max(dim=1).values
            presence_logits = self.presence_head(pooled_queries).squeeze(-1)

        return {
            "queries": queries,
            "masks": masks,
            "boxes": boxes,
            "class_logits": class_logits,
            "presence_logits": presence_logits,
            "encoder_feature": encoder_map_used,
        }

    def forward_sequence(
        self,
        images: torch.Tensor,
        captions: Sequence[Sequence[str]] | Sequence[str],
    ) -> dict[str, torch.Tensor]:
        batch_size, frame_count = images.shape[:2]
        batch_indices = torch.arange(batch_size, device=images.device)
        prev_query = prev_mask = prev_encoder = prev_box = None
        sequence_masks = []
        sequence_boxes = []
        sequence_scores = []
        sequence_presence = []

        for frame_index in range(frame_count):
            output = self.forward_frame(
                images[:, frame_index],
                captions,
                prev_query=prev_query,
                prev_mask=prev_mask,
                prev_encoder_feature=prev_encoder,
                prev_box=prev_box,
            )
            presence_logits = output["presence_logits"]
            if self.use_presence:
                present = (presence_logits.sigmoid() >= self.presence_tau).float()
            else:
                present = torch.ones_like(presence_logits)

            best = self._best_query(output["class_logits"])
            best_query = output["queries"][batch_indices, best]
            best_mask = output["masks"][batch_indices, best]
            best_box = output["boxes"][batch_indices, best]

            if self.use_propagation:
                prev_query = (
                    self.track_mlp(best_query) * present.view(batch_size, 1)
                ).detach()
                prev_mask = (
                    best_mask * present.view(batch_size, 1, 1)
                ).detach()
                prev_box = (best_box * present.view(batch_size, 1)).detach()
                prev_encoder = (
                    output["encoder_feature"]
                    * present.view(batch_size, 1, 1, 1)
                ).detach()
            else:
                prev_query = prev_mask = prev_encoder = prev_box = None

            masks = self._pad_queries(output["masks"], pad_value=-20.0)
            boxes = self._pad_queries(output["boxes"], pad_value=0.0)
            scores = self._pad_queries(
                output["class_logits"].squeeze(-1), pad_value=-1e9
            )
            absent = present < 0.5
            if absent.any():
                masks = masks.clone()
                boxes = boxes.clone()
                scores = scores.clone()
                masks[absent] = -20.0
                boxes[absent] = 0.0
                scores[absent] = -1e9

            sequence_masks.append(masks)
            sequence_boxes.append(boxes)
            sequence_scores.append(scores)
            sequence_presence.append(presence_logits)

        return {
            "masks": torch.stack(sequence_masks, dim=1),
            "boxes": torch.stack(sequence_boxes, dim=1),
            "scores": torch.stack(sequence_scores, dim=1),
            "presence_logits": torch.stack(sequence_presence, dim=1),
        }

    def forward_sequence_train(
        self,
        images: torch.Tensor,
        captions: Sequence[Sequence[str]] | Sequence[str],
        ground_truth_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, frame_count = images.shape[:2]
        batch_indices = torch.arange(batch_size, device=images.device)
        prev_query = prev_mask = prev_encoder = prev_box = None
        total_loss = images.new_zeros(())
        sequence_masks = []
        sequence_boxes = []
        sequence_scores = []
        sequence_presence = []

        lambda_class = 2.0
        lambda_box = 5.0
        lambda_giou = 2.0
        lambda_mask = 1.0

        for frame_index in range(frame_count):
            ground_truth = ground_truth_masks[:, frame_index]
            output = self.forward_frame(
                images[:, frame_index],
                captions,
                prev_query=prev_query,
                prev_mask=prev_mask,
                prev_encoder_feature=prev_encoder,
                prev_box=prev_box,
            )
            class_logits = output["class_logits"].squeeze(-1)
            masks = output["masks"]
            boxes = output["boxes"]
            _, _, mask_height, mask_width = masks.shape

            ground_truth_present = (
                ground_truth.flatten(1).sum(dim=1) > 0
            ).float()
            resized_ground_truth = F.interpolate(
                ground_truth.unsqueeze(1).float(),
                size=(mask_height, mask_width),
                mode="nearest",
            ).squeeze(1)
            ground_truth_boxes = self._boxes_from_masks(ground_truth)

            best_indices = torch.zeros(
                batch_size, dtype=torch.long, device=images.device
            )
            frame_loss = images.new_zeros(())
            for batch_index in range(batch_size):
                if self.use_presence:
                    presence_loss = F.binary_cross_entropy_with_logits(
                        output["presence_logits"][batch_index : batch_index + 1],
                        ground_truth_present[batch_index : batch_index + 1],
                    )
                else:
                    presence_loss = images.new_zeros(())

                if ground_truth_present[batch_index].item() == 0.0:
                    class_target = torch.zeros_like(
                        class_logits[batch_index : batch_index + 1]
                    )
                    class_loss = sigmoid_focal_loss(
                        class_logits[batch_index : batch_index + 1], class_target
                    )
                    empty_penalty = masks[batch_index].sigmoid().mean()
                    frame_loss = frame_loss + (
                        self.lambda_presence * presence_loss
                        + lambda_class * class_loss
                        + self.lambda_empty * empty_penalty
                    )
                    best_indices[batch_index] = class_logits[batch_index].argmax()
                    continue

                target_box = ground_truth_boxes[batch_index]
                l1_cost = (boxes[batch_index] - target_box.unsqueeze(0)).abs().mean(-1)
                predicted_xyxy = box_cxcywh_to_xyxy(boxes[batch_index])
                target_xyxy = box_cxcywh_to_xyxy(target_box.unsqueeze(0))
                giou_cost = 1.0 - generalized_box_iou(
                    predicted_xyxy, target_xyxy
                ).squeeze(1)
                mask_cost = mask_focal_dice_loss_per_query(
                    masks[batch_index : batch_index + 1],
                    resized_ground_truth[batch_index : batch_index + 1],
                ).squeeze(0)
                positive_class_cost = sigmoid_focal_loss(
                    class_logits[batch_index : batch_index + 1],
                    torch.ones_like(class_logits[batch_index : batch_index + 1]),
                    reduction="none",
                ).squeeze(0)
                matching_cost = (
                    lambda_class * positive_class_cost
                    + lambda_box * l1_cost
                    + lambda_giou * giou_cost
                    + lambda_mask * mask_cost
                )
                best_query = int(matching_cost.argmin().item())
                best_indices[batch_index] = best_query

                class_target = torch.zeros_like(
                    class_logits[batch_index : batch_index + 1]
                )
                class_target[0, best_query] = 1.0
                class_loss = sigmoid_focal_loss(
                    class_logits[batch_index : batch_index + 1], class_target
                )
                predicted_box = boxes[batch_index, best_query].unsqueeze(0)
                l1_loss = F.l1_loss(
                    predicted_box, target_box.unsqueeze(0), reduction="mean"
                )
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(predicted_box), target_xyxy
                ).squeeze()
                mask_loss = mask_focal_dice_loss_per_query(
                    masks[
                        batch_index : batch_index + 1,
                        best_query : best_query + 1,
                    ],
                    resized_ground_truth[batch_index : batch_index + 1],
                ).mean()
                frame_loss = frame_loss + (
                    self.lambda_presence * presence_loss
                    + lambda_class * class_loss
                    + lambda_box * l1_loss
                    + lambda_giou * (1.0 - giou)
                    + lambda_mask * mask_loss
                )

            total_loss = total_loss + frame_loss / batch_size

            best_query = output["queries"][batch_indices, best_indices]
            best_mask = masks[batch_indices, best_indices]
            best_box = boxes[batch_indices, best_indices]
            if self.use_presence:
                propagation_gate = ground_truth_present
            else:
                propagation_gate = torch.ones_like(ground_truth_present)

            if self.use_propagation:
                prev_query = self.track_mlp(best_query) * propagation_gate.view(
                    batch_size, 1
                )
                prev_mask = best_mask * propagation_gate.view(batch_size, 1, 1)
                prev_box = best_box * propagation_gate.view(batch_size, 1)
                prev_encoder = output["encoder_feature"] * propagation_gate.view(
                    batch_size, 1, 1, 1
                )
            else:
                prev_query = prev_mask = prev_encoder = prev_box = None

            sequence_masks.append(self._pad_queries(masks, pad_value=-20.0))
            sequence_boxes.append(self._pad_queries(boxes, pad_value=0.0))
            sequence_scores.append(
                self._pad_queries(class_logits, pad_value=-1e9)
            )
            sequence_presence.append(output["presence_logits"])

        return {
            "loss": total_loss / frame_count,
            "masks": torch.stack(sequence_masks, dim=1),
            "boxes": torch.stack(sequence_boxes, dim=1),
            "scores": torch.stack(sequence_scores, dim=1),
            "presence_logits": torch.stack(sequence_presence, dim=1),
        }

    @staticmethod
    def _boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
        boxes = []
        for mask in masks:
            y, x = torch.where(mask > 0.5)
            if y.numel() == 0:
                boxes.append(mask.new_tensor([0.5, 0.5, 0.0, 0.0]))
                continue
            height, width = mask.shape
            x0 = x.min().float() / width
            x1 = (x.max().float() + 1.0) / width
            y0 = y.min().float() / height
            y1 = (y.max().float() + 1.0) / height
            boxes.append(
                torch.stack(
                    [
                        (x0 + x1) * 0.5,
                        (y0 + y1) * 0.5,
                        (x1 - x0).clamp_min(0),
                        (y1 - y0).clamp_min(0),
                    ]
                )
            )
        return torch.stack(boxes)


def build_ablation_model(
    config: ExperimentConfig,
    pretrained_vision: bool = True,
    pretrained_text: bool = True,
    image_size: int = 512,
) -> TGSEGAblationModel:
    return TGSEGAblationModel(
        config,
        pretrained_vision=pretrained_vision,
        pretrained_text=pretrained_text,
        image_size=image_size,
    )
