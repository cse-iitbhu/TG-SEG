from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def dice_cost_from_logits(mask_logits: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    prob = torch.sigmoid(mask_logits)
    q = prob.shape[0]
    g = gt_masks.shape[0]
    prob = prob[:, None]
    gt_masks = gt_masks[None]
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * gt_masks).sum(dim=dims)
    denominator = prob.sum(dim=dims) + gt_masks.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def bce_cost_from_logits(mask_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    q = mask_logits.shape[0]
    g = gt_masks.shape[0]
    pred = mask_logits[:, None].expand(q, g, *mask_logits.shape[1:])
    tgt = gt_masks[None].expand(q, g, *gt_masks.shape[1:])
    loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction='none')
    dims = tuple(range(2, loss.ndim))
    return loss.mean(dim=dims)


class HungarianMatcher:
    def __init__(self, lambda_mask: float = 0.7, lambda_cls: float = 0.3):
        self.lambda_mask = lambda_mask
        self.lambda_cls = lambda_cls

    def build_gt(self, target: torch.Tensor, num_foreground_classes: int):
        present_classes = torch.unique(target)
        present_classes = present_classes[present_classes > 0]
        labels = []
        masks = []
        for c in present_classes.tolist():
            labels.append(int(c))
            masks.append((target == c).float())
        if len(labels) == 0:
            return torch.empty(0, dtype=torch.long, device=target.device), torch.empty(0, *target.shape, device=target.device)
        gt_labels = torch.tensor(labels, dtype=torch.long, device=target.device)
        gt_masks = torch.stack(masks, dim=0)
        return gt_labels, gt_masks

    def __call__(self, pred_logits: torch.Tensor, pred_masks: torch.Tensor, target: torch.Tensor):
        gt_labels, gt_masks = self.build_gt(target, pred_logits.shape[-1] - 1)
        q = pred_logits.shape[0]
        if gt_labels.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=target.device),
                torch.empty(0, dtype=torch.long, device=target.device),
                gt_labels,
                gt_masks,
            )
        cls_log_probs = F.log_softmax(pred_logits, dim=-1)
        cls_cost = -cls_log_probs[:, gt_labels]
        mask_cost = bce_cost_from_logits(pred_masks, gt_masks) + dice_cost_from_logits(pred_masks, gt_masks)
        total_cost = self.lambda_mask * mask_cost + self.lambda_cls * cls_cost
        row, col = linear_sum_assignment(total_cost.detach().cpu().numpy())
        row = torch.as_tensor(row, dtype=torch.long, device=target.device)
        col = torch.as_tensor(col, dtype=torch.long, device=target.device)
        return row, col, gt_labels, gt_masks


class MaskClassificationCriterion(nn.Module):
    def __init__(self, lambda_mask: float = 0.7, lambda_cls: float = 0.3, deep_supervision: bool = True):
        super().__init__()
        self.matcher = HungarianMatcher(lambda_mask=lambda_mask, lambda_cls=lambda_cls)
        self.lambda_mask = lambda_mask
        self.lambda_cls = lambda_cls
        self.deep_supervision = deep_supervision

    def _loss_single(self, pred_logits: torch.Tensor, pred_masks: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = pred_logits.device
        total_cls = torch.tensor(0.0, device=device)
        total_mask = torch.tensor(0.0, device=device)
        batch_size = pred_logits.shape[0]

        for b in range(batch_size):
            row, col, gt_labels, gt_masks = self.matcher(pred_logits[b], pred_masks[b], target[b, 0])
            tgt_classes = torch.zeros(pred_logits.shape[1], dtype=torch.long, device=device)
            if row.numel() > 0:
                tgt_classes[row] = gt_labels[col]
                matched_pred_masks = pred_masks[b, row]
                matched_gt_masks = gt_masks[col]
                bce = F.binary_cross_entropy_with_logits(matched_pred_masks, matched_gt_masks)
                dice = dice_cost_from_logits(matched_pred_masks, matched_gt_masks).diag().mean()
                total_mask = total_mask + (bce + dice)
            cls = F.cross_entropy(pred_logits[b], tgt_classes)
            total_cls = total_cls + cls

        total_cls = total_cls / batch_size
        total_mask = total_mask / batch_size
        total = self.lambda_mask * total_mask + self.lambda_cls * total_cls
        return {
            'loss_total': total,
            'loss_cls': total_cls,
            'loss_mask': total_mask,
        }

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = self._loss_single(outputs['pred_logits'], outputs['pred_masks'], target)
        if self.deep_supervision and 'aux_outputs' in outputs and len(outputs['aux_outputs']) > 1:
            aux_total = 0.0
            aux_cls = 0.0
            aux_mask = 0.0
            for aux in outputs['aux_outputs'][:-1]:
                aux_losses = self._loss_single(aux['pred_logits'], aux['pred_masks'], target)
                aux_total += aux_losses['loss_total']
                aux_cls += aux_losses['loss_cls']
                aux_mask += aux_losses['loss_mask']
            n = len(outputs['aux_outputs']) - 1
            losses['loss_total'] = losses['loss_total'] + 0.5 * (aux_total / n)
            losses['loss_cls'] = losses['loss_cls'] + 0.5 * (aux_cls / n)
            losses['loss_mask'] = losses['loss_mask'] + 0.5 * (aux_mask / n)
        return losses
