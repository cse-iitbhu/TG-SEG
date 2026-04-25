from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target = target.long().squeeze(1)
        target_1h = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        if not self.include_background:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]
        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target_1h).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_1h.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss(include_background=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long().squeeze(1))
        dice = self.dice(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice
