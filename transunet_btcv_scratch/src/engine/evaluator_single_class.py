from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from monai.inferers import sliding_window_inference


def compute_binary_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    pred_fg = pred == 1
    target_fg = target == 1

    denom = pred_fg.sum() + target_fg.sum()
    if denom == 0:
        return 1.0

    dice = (2.0 * (pred_fg & target_fg).sum()) / denom
    return float(dice)


@torch.no_grad()
def validate_single_class(model, loader, device, cfg):
    model.eval()
    all_scores: List[float] = []
    target_name = str(cfg.data.target_name)

    def _predictor(x):
        return model(x)["semantic_logits"]

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        logits = sliding_window_inference(
            inputs=image,
            roi_size=tuple(cfg.data.crop_size),
            sw_batch_size=int(cfg.eval.sw_batch_size),
            predictor=_predictor,
            overlap=float(cfg.eval.overlap),
            mode="gaussian",
        )

        pred = torch.argmax(logits, dim=1)
        dice = compute_binary_dice(pred[0], label[0, 0])
        all_scores.append(dice)

    mean_dice = float(np.mean(all_scores))
    return {
        target_name: mean_dice,
        "mean_dice": mean_dice,
    }