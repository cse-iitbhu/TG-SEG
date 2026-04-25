from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src.data.constants import BTCV_LABELS


def get_label_map(cfg) -> Dict[int, str]:
    manifest_path = Path(cfg.data.dataset_root) / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        raw_labels = manifest.get("labels", {})
        if raw_labels:
            return {int(k): str(v) for k, v in raw_labels.items()}

    return BTCV_LABELS


def compute_per_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    label_map: Dict[int, str],
) -> Dict[str, float]:
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    results = {}
    dices = []

    for c in range(1, num_classes):
        pred_c = pred == c
        target_c = target == c
        denom = pred_c.sum() + target_c.sum()

        if denom == 0:
            dice = 1.0
        else:
            dice = (2.0 * (pred_c & target_c).sum()) / denom

        results[label_map.get(c, f"class_{c}")] = float(dice)
        dices.append(dice)

    results["mean_dice"] = float(np.mean(dices))
    return results


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    all_scores: List[Dict[str, float]] = []
    label_map = get_label_map(cfg)

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
        score = compute_per_class_dice(
            pred[0],
            label[0, 0],
            int(cfg.model.num_classes),
            label_map,
        )
        all_scores.append(score)

    keys = list(all_scores[0].keys())
    avg = {k: float(np.mean([s[k] for s in all_scores])) for k in keys}
    return avg
    
    # from __future__ import annotations

# from typing import Dict, List

# import numpy as np
# import torch
# from monai.inferers import sliding_window_inference

# from src.data.constants import BTCV_LABELS


# def compute_per_class_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
#     pred = pred.cpu().numpy()
#     target = target.cpu().numpy()
#     results = {}
#     dices = []
#     for c in range(1, num_classes):
#         pred_c = pred == c
#         target_c = target == c
#         denom = pred_c.sum() + target_c.sum()
#         if denom == 0:
#             dice = 1.0
#         else:
#             dice = (2.0 * (pred_c & target_c).sum()) / denom
#         results[BTCV_LABELS[c]] = float(dice)
#         dices.append(dice)
#     results['mean_dice'] = float(np.mean(dices))
#     return results


# @torch.no_grad()
# def validate(model, loader, device, cfg):
#     model.eval()
#     all_scores: List[Dict[str, float]] = []

#     def _predictor(x):
#         return model(x)['semantic_logits']

#     for batch in loader:
#         image = batch['image'].to(device)
#         label = batch['label'].to(device)
#         logits = sliding_window_inference(
#             inputs=image,
#             roi_size=tuple(cfg.data.crop_size),
#             sw_batch_size=int(cfg.eval.sw_batch_size),
#             predictor=_predictor,
#             overlap=float(cfg.eval.overlap),
#             mode='gaussian',
#         )
#         pred = torch.argmax(logits, dim=1)
#         score = compute_per_class_dice(pred[0], label[0, 0], int(cfg.model.num_classes))
#         all_scores.append(score)

#     keys = list(all_scores[0].keys())
#     avg = {k: float(np.mean([s[k] for s in all_scores])) for k in keys}
#     return avg
