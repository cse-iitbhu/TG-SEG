from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tgseg.ablation.configs import ExperimentConfig
from tgseg.ablation.data import SequenceManifestDataset, collate_sequences
from tgseg.ablation.modeling import build_ablation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one TG-SEG ablation checkpoint for Tables 3-5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The manifest uses the same JSONL format as train_ablation.py; "
            "evaluation consumes every frame in each sequence."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Evaluation JSONL manifest.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def frame_scores(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().item()
    prediction_size = prediction.sum().item()
    target_size = target.sum().item()
    if prediction_size == 0 and target_size == 0:
        return 1.0, 1.0
    dice = 2.0 * intersection / max(prediction_size + target_size, 1)
    union = prediction_size + target_size - intersection
    iou = intersection / max(union, 1)
    return dice, iou


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_threshold: float,
) -> dict[str, float | int | str | bool]:
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    foreground_dice_sum = 0.0
    foreground_iou_sum = 0.0
    frames = 0
    foreground_frames = 0
    pixel_false_negatives = 0
    pixel_true_positives = 0
    presence_tp = presence_fp = presence_tn = presence_fn = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        ground_truth_masks = batch["masks"].to(device, non_blocking=True)
        output = model.forward_sequence(images, batch["prompts"])
        batch_size, frame_count = output["masks"].shape[:2]

        for batch_index in range(batch_size):
            for frame_index in range(frame_count):
                best_query = output["scores"][batch_index, frame_index].argmax()
                logits = output["masks"][
                    batch_index, frame_index, best_query
                ]
                target = ground_truth_masks[batch_index, frame_index]
                target = F.interpolate(
                    target[None, None].float(),
                    size=logits.shape[-2:],
                    mode="nearest",
                ).squeeze(0).squeeze(0) > 0.5
                prediction = logits.sigmoid() >= mask_threshold

                dice, iou = frame_scores(prediction, target)
                dice_sum += dice
                iou_sum += iou
                frames += 1

                target_present = bool(target.any().item())
                predicted_present = bool(
                    (
                        output["presence_logits"][batch_index, frame_index].sigmoid()
                        >= model.presence_tau
                    ).item()
                )
                if predicted_present and target_present:
                    presence_tp += 1
                elif predicted_present:
                    presence_fp += 1
                elif target_present:
                    presence_fn += 1
                else:
                    presence_tn += 1

                if target_present:
                    foreground_dice_sum += dice
                    foreground_iou_sum += iou
                    foreground_frames += 1
                    pixel_true_positives += (prediction & target).sum().item()
                    pixel_false_negatives += ((~prediction) & target).sum().item()

    presence_precision = presence_tp / max(presence_tp + presence_fp, 1)
    presence_recall = presence_tp / max(presence_tp + presence_fn, 1)
    presence_f1 = (
        2.0 * presence_precision * presence_recall
        / max(presence_precision + presence_recall, 1e-12)
    )
    return {
        "experiment": model.config.name,
        "vision_encoder": model.config.vision_encoder,
        "text_encoder": model.config.text_encoder,
        "presence_enabled": model.config.use_presence,
        "dice": 100.0 * dice_sum / max(frames, 1),
        "iou": 100.0 * iou_sum / max(frames, 1),
        "foreground_dice": 100.0 * foreground_dice_sum / max(foreground_frames, 1),
        "foreground_iou": 100.0 * foreground_iou_sum / max(foreground_frames, 1),
        "fnr": 100.0
        * pixel_false_negatives
        / max(pixel_true_positives + pixel_false_negatives, 1),
        "presence_accuracy": (
            presence_tp + presence_tn
        ) / max(presence_tp + presence_fp + presence_tn + presence_fn, 1),
        "presence_precision": presence_precision,
        "presence_recall": presence_recall,
        "presence_f1": presence_f1,
        "frames": frames,
        "foreground_frames": foreground_frames,
        "presence_tp": presence_tp,
        "presence_fp": presence_fp,
        "presence_tn": presence_tn,
        "presence_fn": presence_fn,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "experiment" not in checkpoint or "model" not in checkpoint:
        raise ValueError(
            "Checkpoint must be produced by train_ablation.py and include "
            "'experiment' and 'model'."
        )
    config = ExperimentConfig(**checkpoint["experiment"])
    image_size = args.image_size or int(checkpoint.get("image_size", 512))

    dataset = SequenceManifestDataset(
        manifest=args.manifest,
        root=args.data_root,
        image_size=image_size,
        clip_len=None,
        training=False,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )
    model = build_ablation_model(
        config,
        pretrained_vision=False,
        pretrained_text=False,
        image_size=image_size,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    metrics = evaluate(model, loader, device, args.mask_threshold)
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
