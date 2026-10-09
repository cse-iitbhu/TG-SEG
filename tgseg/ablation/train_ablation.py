from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tgseg.ablation.configs import EXPERIMENTS, get_experiment
from tgseg.ablation.data import SequenceManifestDataset, collate_sequences
from tgseg.ablation.modeling import build_ablation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one TG-SEG encoder or presence-head ablation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Manifest rows are JSON objects such as:\n"
            '{"id":"case-1","frames":["images/0.png"],'
            '"masks":["masks/0.png"],"prompts":["The spleen ..."]}'
        ),
    )
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--manifest", required=True, help="Training JSONL manifest.")
    parser.add_argument("--data-root", default=None, help="Optional base for relative paths.")
    parser.add_argument("--output-dir", default="checkpoints/ablations")
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--clip-len", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrained vision and text weights.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    experiment_config: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "experiment": experiment_config,
            "dataset_name": args.dataset_name,
            "image_size": args.image_size,
            "clip_len": args.clip_len,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    config = get_experiment(args.experiment)

    dataset = SequenceManifestDataset(
        manifest=args.manifest,
        root=args.data_root,
        image_size=args.image_size,
        clip_len=args.clip_len,
        training=True,
        augment=args.augment,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
        generator=generator,
    )

    model = build_ablation_model(
        config,
        pretrained_vision=args.pretrained,
        pretrained_text=args.pretrained,
        image_size=args.image_size,
    ).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3],
        gamma=0.1,
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = checkpoint.get("experiment")
        if saved_config is not None and saved_config != config.to_dict():
            raise ValueError("The resume checkpoint belongs to a different experiment.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    output_dir = Path(args.output_dir) / args.dataset_name / config.name
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model.forward_sequence_train(
                images,
                batch["prompts"],
                masks,
            )
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, step {step}: {loss.item()}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            running_loss += loss.detach().item()

            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"step={step}/{len(loader)} loss={loss.item():.6f}",
                flush=True,
            )

        scheduler.step()
        mean_loss = running_loss / max(len(loader), 1)
        print(
            f"epoch={epoch + 1} mean_loss={mean_loss:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            config.to_dict(),
            args,
        )


if __name__ == "__main__":
    main()
