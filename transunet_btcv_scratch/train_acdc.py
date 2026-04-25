from __future__ import annotations

import argparse

import torch

from src.data.acdc_dataset import ACDCDataModule
from src.engine.trainer import Trainer
from src.losses.dice import SegmentationLoss
from src.losses.hungarian import MaskClassificationCriterion
from src.models.transunet3d import TransUNet3D
from src.utils.config import load_config
from src.utils.seed import seed_everything


def build_optimizer(model, cfg):
    name = cfg.train.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.train.lr),
            momentum=float(cfg.train.momentum),
            weight_decay=float(cfg.train.weight_decay),
            nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def build_scheduler(optimizer, cfg):
    name = cfg.train.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
            eta_min=float(cfg.train.min_lr),
        )
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = ACDCDataModule(cfg)
    train_loader = data.build_train_loader()
    val_loader = data.build_val_loader()

    model = TransUNet3D(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    if cfg.model.mode == "encoder_only":
        criterion = SegmentationLoss(
            ce_weight=float(cfg.loss.ce_weight),
            dice_weight=float(cfg.loss.dice_weight),
        )
    else:
        criterion = MaskClassificationCriterion(
            lambda_mask=float(cfg.loss.lambda_mask),
            lambda_cls=float(cfg.loss.lambda_cls),
            deep_supervision=bool(cfg.loss.deep_supervision),
        )

    trainer = Trainer(model, optimizer, scheduler, criterion, train_loader, val_loader, device, cfg)
    trainer.fit()


if __name__ == "__main__":
    main()