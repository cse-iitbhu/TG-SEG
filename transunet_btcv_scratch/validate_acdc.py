from __future__ import annotations

import argparse

import torch

from src.data.acdc_dataset import ACDCDataModule
from src.engine.evaluator import validate
from src.models.transunet3d import TransUNet3D
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransUNet3D(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    data = ACDCDataModule(cfg)
    val_loader = data.build_val_loader()
    metrics = validate(model, val_loader, device, cfg)
    print(metrics)


if __name__ == "__main__":
    main()