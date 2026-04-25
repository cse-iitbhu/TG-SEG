from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
from monai.transforms import Compose, Lambdad

from src.data.acdc_transforms import get_train_transforms, get_val_transforms
from src.utils.io import load_json


def _resolve_items(manifest_items: List[Dict], ids: List[str]) -> List[Dict]:
    id_set = set(ids)
    return [item for item in manifest_items if item["case_id"] in id_set]


def _binarize_label(x, target_class_id: int):
    if isinstance(x, torch.Tensor):
        return (x == target_class_id).long()
    x = np.asarray(x)
    return (x == target_class_id).astype(np.int64)


def _wrap_binary_label_transform(base_transform, target_class_id: int):
    return Compose(
        [
            base_transform,
            Lambdad(keys=["label"], func=lambda x: _binarize_label(x, target_class_id)),
        ]
    )


class ACDCSingleClassDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_root = Path(cfg.data.dataset_root)
        self.manifest = load_json(self.dataset_root / "manifest.json")
        self.splits = load_json(self.dataset_root / "split_acdc.json")
        self.target_class_id = int(cfg.data.target_class_id)

    def get_train_val_items(self) -> Tuple[List[Dict], List[Dict]]:
        split_name = self.cfg.data.split_name
        split = self.splits[split_name]
        train_items = _resolve_items(self.manifest["training"], split["train"])
        val_items = _resolve_items(self.manifest["training"], split["val"])
        return train_items, val_items

    def build_train_loader(self):
        train_items, _ = self.get_train_val_items()
        ds = CacheDataset(
            data=train_items,
            transform=_wrap_binary_label_transform(
                get_train_transforms(
                    spacing=self.cfg.data.spacing,
                    crop_size=self.cfg.data.crop_size,
                    num_samples=self.cfg.data.num_samples_per_volume,
                ),
                self.target_class_id,
            ),
            cache_rate=float(self.cfg.data.train_cache_rate),
            num_workers=int(self.cfg.data.num_workers),
        )
        return DataLoader(
            ds,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=True,
            collate_fn=list_data_collate,
            persistent_workers=int(self.cfg.data.num_workers) > 0,
        )

    def build_val_loader(self):
        _, val_items = self.get_train_val_items()
        ds = Dataset(
            data=val_items,
            transform=_wrap_binary_label_transform(
                get_val_transforms(self.cfg.data.spacing),
                self.target_class_id,
            ),
        )
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, int(self.cfg.data.num_workers) // 2),
            pin_memory=True,
        )