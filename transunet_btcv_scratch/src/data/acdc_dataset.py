from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate

from src.data.acdc_transforms import get_test_transforms, get_train_transforms, get_val_transforms
from src.utils.io import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_manifest_paths(items: List[Dict]) -> List[Dict]:
    resolved = []
    for item in items:
        new_item = dict(item)
        new_item["image"] = str((PROJECT_ROOT / item["image"]).resolve())
        if "label" in item:
            new_item["label"] = str((PROJECT_ROOT / item["label"]).resolve())
        resolved.append(new_item)
    return resolved


def _resolve_items(manifest_items: List[Dict], ids: List[str]) -> List[Dict]:
    id_set = set(ids)
    selected = [item for item in manifest_items if item["case_id"] in id_set]
    return _resolve_manifest_paths(selected)


class ACDCDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_root = Path(cfg.data.dataset_root)
        self.manifest = load_json(self.dataset_root / "manifest.json")
        self.splits = load_json(self.dataset_root / "split_acdc.json")

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
            transform=get_train_transforms(
                spacing=self.cfg.data.spacing,
                crop_size=self.cfg.data.crop_size,
                num_samples=self.cfg.data.num_samples_per_volume,
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
        ds = Dataset(data=val_items, transform=get_val_transforms(self.cfg.data.spacing))
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, int(self.cfg.data.num_workers) // 2),
            pin_memory=True,
        )

    def build_test_loader(self):
        test_items = _resolve_manifest_paths(self.manifest["testing"])
        ds = Dataset(data=test_items, transform=get_test_transforms(self.cfg.data.spacing))
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    
    
    
    
    
    
# from __future__ import annotations

# from pathlib import Path
# from typing import Dict, List, Tuple

# from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate

# from src.data.acdc_transforms import get_test_transforms, get_train_transforms, get_val_transforms
# from src.utils.io import load_json


# def _resolve_items(manifest_items: List[Dict], ids: List[str]) -> List[Dict]:
#     id_set = set(ids)
#     return [item for item in manifest_items if item["case_id"] in id_set]


# class ACDCDataModule:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.dataset_root = Path(cfg.data.dataset_root)
#         self.manifest = load_json(self.dataset_root / "manifest.json")
#         self.splits = load_json(self.dataset_root / "split_acdc.json")

#     def get_train_val_items(self) -> Tuple[List[Dict], List[Dict]]:
#         split_name = self.cfg.data.split_name
#         split = self.splits[split_name]
#         train_items = _resolve_items(self.manifest["training"], split["train"])
#         val_items = _resolve_items(self.manifest["training"], split["val"])
#         return train_items, val_items

#     def build_train_loader(self):
#         train_items, _ = self.get_train_val_items()
#         ds = CacheDataset(
#             data=train_items,
#             transform=get_train_transforms(
#                 spacing=self.cfg.data.spacing,
#                 crop_size=self.cfg.data.crop_size,
#                 num_samples=self.cfg.data.num_samples_per_volume,
#             ),
#             cache_rate=float(self.cfg.data.train_cache_rate),
#             num_workers=int(self.cfg.data.num_workers),
#         )
#         return DataLoader(
#             ds,
#             batch_size=int(self.cfg.train.batch_size),
#             shuffle=True,
#             num_workers=int(self.cfg.data.num_workers),
#             pin_memory=True,
#             collate_fn=list_data_collate,
#             persistent_workers=int(self.cfg.data.num_workers) > 0,
#         )

#     def build_val_loader(self):
#         _, val_items = self.get_train_val_items()
#         ds = Dataset(data=val_items, transform=get_val_transforms(self.cfg.data.spacing))
#         return DataLoader(
#             ds,
#             batch_size=1,
#             shuffle=False,
#             num_workers=max(1, int(self.cfg.data.num_workers) // 2),
#             pin_memory=True,
#         )

#     def build_test_loader(self):
#         ds = Dataset(data=self.manifest["testing"], transform=get_test_transforms(self.cfg.data.spacing))
#         return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)