from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resolve_path(path: str, base_dir: Path, root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root if root is not None else base_dir) / candidate


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            if "array" not in value:
                raise ValueError(f"Tensor dictionary {path} must contain an 'array' key.")
            value = value["array"]
        return torch.as_tensor(value).cpu().numpy()
    with Image.open(path) as image:
        return np.asarray(image.copy())


def _as_image_tensor(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D frame, received shape {array.shape}.")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]

    tensor = torch.from_numpy(np.ascontiguousarray(array)).float().permute(2, 0, 1)
    low, high = tensor.amin(), tensor.amax()
    if high > 1.0 or low < 0.0:
        tensor = (tensor - low) / (high - low).clamp_min(1e-6)
    return tensor


def _as_mask_tensor(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[..., 0] if array.shape[-1] <= 4 else array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, received shape {array.shape}.")
    return (torch.from_numpy(np.ascontiguousarray(array)).float() > 0).float()


class SequenceManifestDataset(Dataset):
    """Dataset shared by all ablations.

    Each JSONL row must contain equally sized ``frames`` and ``masks`` lists,
    plus either ``prompts`` (a list) or ``caption`` (a string). Paths may be
    absolute, relative to the manifest, or relative to ``root``.
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path | None = None,
        image_size: int = 512,
        clip_len: int | None = 3,
        training: bool = True,
        augment: bool = True,
    ):
        self.manifest = Path(manifest).expanduser().resolve()
        self.base_dir = self.manifest.parent
        self.root = Path(root).expanduser().resolve() if root else None
        self.image_size = image_size
        self.clip_len = clip_len
        self.training = training
        self.augment = augment and training

        if image_size <= 0 or image_size % 32 != 0:
            raise ValueError("image_size must be a positive multiple of 32.")
        if clip_len is not None and clip_len <= 0:
            raise ValueError("clip_len must be positive or None.")

        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"No samples found in {self.manifest}.")
        for index, record in enumerate(self.records):
            self._validate_record(record, index)

    @staticmethod
    def _validate_record(record: dict[str, Any], index: int) -> None:
        frames = record.get("frames")
        masks = record.get("masks")
        prompts = record.get("prompts")
        caption = record.get("caption")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Record {index} requires a non-empty 'frames' list.")
        if not isinstance(masks, list) or len(masks) != len(frames):
            raise ValueError(f"Record {index} requires one mask per frame.")
        if not prompts and not isinstance(caption, str):
            raise ValueError(f"Record {index} requires 'prompts' or 'caption'.")
        if prompts and (
            not isinstance(prompts, list)
            or not all(isinstance(prompt, str) and prompt for prompt in prompts)
        ):
            raise ValueError(f"Record {index} has an invalid 'prompts' list.")

    def __len__(self) -> int:
        return len(self.records)

    def _indices(self, length: int) -> list[int]:
        if self.clip_len is None:
            return list(range(length))
        if length >= self.clip_len:
            if self.training:
                return sorted(random.sample(range(length), self.clip_len))
            positions = torch.linspace(0, length - 1, self.clip_len)
            return positions.round().long().tolist()
        return list(range(length)) + [length - 1] * (self.clip_len - length)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        indices = self._indices(len(record["frames"]))
        horizontal_flip = self.augment and random.random() < 0.5
        vertical_flip = self.augment and random.random() < 0.5

        frames = []
        masks = []
        for frame_index in indices:
            frame_path = _resolve_path(
                record["frames"][frame_index], self.base_dir, self.root
            )
            mask_path = _resolve_path(
                record["masks"][frame_index], self.base_dir, self.root
            )
            frame = _as_image_tensor(_load_array(frame_path))
            mask = _as_mask_tensor(_load_array(mask_path))

            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = F.interpolate(
                mask[None, None],
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).squeeze(0)

            if horizontal_flip:
                frame = frame.flip(-1)
                mask = mask.flip(-1)
            if vertical_flip:
                frame = frame.flip(-2)
                mask = mask.flip(-2)

            frames.append((frame - IMAGENET_MEAN) / IMAGENET_STD)
            masks.append(mask)

        prompts = record.get("prompts") or [record["caption"]]
        return {
            "images": torch.stack(frames),
            "masks": torch.stack(masks),
            "prompts": list(prompts),
            "sample_id": str(record.get("id", index)),
        }


def collate_sequences(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "masks": torch.stack([sample["masks"] for sample in samples]),
        "prompts": [sample["prompts"] for sample in samples],
        "sample_ids": [sample["sample_id"] for sample in samples],
    }
