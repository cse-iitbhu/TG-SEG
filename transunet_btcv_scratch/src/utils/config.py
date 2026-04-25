from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cfg = _to_namespace(data)
    cfg.config_path = str(path.resolve())
    return cfg


def save_config(cfg: SimpleNamespace, path: str | Path) -> None:
    path = Path(path)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False)
