"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Load a YAML configuration file (defaults to ``configs/default.yaml``).

    A config may set ``defaults: <file>`` (relative to its own directory) to
    inherit from a base config; the current file's keys override the base's.
    This lets experiment configs restate only what differs, so they cannot drift
    out of sync with the shared defaults.
    """
    path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(path, "r") as handle:
        cfg = yaml.safe_load(handle) or {}

    base = cfg.pop("defaults", None)
    if base is not None:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        return merged
    return cfg
