"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Load a YAML configuration file (defaults to ``configs/default.yaml``)."""
    path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(path, "r") as handle:
        return yaml.safe_load(handle)
