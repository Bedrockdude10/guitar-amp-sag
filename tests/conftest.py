"""Shared pytest fixtures and path setup for the power_sag test suite."""

import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def config():
    from power_sag import load_config

    return load_config()
