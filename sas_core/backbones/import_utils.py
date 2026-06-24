"""Import helpers for third-party modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def import_from_file(module_name: str, path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

