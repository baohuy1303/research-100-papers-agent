"""Shared helpers for repository scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"


def safe_print(msg: str) -> None:
    """Print safely in Windows consoles configured with cp1252."""
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
