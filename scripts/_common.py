"""Shared helpers for the Needle 3 finetuning scripts.

Every script takes ``--task`` pointing at a task.toml. All task-specific
paths, quotas, languages, and hyperparameters come from that file, never from
literals in logic (agents2.md Rule 1).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path


def cactus_bin() -> str:
    """Resolve the ``cactus`` CLI deterministically.

    Prefer the binary sitting beside the running interpreter (the venv's
    ``bin/``), so the scripts work without an activated environment;
    otherwise fall back to ``PATH`` lookup.
    """
    sibling = Path(sys.executable).parent / "cactus"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return "cactus"


def kaggle_bin() -> str:
    """Resolve the ``kaggle`` CLI via PATH, dying with the fix if absent."""
    path = shutil.which("kaggle")
    if path is None:
        die("kaggle CLI not on PATH; install with: uv tool install kaggle")
    return path


def add_task_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the required ``--task`` argument to a script's parser."""
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task.toml describing this finetune task.",
    )
    return parser


def load_task_config(task_path: str) -> tuple[dict, Path]:
    """Load task.toml, returning (config, task_dir)."""
    task_file = Path(task_path).resolve()
    if not task_file.is_file():
        die(f"task file not found: {task_path}")
    with open(task_file, "rb") as handle:
        config = tomllib.load(handle)
    return config, task_file.parent


def task_relative_path(task_dir: Path, value: str) -> Path:
    """Resolve a task.toml path value against the task directory."""
    return (task_dir / value).resolve()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def die(message: str, code: int = 1) -> "NoReturn":
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def check_no_overwrite(paths, overwrite: bool) -> None:
    """Die naming every existing output path, unless overwrite is true."""
    if overwrite:
        return
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        die(
            "refusing to overwrite existing outputs: "
            + ", ".join(existing)
            + " (pass --overwrite to replace them)"
        )
