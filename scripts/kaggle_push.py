#!/usr/bin/env python3
"""Render the kernel bundle and push a GPU training run.

Stages the training driver plus a rendered ``kernel-metadata.json``
(GPU on, internet on for pip + HuggingFace, input dataset attached,
private) and runs ``kaggle kernels push`` with the configured timeout.
``--dry-run`` stages files and prints the argv without pushing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common
import _kaggle

DRIVER_NAME = "kaggle_train_driver.py"

# Metadata booleans are strings in both official sources (the documented
# example and the `kaggle kernels init` template), so render strings.
TRUE = "true"


def kernel_ref(config: dict) -> str:
    """Full ``owner/slug`` kernel ref from the ``[kaggle]`` section."""
    kaggle = config["kaggle"]
    return f"{kaggle['username']}/{kaggle['kernel']}"


def render_kernel_metadata(config: dict) -> dict:
    """Render ``kernel-metadata.json`` for a GPU training run."""
    kaggle = config["kaggle"]
    return {
        "id": kernel_ref(config),
        "title": _kaggle.title_from_slug(kaggle["kernel"]),
        "code_file": DRIVER_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": TRUE,
        "enable_gpu": TRUE,
        "enable_internet": TRUE,
        "machine_shape": kaggle["accelerator"],
        "dataset_sources": [_kaggle.dataset_ref(config, "input_dataset")],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def stage_kernel(config: dict, task_dir: Path, staging_dir: Path) -> Path:
    """Copy the driver and write metadata into the kernel staging dir."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    driver = task_dir.parent.parent / "scripts" / DRIVER_NAME
    shutil.copy2(driver, staging_dir / DRIVER_NAME)
    with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as handle:
        json.dump(render_kernel_metadata(config), handle, indent=2)
    return staging_dir


def build_push_argv(kaggle: str, staging_dir: Path, timeout: int) -> list[str]:
    """Build the ``kaggle kernels push`` argv."""
    return [
        kaggle, "kernels", "push",
        "-p", str(staging_dir), "-t", str(timeout),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files and print the kaggle argv without executing it.",
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    kaggle = _common.kaggle_bin()
    staging = _common.task_relative_path(
        task_dir, config["task"]["out_dir"]
    ) / "staging_kernel"
    stage_kernel(config, task_dir, staging)
    argv = build_push_argv(kaggle, staging, config["kaggle"]["timeout"])
    print(f"kernel {kernel_ref(config)}")
    return _kaggle.run_or_print(argv, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
