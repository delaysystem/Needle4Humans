#!/usr/bin/env python3
"""Create or version the task's Kaggle input dataset.

Stages task.toml, tools.json, slot_banks.json, the seed banks, and the
pipeline scripts into a folder with dataset-metadata.json, then creates
the dataset or cuts a new version depending on whether the ref already
exists. Datasets stay private. ``--dry-run`` prints without calling Kaggle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common
import _kaggle

TASK_FILES = ("task.toml", "tools.json", "slot_banks.json")


def stage_input_dataset(
    config: dict, task_dir: Path, staging_dir: Path
) -> Path:
    """Copy task bundle + scripts into the dataset staging dir."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name in TASK_FILES:
        shutil.copy2(task_dir / name, staging_dir / name)
    for key in ("seed_bank", "eval_bank"):
        shutil.copy2(task_dir / config["task"][key], staging_dir)
    scripts_dir = task_dir.parent.parent / "scripts"
    for script in sorted(scripts_dir.glob("*.py")):
        shutil.copy2(script, staging_dir / script.name)
    ref = _kaggle.dataset_ref(config, "input_dataset")
    _kaggle.write_dataset_metadata(
        staging_dir,
        ref,
        _kaggle.title_from_slug(config["kaggle"]["input_dataset"]),
    )
    return staging_dir


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Version notes (new versions only; defaults to an update line).",
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
    ) / "staging_input"
    stage_input_dataset(config, task_dir, staging)
    ref = _kaggle.dataset_ref(config, "input_dataset")
    exists = False if args.dry_run else _kaggle.dataset_exists(kaggle, ref)
    message = args.message or f"Update {config['task']['name']} inputs"
    argv = _kaggle.push_dataset_argv(kaggle, exists, staging, message)
    print(f"dataset {ref} ({'version' if exists else 'create'})")
    return _kaggle.run_or_print(argv, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
