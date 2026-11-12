#!/usr/bin/env python3
"""Publish downloaded training binaries as a versioned Kaggle dataset.

Stages the ``[kaggle] publish_files`` from the task ``out/`` plus
dataset-metadata.json, then creates the output dataset or cuts a new
version. Fails loudly when a listed file is missing. ``--dry-run``
stages files and prints the argv without calling Kaggle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common
import _kaggle


def stage_output_dataset(
    config: dict, task_dir: Path, staging_dir: Path
) -> Path:
    """Copy published binaries into the dataset staging dir."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    out_dir = _common.task_relative_path(task_dir, config["task"]["out_dir"])
    missing = [
        name
        for name in config["kaggle"]["publish_files"]
        if not (out_dir / name).is_file()
    ]
    if missing:
        _common.die(f"missing output files: {', '.join(sorted(missing))}")
    for name in config["kaggle"]["publish_files"]:
        shutil.copy2(out_dir / name, staging_dir / name)
    ref = _kaggle.dataset_ref(config, "output_dataset")
    _kaggle.write_dataset_metadata(
        staging_dir,
        ref,
        _kaggle.title_from_slug(config["kaggle"]["output_dataset"]),
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
    ) / "staging_output"
    stage_output_dataset(config, task_dir, staging)
    ref = _kaggle.dataset_ref(config, "output_dataset")
    exists = False if args.dry_run else _kaggle.dataset_exists(kaggle, ref)
    message = args.message or f"Update {config['task']['name']} outputs"
    argv = _kaggle.push_dataset_argv(kaggle, exists, staging, message)
    print(f"dataset {ref} ({'version' if exists else 'create'})")
    return _kaggle.run_or_print(argv, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
