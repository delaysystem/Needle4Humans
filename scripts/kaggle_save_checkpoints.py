#!/usr/bin/env python3
"""Rescue checkpoints from a kernel run into the checkpoints dataset.

Downloads checkpoint files (latest.json plus ckpt-*.npz/json pairs) from
a kernel run's outputs, verifies at least one complete pair exists, and
creates or versions the checkpoints dataset that a `--resume` push
mounts. `--version` picks which kernel run (default: latest).
``--dry-run`` prints the plan without calling Kaggle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common
import _kaggle

CHECKPOINT_PATTERN = "(latest\\.json|ckpt-.*)$"


def build_download_argv(
    kaggle: str, ref: str, staging_dir: Path, version: str | None
) -> list[str]:
    """Argv downloading only checkpoint files from a kernel run."""
    target = f"{ref}/{version}" if version else ref
    return [
        kaggle, "kernels", "output", target,
        "-p", str(staging_dir), "-o", "-q",
        "--file-pattern", CHECKPOINT_PATTERN,
    ]


def flatten_downloaded(staging_dir: Path) -> None:
    """Move one nested checkpoints set to the staging root.

    Kernel downloads preserve the remote tree (task/out/checkpoints/),
    but dataset upload skips subdirectories, so the pair must sit flat
    beside dataset-metadata.json. Dies unless exactly one set exists.
    """
    pointers = sorted(staging_dir.rglob("latest.json"))
    if len(pointers) != 1:
        _common.die(
            f"expected exactly one checkpoints set under {staging_dir}, "
            f"found {len(pointers)}"
        )
    for path in sorted(pointers[0].parent.iterdir()):
        if path.is_file():
            shutil.move(str(path), staging_dir / path.name)
    for child in sorted(staging_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)


def verify_checkpoints(staging_dir: Path) -> str:
    """Return the latest step, dying unless a complete pair exists."""
    latest = staging_dir / "latest.json"
    if not latest.is_file():
        _common.die(f"no latest.json downloaded to {staging_dir}")
    with open(latest, encoding="utf-8") as handle:
        step = json.load(handle).get("step")
    pair = [
        staging_dir / f"ckpt-{step:06d}.npz",
        staging_dir / f"ckpt-{step:06d}.json",
    ]
    missing = [str(path) for path in pair if not path.is_file()]
    if missing:
        _common.die(f"incomplete checkpoint pair: {', '.join(missing)}")
    return step


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Kernel run version to rescue from (default: latest).",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Version notes (new versions only; defaults to an update line).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without calling Kaggle.",
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    kaggle = _common.kaggle_bin()
    kernel_ref = f"{config['kaggle']['username']}/{config['kaggle']['kernel']}"
    staging = _common.task_relative_path(
        task_dir, config["task"]["out_dir"]
    ) / "staging_checkpoints"
    if args.dry_run:
        print(
            " ".join(
                build_download_argv(kaggle, kernel_ref, staging, args.version)
            )
        )
        return 0
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    downloaded = subprocess.run(
        build_download_argv(kaggle, kernel_ref, staging, args.version),
        check=False,
    )
    if downloaded.returncode != 0:
        return downloaded.returncode
    flatten_downloaded(staging)
    step = verify_checkpoints(staging)
    ref = _kaggle.dataset_ref(config, "checkpoints_dataset")
    _kaggle.write_dataset_metadata(
        staging, ref, _kaggle.title_from_slug(config["kaggle"]["checkpoints_dataset"])
    )
    exists = _kaggle.dataset_exists(kaggle, ref)
    message = args.message or (
        f"Checkpoints from {kernel_ref} at step {step}"
    )
    argv = _kaggle.push_dataset_argv(kaggle, exists, staging, message)
    print(f"dataset {ref} ({'version' if exists else 'create'})")
    return _kaggle.run_or_print(argv, False)


if __name__ == "__main__":
    raise SystemExit(main())
