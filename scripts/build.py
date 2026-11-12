#!/usr/bin/env python3
"""Export the trained adapter to a ``.cact`` archive with ``cactus build``.

Merges the LoRA adapter from ``scripts/finetune.py`` into the base weights
and slices the configured layer rung. ``--dry-run`` prints the argv
without executing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def build_argv(config: dict, task_dir: Path) -> list[str]:
    """Build the ``cactus build`` argv from the task config."""
    data = config["data"]
    out_dir = _common.task_relative_path(task_dir, config["task"]["out_dir"])
    return [
        _common.cactus_bin(),
        "build",
        "--lora",
        str(out_dir / config["train"]["adapter_file"]),
        "--layers",
        str(config["build"]["layers"]),
        "--out",
        str(out_dir / config["build"]["weights_file"]),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cactus argv without executing it.",
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    cli_argv = build_argv(config, task_dir)
    if args.dry_run:
        print(" ".join(cli_argv))
        return 0
    completed = subprocess.run(cli_argv, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
