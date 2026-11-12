#!/usr/bin/env python3
"""Train a LoRA adapter with ``cactus finetune`` from the task config.

Builds the CLI argv from ``[train]`` plus the generated train file, streams
the subprocess output to stdout and the configured log file, and passes
the exit code through. ``--dry-run`` prints the argv without executing.

``[train] backend`` selects the JAX device: ``cpu`` hides GPUs via
``CUDA_VISIBLE_DEVICES``; ``gpu`` and ``auto`` (the default) inherit the
environment.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def build_argv(config: dict, task_dir: Path) -> list[str]:
    """Build the ``cactus finetune`` argv from the task config."""
    data = config["data"]
    train = config["train"]
    out_dir = _common.task_relative_path(task_dir, config["task"]["out_dir"])
    return [
        _common.cactus_bin(),
        "finetune",
        str(out_dir / data["train_file"]),
        "--epochs",
        str(train["epochs"]),
        "--batch-size",
        str(train["batch_size"]),
        "--lr",
        str(train["lr"]),
        "--lora-rank",
        str(train["lora_rank"]),
        "--lora-alpha",
        str(train["lora_alpha"]),
        "--max-len",
        str(train["max_len"]),
        "--val-split",
        str(train["val_split"]),
        "--seed",
        str(train["seed"]),
        "--out",
        str(out_dir / train["adapter_file"]),
    ]


def apply_overrides(
    config: dict, backend: str | None, batch_size: str | None
) -> dict:
    """Return config with CLI --backend/--batch-size applied to [train]."""
    train = dict(config["train"])
    if backend is not None:
        train["backend"] = backend
    if batch_size is not None:
        train["batch_size"] = batch_size
    return {**config, "train": train}


def train_env(config: dict, task_dir: Path) -> dict[str, str]:
    """Environment overrides for the trainer.

    ``[train] backend`` picks the device (``cpu`` hides GPUs, ``gpu`` and
    ``auto`` inherit the environment). ``[train] jax_cache_dir``, relative
    to the task dir, enables JAX's persistent compilation cache so a
    restarted run reloads the compiled step instead of recompiling; an
    already-set ``JAX_COMPILATION_CACHE_DIR`` wins over the config.
    """
    backend = config["train"].get("backend", "auto")
    if backend == "cpu":
        env = {"CUDA_VISIBLE_DEVICES": ""}
    elif backend in ("gpu", "auto"):
        env = {}
    else:
        _common.die(
            f"unknown [train] backend: {backend!r} (want cpu/gpu/auto)"
        )
        raise AssertionError("unreachable")
    cache = config["train"].get("jax_cache_dir")
    if cache and "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        cache_path = _common.task_relative_path(task_dir, cache)
        cache_path.mkdir(parents=True, exist_ok=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_path)
    return env


def run_tee(
    argv: list[str], log_path: Path, env: dict[str, str] | None = None
) -> int:
    """Run argv, tee-ing output to stdout and the log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(os.environ)
    merged.update(env or {})
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged,
            bufsize=1,
        )
        assert process.stdout is not None
        # readline loop, not pipe iteration: the iterator's readahead holds
        # output until 8 KB accumulates or the child exits, hiding progress.
        while True:
            line = process.stdout.readline()
            if not line:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cactus argv without executing it.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Override [train] backend (cpu/gpu/auto) for this run.",
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        help="Override [train] batch_size for this run.",
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    config = apply_overrides(config, args.backend, args.batch_size)
    cli_argv = build_argv(config, task_dir)
    if args.dry_run:
        print(" ".join(cli_argv))
        return 0
    log_path = _common.task_relative_path(
        task_dir, config["task"]["out_dir"]
    ) / config["train"]["log_file"]
    return run_tee(cli_argv, log_path, train_env(config, task_dir))


if __name__ == "__main__":
    raise SystemExit(main())
