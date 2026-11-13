#!/usr/bin/env python3
"""Run one finetune task end to end inside a Kaggle kernel.

Copies the mounted input dataset to the writable workspace (input mounts
are read-only), installs the GPU training stack, and runs generate →
validate → finetune → build with the task config plus CLI overrides for
the Kaggle backend and batch size. Evaluation runs locally after
downloading the ``.cact`` (inference needs no JAX), so the driver stops
at build. Every stage banners to stdout (visible in kernel logs) and is
recorded in STATUS.json; the first failing stage stops the run.

Stdlib-only with no repo imports: this file executes standalone on
Kaggle. ``KAGGLE_INPUT_ROOT``/``KAGGLE_WORKSPACE`` override the platform
paths (also the seam the tests use).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Origin: Kaggle kernel filesystem layout (input mounts read-only,
# /kaggle/working persists as the version's output).
INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
WORKSPACE = os.environ.get("KAGGLE_WORKSPACE", "/kaggle/working")

# Pinned: kaggle_train_loop imports this release's training internals
# (verified identical between 4.1.0 and 4.2.0); floating would risk a
# silent internal-API mismatch. Origin: Fine-tuning Needle guide + pin.
TRAIN_PACKAGE = "cactus-compute[train,gpu]==4.2.0"

STAGES = ("generate_data", "validate_data", "finetune", "build")

# Origin: kaggle_train_loop.py checkpoint layout.
CHECKPOINTS_DIRNAME = "checkpoints"


def find_task(input_root: str) -> Path:
    """Locate the single task dir (holding task.toml) under the mount.

    Recursive: datasets mount nested (e.g. datasets/<owner>/<slug>/),
    not directly under the input root.
    """
    found = sorted(Path(input_root).rglob("task.toml"))
    if len(found) != 1:
        try:
            listing = sorted(
                str(path.relative_to(input_root))
                for path in Path(input_root).rglob("*")
            )
        except OSError:
            listing = ["<unreadable>"]
        raise SystemExit(
            f"expected exactly one task under {input_root}, "
            f"found {len(found)}; contents: {listing}"
        )
    return found[0].parent


def build_stage_argv(
    script_path: Path, task_file: Path, extra: list[str]
) -> list[str]:
    """Argv running one pipeline script against the workspace task."""
    return [sys.executable, str(script_path), "--task", str(task_file), *extra]


def find_prior_checkpoints(input_root: str) -> list[Path]:
    """latest.json pointers under the mounts with checkpoint files beside.

    The checkpoints dataset stages files flat (dataset upload skips
    subdirectories), so any latest.json with ckpt-*.npz siblings counts.
    """
    found = []
    for pointer in sorted(Path(input_root).rglob("latest.json")):
        if list(pointer.parent.glob("ckpt-*.npz")):
            found.append(pointer)
    return found


def stage_prior_checkpoints(input_root: str, dest_dir: Path) -> bool:
    """Copy a previous run's checkpoints into the workspace, if exactly one.

    Returns False when none are mounted (fresh run). Dies on several:
    resuming the wrong run's weights would silently corrupt training.
    """
    found = find_prior_checkpoints(input_root)
    if not found:
        print("no prior checkpoints mounted; fresh run", flush=True)
        return False
    if len(found) > 1:
        raise SystemExit(f"ambiguous prior checkpoints: {found}")
    src_dir = found[0].parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, dest_dir / path.name)
    print(f"staged prior checkpoints from {src_dir}", flush=True)
    return True


REQUIRED_TRAIN_KEYS = (
    "epochs", "batch_size", "lr", "lora_rank", "lora_alpha",
    "max_len", "val_split", "seed", "adapter_file", "save_every_min",
)


def check_train_config(train_config: dict) -> None:
    """Die naming missing keys (usually a stale mounted dataset)."""
    missing = [key for key in REQUIRED_TRAIN_KEYS if key not in train_config]
    if missing:
        raise SystemExit(
            f"task.toml lacks {missing}; the mounted input dataset is "
            "probably stale — wait for processing and re-push"
        )


def build_loop_argv(
    loop_path: Path, task: dict, out_dir: Path, batch_size: str
) -> list[str]:
    """Argv for the resumable training loop from the task config."""
    train_file = out_dir / "train.jsonl"
    return [
        sys.executable,
        str(loop_path),
        str(train_file),
        "--epochs", str(task["epochs"]),
        "--batch-size", str(batch_size),
        "--lr", str(task["lr"]),
        "--lora-rank", str(task["lora_rank"]),
        "--lora-alpha", str(task["lora_alpha"]),
        "--max-len", str(task["max_len"]),
        "--val-split", str(task["val_split"]),
        "--seed", str(task["seed"]),
        "--out", str(out_dir / task["adapter_file"]),
        "--checkpoint-dir", str(out_dir / CHECKPOINTS_DIRNAME),
        "--save-every-min", str(task["save_every_min"]),
    ]


def get_stage_timeout(task_file: Path) -> float | None:
    """Per-stage timeout from ``[kaggle] stage_timeout``.

    Zero or missing means unbounded: the kernel timeout stays the outer
    backstop, and periodic checkpoints bound any kill's cost to the
    save interval instead of the whole run.
    """
    import tomllib

    with open(task_file, "rb") as handle:
        config = tomllib.load(handle)
    return config.get("kaggle", {}).get("stage_timeout") or None


def run_stage(
    argv: list[str],
    status: list[dict],
    timeout: float | None,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run one stage, bannered and recorded; fail fast on error.

    A stage past ``timeout`` is recorded with timeout=true and stops the
    run instead of hanging until the kernel timeout.
    """
    name = name or Path(argv[1]).stem
    print(f"=== stage {name} ===", flush=True)
    start = time.monotonic()
    merged = dict(os.environ)
    if env:
        merged.update(env)
    try:
        completed = subprocess.run(argv, check=False, timeout=timeout, env=merged)
    except subprocess.TimeoutExpired:
        status.append(
            {
                "stage": name,
                "returncode": None,
                "elapsed_s": round(time.monotonic() - start, 1),
                "timeout": True,
            }
        )
        raise SystemExit(f"stage {name} timed out after {timeout}s")
    status.append(
        {
            "stage": name,
            "returncode": completed.returncode,
            "elapsed_s": round(time.monotonic() - start, 1),
            "timeout": False,
        }
    )
    if completed.returncode != 0:
        raise SystemExit(f"stage {name} failed with {completed.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="gpu",
        help="Backend override passed to finetune.py.",
    )
    parser.add_argument(
        "--batch-size",
        default="16",
        help="Batch-size override passed to finetune.py.",
    )
    args = parser.parse_args(argv)
    if args.backend not in ("cpu", "gpu", "auto"):
        raise SystemExit(
            f"unknown --backend: {args.backend!r} (want cpu/gpu/auto)"
        )
    task_src = find_task(INPUT_ROOT)
    task_dir = Path(WORKSPACE) / "task"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    shutil.copytree(task_src, task_dir)
    task_file = task_dir / "task.toml"
    out_dir = task_dir / "out"
    stage_prior_checkpoints(INPUT_ROOT, out_dir / CHECKPOINTS_DIRNAME)
    # Scripts sit flat beside task.toml: `datasets create` skips
    # subdirectories by default, so the input dataset has no scripts/ dir.
    scripts_dir = task_dir
    status: list[dict] = []
    status_path = task_dir / "out" / "STATUS.json"
    stage_timeout = get_stage_timeout(task_file)

    print("=== stage pip-install ===", flush=True)
    start = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", TRAIN_PACKAGE],
        check=False,
    )
    status.append(
        {
            "stage": "pip-install",
            "returncode": completed.returncode,
            "elapsed_s": round(time.monotonic() - start, 1),
            "timeout": False,
        }
    )
    if completed.returncode != 0:
        raise SystemExit("pip install of the training stack failed")

    import tomllib

    with open(task_file, "rb") as handle:
        train_config = tomllib.load(handle)["train"]
    check_train_config(train_config)
    loop_env = {"CUDA_VISIBLE_DEVICES": ""} if args.backend == "cpu" else None
    for stage in STAGES:
        if stage == "finetune":
            argv = build_loop_argv(
                scripts_dir / "kaggle_train_loop.py",
                train_config,
                out_dir,
                args.batch_size,
            )
            run_stage(argv, status, stage_timeout, name="finetune", env=loop_env)
        else:
            run_stage(
                build_stage_argv(scripts_dir / f"{stage}.py", task_file, []),
                status,
                stage_timeout,
            )
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2)
    finalize_outputs(task_dir / "out", Path(WORKSPACE))
    print("=== driver done ===", flush=True)
    return 0


def finalize_outputs(out_dir: Path, workspace: Path) -> None:
    """Prune the compile cache and flatten artifacts to the workspace root.

    The GPU compile cache is gigabytes of arch-specific files that are
    useless off the kernel and make output download crawl, so it is
    deleted. The remaining top-level out/ files are copied flat to the
    workspace root so ``kernels output`` lands them without nesting.
    """
    shutil.rmtree(out_dir / "jax_cache", ignore_errors=True)
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, workspace / path.name)


if __name__ == "__main__":
    raise SystemExit(main())
