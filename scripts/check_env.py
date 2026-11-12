#!/usr/bin/env python3
"""Verify the machine can train Needle 3 LoRA adapters locally.

Checks (all from the
`Fine-tuning Needle <https://cactuscompute.com/blog/finetuning-needle>`_
guide and the cactus README):

- Python version floor
- ``cactus`` package importable
- ``cactus`` CLI on PATH
- JAX importable, with CUDA/CPU backend reported
- GPU VRAM reported, optionally gated by ``--min-vram-mb``

Exit 0 when every check passes, 1 otherwise. Deliberately stdlib-only with
no repo imports so it runs on any interpreter, including outside the venv.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

# Origin: cactus README, "Python 3.9 or later".
MIN_PYTHON = (3, 9)


def _ok(label: str, detail: str = "") -> bool:
    suffix = f" ({detail})" if detail else ""
    print(f"[ok] {label}{suffix}")
    return True


def _fail(label: str, detail: str) -> bool:
    print(f"[FAIL] {label}: {detail}")
    return False


def check_python() -> bool:
    current = sys.version_info[:2]
    if current >= MIN_PYTHON:
        return _ok(
            "python version",
            f"{sys.version.split()[0]} >= "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
        )
    return _fail(
        "python version",
        f"{sys.version.split()[0]} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
    )


def check_package() -> bool:
    try:
        import cactus  # noqa: F401
    except ImportError:
        return _fail(
            "cactus package",
            'not importable; run pip install "cactus-compute[train,gpu]"',
        )
    try:
        from importlib.metadata import version

        detail = f"cactus-compute {version('cactus-compute')}"
    except Exception:  # noqa: BLE001 - version lookup is best-effort
        detail = "importable, version unknown"
    return _ok("cactus package", detail)


def check_cli() -> bool:
    path = shutil.which("cactus")
    if path is None:
        return _fail("cactus CLI", "not on PATH; activate the project .venv")
    return _ok("cactus CLI", path)


def check_jax() -> bool:
    try:
        import jax
    except ImportError:
        return _fail(
            "jax",
            'not importable; run pip install "cactus-compute[train,gpu]"',
        )
    try:
        devices = jax.devices()
    except Exception as exc:  # noqa: BLE001 - report backend failure as-is
        return _fail("jax backend", str(exc))
    kinds = sorted({device.device_kind for device in devices})
    return _ok("jax backend", f"{len(devices)} device(s): {', '.join(kinds)}")


def check_vram(min_vram_mb: int | None) -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("[skip] gpu vram: nvidia-smi not on PATH (CPU-only machine)")
        return True
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return _fail("gpu vram", f"nvidia-smi failed: {exc}")
    totals = []
    for line in completed.stdout.strip().splitlines():
        total, _, _ = line.partition(",")
        totals.append(int(total.strip()))
    if not totals:
        return _fail("gpu vram", "no GPUs reported")
    best = max(totals)
    if min_vram_mb is not None and best < min_vram_mb:
        return _fail(
            "gpu vram",
            f"largest GPU has {best} MiB, need {min_vram_mb} MiB; "
            "lower batch size or train on CPU",
        )
    return _ok("gpu vram", f"largest GPU: {best} MiB total")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-vram-mb",
        type=int,
        default=None,
        help="Fail unless a GPU has at least this much total VRAM.",
    )
    args = parser.parse_args(argv)
    results = [
        check_python(),
        check_package(),
        check_cli(),
        check_jax(),
        check_vram(args.min_vram_mb),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
