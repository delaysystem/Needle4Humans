#!/usr/bin/env python3
"""Poll a Kaggle kernel run until it resolves, then fetch outputs.

Loops ``kaggle kernels status`` on the configured interval until the run
reads complete or error (or the overall timeout hits). On complete,
downloads the kernel outputs into the task ``out/``. On error, prints the
kernel logs and exits 1. ``--once`` prints one status line instead of
looping. ``--dry-run`` prints the plan without calling Kaggle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common

# Task choices: bounds that keep every wait finite and transient-safe.
# A hung status call must not stall the poller forever; a throttled one
# must not burn the failure budget.
MAX_CLI_FAILURES = 5
STATUS_TIMEOUT_S = 120
LOGS_TIMEOUT_S = 300
DOWNLOAD_TIMEOUT_S = 1800
RATE_LIMIT_BACKOFF_S = 300


def parse_status(text: str) -> str:
    """Map ``kernels status`` output to complete/running/error/unknown."""
    folded = text.casefold()
    if "error" in folded or "fail" in folded:
        return "error"
    if "complet" in folded:
        return "complete"
    if "run" in folded or "queue" in folded or "pend" in folded:
        return "running"
    return "unknown"


def build_status_argv(kaggle: str, ref: str) -> list[str]:
    return [kaggle, "kernels", "status", ref]


def build_output_argv(kaggle: str, ref: str, out_dir: Path) -> list[str]:
    return [kaggle, "kernels", "output", ref, "-p", str(out_dir), "-o", "-q"]


def build_logs_argv(kaggle: str, ref: str) -> list[str]:
    return [kaggle, "kernels", "logs", ref]


def is_rate_limited(stderr: str) -> bool:
    """True when CLI stderr reports API throttling (HTTP 429)."""
    folded = stderr.casefold()
    return "429" in folded or "rate limited" in folded or "ratelimit" in folded


def run_capture(
    argv: list[str], timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Run argv capturing output; raises TimeoutExpired past ``timeout``."""
    return subprocess.run(
        argv, capture_output=True, text=True, check=False, timeout=timeout
    )


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one status line instead of looping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the poll plan without calling Kaggle.",
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    kaggle = _common.kaggle_bin()
    ref = (
        f"{config['kaggle']['username']}/{config['kaggle']['kernel']}"
    )
    out_dir = _common.task_relative_path(task_dir, config["task"]["out_dir"])
    interval = config["kaggle"]["poll_interval"]
    timeout = config["kaggle"]["poll_timeout"]
    if args.dry_run:
        print(f"would poll {ref} every {interval}s for up to {timeout}s")
        print(" ".join(build_status_argv(kaggle, ref)))
        return 0
    failures = 0
    deadline = time.monotonic() + timeout
    while True:
        try:
            completed = run_capture(
                build_status_argv(kaggle, ref),
                timeout=STATUS_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            completed = None
        if completed is None or completed.returncode != 0:
            stderr = "" if completed is None else completed.stderr
            if is_rate_limited(stderr):
                print(
                    "status call rate-limited; backing off "
                    f"{RATE_LIMIT_BACKOFF_S}s (failures not counted)",
                    flush=True,
                )
                time.sleep(RATE_LIMIT_BACKOFF_S)
            else:
                failures += 1
                print(
                    f"status call failed ({failures}/{MAX_CLI_FAILURES})",
                    flush=True,
                )
                if failures >= MAX_CLI_FAILURES:
                    print(stderr.strip(), flush=True)
                    return 1
                time.sleep(interval)
        else:
            failures = 0
            state = parse_status(completed.stdout)
            print(f"status: {state}", flush=True)
            if args.once:
                return 0
            if state == "complete":
                print(
                    " ".join(build_output_argv(kaggle, ref, out_dir)),
                    flush=True,
                )
                try:
                    downloaded = run_capture(
                        build_output_argv(kaggle, ref, out_dir),
                        timeout=DOWNLOAD_TIMEOUT_S,
                    )
                except subprocess.TimeoutExpired:
                    print(
                        "output download timed out after "
                        f"{DOWNLOAD_TIMEOUT_S}s",
                        flush=True,
                    )
                    return 1
                if downloaded.returncode != 0:
                    print(downloaded.stderr.strip(), flush=True)
                    return 1
                print(f"outputs downloaded to {out_dir}", flush=True)
                return 0
            if state == "error":
                try:
                    logs = run_capture(
                        build_logs_argv(kaggle, ref),
                        timeout=LOGS_TIMEOUT_S,
                    )
                except subprocess.TimeoutExpired:
                    print("log fetch timed out", flush=True)
                    return 1
                print(logs.stdout, flush=True)
                print(logs.stderr, file=sys.stderr)
                return 1
            if state == "unknown":
                print(
                    f"warning: unparsed status: {completed.stdout!r}",
                    flush=True,
                )
            time.sleep(interval)
        if time.monotonic() >= deadline:
            print(f"poll timed out after {timeout}s", flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
