"""Shared Kaggle helpers: dataset refs, metadata, create-or-version."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common

# Origin: the `kaggle datasets init` template default.
DATASET_LICENSE = "CC0-1.0"

# Origin: task choices. Uploads return before server-side processing
# finishes; pushing a kernel in that window mounts the previous dataset
# version (observed: stale task.toml → KeyError seconds into the run).
LIST_TIMEOUT_S = 60
READY_POLL_INTERVAL_S = 15
READY_TIMEOUT_S = 1200
# Processing lag observed: a kernel pushed 7 min after upload still
# mounted the previous version. lastUpdated moves at accept time, so the
# grace must cover the processing tail itself.
READY_GRACE_S = 300


def dataset_ref(config: dict, slug_key: str) -> str:
    """Full ``owner/slug`` ref from the ``[kaggle]`` section."""
    kaggle = config["kaggle"]
    return f"{kaggle['username']}/{kaggle[slug_key]}"


def title_from_slug(slug: str) -> str:
    """Title that re-slugifies to ``slug``.

    Kaggle derives the kernel/dataset slug from the title (lowercase,
    separators to dashes) and warns when it does not match the metadata
    id, so titles are generated from slugs, never authored separately.
    """
    return slug.replace("-", " ")


def render_dataset_metadata(ref: str, title: str) -> dict:
    """Render a ``dataset-metadata.json`` body for create/version."""
    return {"title": title, "id": ref, "licenses": [{"name": DATASET_LICENSE}]}


def write_dataset_metadata(staging_dir: Path, ref: str, title: str) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as handle:
        json.dump(render_dataset_metadata(ref, title), handle, indent=2)


def dataset_exists(kaggle: str, ref: str) -> bool:
    """True when the dataset ref already exists on Kaggle."""
    completed = subprocess.run(
        [kaggle, "datasets", "files", ref, "-v", "--page-size", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def push_dataset_argv(
    kaggle: str, exists: bool, staging_dir: Path, message: str
) -> list[str]:
    """Create argv for a new dataset, version argv for an existing one."""
    if exists:
        return [
            kaggle, "datasets", "version",
            "-p", str(staging_dir), "-m", message,
        ]
    return [kaggle, "datasets", "create", "-p", str(staging_dir)]


def run_or_print(argv: list[str], dry_run: bool) -> int:
    """Print the argv under --dry-run, else run it and return its code."""
    if dry_run:
        print(" ".join(argv))
        return 0
    completed = subprocess.run(argv, check=False)
    return completed.returncode


def _fetch_dataset_rows(kaggle: str) -> list:
    """My datasets as parsed rows, or [] when the call fails."""
    try:
        completed = subprocess.run(
            [
                kaggle, "datasets", "list", "-m", "--format", "json",
                "--page-size", "200",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=LIST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return []
    if completed.returncode != 0:
        return []
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(rows, dict):
        rows = rows.get("datasets", [])
    return rows if isinstance(rows, list) else []


def dataset_last_updated(kaggle: str, ref: str):
    """lastUpdated of ``ref`` as naive UTC datetime, else None."""
    for row in _fetch_dataset_rows(kaggle):
        if not isinstance(row, dict) or row.get("ref") != ref:
            continue
        try:
            return datetime.strptime(
                row["lastUpdated"][:19], "%Y-%m-%d %H:%M:%S"
            )
        except (KeyError, ValueError, TypeError):
            return None
    return None


def wait_for_dataset_updated(
    kaggle: str,
    ref: str,
    before,
    timeout: float = READY_TIMEOUT_S,
    sleep_fn=time.sleep,
) -> None:
    """Block until ``ref`` shows a newer lastUpdated than ``before``.

    ``before`` is None for brand-new datasets. A grace sleep follows the
    advance because lastUpdated may move at accept time while processing
    still runs. Dies on timeout instead of pushing against a stale mount.
    """
    deadline = time.monotonic() + timeout
    while True:
        current = dataset_last_updated(kaggle, ref)
        if current is not None and current != before:
            sleep_fn(READY_GRACE_S)
            return
        if time.monotonic() >= deadline:
            _common.die(f"dataset {ref} not ready after {timeout}s")
        sleep_fn(READY_POLL_INTERVAL_S)
