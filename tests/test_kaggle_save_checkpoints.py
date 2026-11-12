"""Tests for scripts/kaggle_save_checkpoints.py pure helpers."""

import json
import re

import pytest

import kaggle_save_checkpoints as save_ckpts


def test_pattern_matches_checkpoints_only():
    pattern = save_ckpts.CHECKPOINT_PATTERN
    assert re.search(pattern, "latest.json")
    assert re.search(pattern, "ckpt-000044.npz")
    assert re.search(pattern, "ckpt-000044.json")
    assert not re.search(pattern, "adapter.safetensors")
    assert not re.search(pattern, "train.log")


def test_build_download_argv_latest_and_version(tmp_path):
    latest = save_ckpts.build_download_argv("kaggle", "o/s", tmp_path, None)
    assert latest[3] == "o/s"
    assert "--file-pattern" in latest
    pinned = save_ckpts.build_download_argv("kaggle", "o/s", tmp_path, "2")
    assert pinned[3] == "o/s/2"


def write_pair(directory, step):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"ckpt-{step:06d}.npz").write_text("x", encoding="utf-8")
    (directory / f"ckpt-{step:06d}.json").write_text("{}", encoding="utf-8")
    (directory / "latest.json").write_text(
        json.dumps({"step": step}), encoding="utf-8"
    )


def test_flatten_moves_single_set_to_root(tmp_path):
    write_pair(tmp_path / "task" / "out" / "checkpoints", 44)
    save_ckpts.flatten_downloaded(tmp_path)
    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "ckpt-000044.npz").is_file()
    assert not (tmp_path / "task").exists()


def test_flatten_dies_on_zero_or_two(tmp_path):
    with pytest.raises(SystemExit):
        save_ckpts.flatten_downloaded(tmp_path)
    write_pair(tmp_path / "a" / "checkpoints", 1)
    write_pair(tmp_path / "b" / "checkpoints", 2)
    with pytest.raises(SystemExit):
        save_ckpts.flatten_downloaded(tmp_path)


def test_verify_ok_and_missing(tmp_path):
    write_pair(tmp_path, 44)
    assert save_ckpts.verify_checkpoints(tmp_path) == 44
    (tmp_path / "ckpt-000044.npz").unlink()
    with pytest.raises(SystemExit):
        save_ckpts.verify_checkpoints(tmp_path)
    (tmp_path / "latest.json").unlink()
    with pytest.raises(SystemExit):
        save_ckpts.verify_checkpoints(tmp_path)
