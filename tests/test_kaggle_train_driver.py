"""Tests for scripts/kaggle_train_driver.py pure helpers."""

import sys

import pytest

import kaggle_train_driver


def test_find_task_single(tmp_path):
    task_dir = tmp_path / "some-dataset"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("x", encoding="utf-8")
    assert kaggle_train_driver.find_task(str(tmp_path)) == task_dir


def test_find_task_nested_mount(tmp_path):
    # Mirrors Kaggle: datasets/<owner>/<slug>/task.toml.
    task_dir = tmp_path / "datasets" / "owner" / "slug"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("x", encoding="utf-8")
    assert kaggle_train_driver.find_task(str(tmp_path)) == task_dir


def test_find_task_none_dies(tmp_path):
    (tmp_path / "empty-dir").mkdir()
    with pytest.raises(SystemExit) as excinfo:
        kaggle_train_driver.find_task(str(tmp_path))
    assert "found 0" in str(excinfo.value)
    assert "empty-dir" in str(excinfo.value)


def test_find_task_two_dies(tmp_path):
    for name in ("a", "b"):
        task_dir = tmp_path / name
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        kaggle_train_driver.find_task(str(tmp_path))


def test_finalize_outputs_prunes_cache_and_flattens(tmp_path):
    out_dir = tmp_path / "task" / "out"
    (out_dir / "jax_cache").mkdir(parents=True)
    (out_dir / "jax_cache" / "blob").write_text("x", encoding="utf-8")
    (out_dir / "adapter.safetensors").write_text("a", encoding="utf-8")
    (out_dir / "train.log").write_text("l", encoding="utf-8")
    workspace = tmp_path / "working"
    workspace.mkdir()
    kaggle_train_driver.finalize_outputs(out_dir, workspace)
    assert not (out_dir / "jax_cache").exists()
    assert (workspace / "adapter.safetensors").is_file()
    assert (workspace / "train.log").is_file()


def test_get_stage_timeout_present_and_missing(tmp_path):
    present = tmp_path / "a.toml"
    present.write_text(
        "[kaggle]\nstage_timeout = 10800\n", encoding="utf-8"
    )
    assert kaggle_train_driver.get_stage_timeout(present) == 10800
    missing = tmp_path / "b.toml"
    missing.write_text("[task]\nname = \"x\"\n", encoding="utf-8")
    assert kaggle_train_driver.get_stage_timeout(missing) is None


def test_run_stage_timeout_records_and_dies(tmp_path, capsys):
    status = []
    with pytest.raises(SystemExit):
        kaggle_train_driver.run_stage(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            status,
            0.05,
        )
    assert status[0]["returncode"] is None
    assert status[0]["timeout"] is True


def test_run_stage_success_records_no_timeout(tmp_path):
    status = []
    kaggle_train_driver.run_stage(
        [sys.executable, "-c", "pass"], status, 30
    )
    assert status[0]["returncode"] == 0
    assert status[0]["timeout"] is False


def test_build_stage_argv(tmp_path):
    argv = kaggle_train_driver.build_stage_argv(
        tmp_path / "finetune.py", tmp_path / "task.toml",
        ["--backend", "gpu"],
    )
    assert argv[1:] == [
        str(tmp_path / "finetune.py"),
        "--task", str(tmp_path / "task.toml"),
        "--backend", "gpu",
    ]
