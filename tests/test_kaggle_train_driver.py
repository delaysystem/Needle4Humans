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
    disabled = tmp_path / "c.toml"
    disabled.write_text("[kaggle]\nstage_timeout = 0\n", encoding="utf-8")
    assert kaggle_train_driver.get_stage_timeout(disabled) is None


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


def test_check_train_config():
    full = {key: 1 for key in kaggle_train_driver.REQUIRED_TRAIN_KEYS}
    kaggle_train_driver.check_train_config(full)
    thin = dict(full)
    del thin["save_every_min"]
    with pytest.raises(SystemExit) as excinfo:
        kaggle_train_driver.check_train_config(thin)
    assert "save_every_min" in str(excinfo.value)


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


def test_find_prior_checkpoints_none_and_one(tmp_path):
    assert kaggle_train_driver.find_prior_checkpoints(str(tmp_path)) == []
    ckpt_dir = tmp_path / "datasets" / "o" / "s"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "latest.json").write_text("{}", encoding="utf-8")
    # latest.json without ckpt siblings does not count.
    assert kaggle_train_driver.find_prior_checkpoints(str(tmp_path)) == []
    (ckpt_dir / "ckpt-000001.npz").write_text("x", encoding="utf-8")
    found = kaggle_train_driver.find_prior_checkpoints(str(tmp_path))
    assert found == [ckpt_dir / "latest.json"]


def test_stage_prior_checkpoints_none_is_fresh(tmp_path, capsys):
    dest = tmp_path / "out" / "checkpoints"
    assert kaggle_train_driver.stage_prior_checkpoints(str(tmp_path), dest) is False
    assert "fresh run" in capsys.readouterr().out


def test_stage_prior_checkpoints_copies_single(tmp_path):
    ckpt_dir = tmp_path / "m" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "latest.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "ckpt-000001.npz").write_text("x", encoding="utf-8")
    dest = tmp_path / "out" / "checkpoints"
    assert kaggle_train_driver.stage_prior_checkpoints(str(tmp_path), dest) is True
    assert (dest / "latest.json").is_file()
    assert (dest / "ckpt-000001.npz").is_file()


def test_stage_prior_checkpoints_ambiguous_dies(tmp_path):
    for name in ("a", "b"):
        ckpt_dir = tmp_path / name
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "latest.json").write_text("{}", encoding="utf-8")
        (ckpt_dir / "ckpt-000001.npz").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        kaggle_train_driver.stage_prior_checkpoints(
            str(tmp_path), tmp_path / "out" / "checkpoints"
        )


def test_build_loop_argv(tmp_path):
    task = {
        "epochs": 14, "lr": 0.0001, "lora_rank": 32, "lora_alpha": 32,
        "max_len": 1024, "val_split": 0.1, "seed": 0,
        "adapter_file": "a.safetensors", "save_every_min": 30,
    }
    argv = kaggle_train_driver.build_loop_argv(
        tmp_path / "kaggle_train_loop.py", task, tmp_path / "out", "16"
    )
    assert argv[1] == str(tmp_path / "kaggle_train_loop.py")
    assert argv[2] == str(tmp_path / "out" / "train.jsonl")
    assert "--batch-size" in argv and "16" in argv
    assert "--save-every-min" in argv and "30" in argv
    assert "--checkpoint-dir" in argv
    assert str(tmp_path / "out" / "checkpoints") in argv
