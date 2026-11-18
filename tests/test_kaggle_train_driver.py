"""Tests for scripts/kaggle_train_driver.py pure helpers."""

import json
import sys
import tomllib
from pathlib import Path

import pytest

import kaggle_train_driver

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


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


def test_shipped_task_train_sections_pass_guard():
    # Regression: save_every_min lived under [kaggle] while the driver
    # reads it from [train]; v12 failed on Kaggle on a fresh dataset.
    # Every shipped task's [train] must satisfy the driver's guard.
    task_files = sorted(TASKS_DIR.glob("*/task.toml"))
    assert task_files, f"no task.toml under {TASKS_DIR}"
    for task_file in task_files:
        if task_file.parent.name.startswith("_"):
            continue
        with open(task_file, "rb") as handle:
            config = tomllib.load(handle)
        kaggle_train_driver.check_train_config(config["train"])


def test_resolve_kaggle_defaults_prefers_cli_then_toml(tmp_path):
    task_file = tmp_path / "task.toml"
    task_file.write_text(
        "[kaggle]\nbackend = \"cpu\"\nbatch_size = 32\n", encoding="utf-8"
    )
    assert kaggle_train_driver.resolve_kaggle_defaults(
        task_file, None, None
    ) == ("cpu", "32", False)
    assert kaggle_train_driver.resolve_kaggle_defaults(
        task_file, "gpu", "8"
    ) == ("gpu", "8", False)


def test_resolve_kaggle_defaults_code_fallbacks_and_warm(tmp_path):
    bare = tmp_path / "bare.toml"
    bare.write_text("[task]\nname = \"x\"\n", encoding="utf-8")
    assert kaggle_train_driver.resolve_kaggle_defaults(
        bare, None, None
    ) == ("gpu", "16", False)
    warm = tmp_path / "warm.toml"
    warm.write_text("[kaggle]\nwarm_start = true\n", encoding="utf-8")
    assert kaggle_train_driver.resolve_kaggle_defaults(
        warm, None, None
    ) == ("gpu", "16", True)
    bad = tmp_path / "bad.toml"
    bad.write_text("[kaggle]\nbackend = \"tpu\"\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        kaggle_train_driver.resolve_kaggle_defaults(bad, None, None)


def test_find_warm_start_adapter(tmp_path):
    task_src = tmp_path / "datasets" / "o" / "input"
    task_src.mkdir(parents=True)
    (task_src / "task.toml").write_text("x", encoding="utf-8")
    assert kaggle_train_driver.find_warm_start_adapter(
        str(tmp_path), task_src
    ) is None
    # An adapter inside the task dir itself does not count.
    (task_src / "adapter.safetensors").write_text("x", encoding="utf-8")
    assert kaggle_train_driver.find_warm_start_adapter(
        str(tmp_path), task_src
    ) is None
    out_mount = tmp_path / "datasets" / "o" / "outputs"
    out_mount.mkdir(parents=True)
    adapter = out_mount / "adapter.safetensors"
    adapter.write_text("x", encoding="utf-8")
    assert kaggle_train_driver.find_warm_start_adapter(
        str(tmp_path), task_src
    ) == adapter
    (out_mount / "other.safetensors").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        kaggle_train_driver.find_warm_start_adapter(str(tmp_path), task_src)


def test_build_loop_argv_init_adapter(tmp_path):
    task = {
        "epochs": 14, "lr": 0.0001, "lora_rank": 32, "lora_alpha": 32,
        "max_len": 1024, "val_split": 0.1, "seed": 0,
        "adapter_file": "a.safetensors", "save_every_min": 30,
        "val_every": 1,
    }
    plain = kaggle_train_driver.build_loop_argv(
        tmp_path / "kaggle_train_loop.py", task, tmp_path / "out", "16"
    )
    assert "--init-adapter" not in plain
    warm = kaggle_train_driver.build_loop_argv(
        tmp_path / "kaggle_train_loop.py", task, tmp_path / "out", "16",
        str(tmp_path / "warm.safetensors"),
    )
    assert warm[-2:] == ["--init-adapter", str(tmp_path / "warm.safetensors")]


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


def test_build_loop_argv_expect_backend(tmp_path):
    task = {
        "epochs": 14, "lr": 0.0001, "lora_rank": 32, "lora_alpha": 32,
        "max_len": 1024, "val_split": 0.1, "seed": 0,
        "adapter_file": "a.safetensors", "save_every_min": 30,
        "val_every": 1,
    }
    argv = kaggle_train_driver.build_loop_argv(
        tmp_path / "kaggle_train_loop.py", task, tmp_path / "out", "16",
        None, "tpu",
    )
    assert argv[-2:] == ["--expect-backend", "tpu"]


def test_resolve_train_package_default_and_override(tmp_path):
    bare = tmp_path / "bare.toml"
    bare.write_text("[task]\nname = \"x\"\n", encoding="utf-8")
    assert kaggle_train_driver.resolve_train_package(bare) == (
        kaggle_train_driver.TRAIN_PACKAGE
    )
    custom = tmp_path / "custom.toml"
    custom.write_text(
        "[kaggle]\ntrain_package = \"cactus-compute[train]==4.2.0\"\n",
        encoding="utf-8",
    )
    assert kaggle_train_driver.resolve_train_package(custom) == (
        "cactus-compute[train]==4.2.0"
    )


def test_resolve_probe_toml_and_cli(tmp_path):
    bare = tmp_path / "bare.toml"
    bare.write_text("[task]\nname = \"x\"\n", encoding="utf-8")
    assert kaggle_train_driver.resolve_probe(bare, False) is False
    assert kaggle_train_driver.resolve_probe(bare, True) is True
    probing = tmp_path / "probing.toml"
    probing.write_text("[kaggle]\nprobe = true\n", encoding="utf-8")
    assert kaggle_train_driver.resolve_probe(probing, False) is True


def test_resolve_expect_backend(tmp_path):
    import pytest

    bare = tmp_path / "bare.toml"
    bare.write_text("[task]\nname = \"x\"\n", encoding="utf-8")
    assert kaggle_train_driver.resolve_expect_backend(bare) == "gpu"
    tpu = tmp_path / "tpu.toml"
    tpu.write_text(
        "[kaggle]\ngpu = false\ntpu = true\n", encoding="utf-8"
    )
    assert kaggle_train_driver.resolve_expect_backend(tpu) == "tpu"
    cpu = tmp_path / "cpu.toml"
    cpu.write_text("[kaggle]\ngpu = false\n", encoding="utf-8")
    assert kaggle_train_driver.resolve_expect_backend(cpu) == "cpu"
    both = tmp_path / "both.toml"
    both.write_text(
        "[kaggle]\ngpu = true\ntpu = true\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        kaggle_train_driver.resolve_expect_backend(both)


def test_probe_hardware_reports_without_raising():
    report = kaggle_train_driver.probe_hardware()
    assert "python" in report and "machine" in report
    assert "jax" in report and "installed" in report["jax"]
    assert isinstance(report["env"], dict)


def test_main_probe_branch_skips_stages(tmp_path, monkeypatch, capsys):
    # End-to-end through main() with simulated Kaggle mounts, driven by
    # task.toml alone (Kaggle executes the driver with no argv).
    input_root = tmp_path / "input"
    task_src = input_root / "datasets" / "o" / "slug"
    task_src.mkdir(parents=True)
    (task_src / "task.toml").write_text(
        "[kaggle]\nprobe = true\n", encoding="utf-8"
    )
    workspace = tmp_path / "working"
    workspace.mkdir()
    monkeypatch.setattr(kaggle_train_driver, "INPUT_ROOT", str(input_root))
    monkeypatch.setattr(kaggle_train_driver, "WORKSPACE", str(workspace))
    assert kaggle_train_driver.main([]) == 0
    status = json.loads(
        (workspace / "task" / "out" / "STATUS.json").read_text(
            encoding="utf-8"
        )
    )
    assert status[0]["stage"] == "probe"
    assert status[0]["returncode"] == 0
    assert "jax" in status[0]["report"]
    out = capsys.readouterr().out
    assert "=== stage probe ===" in out
    assert "pip-install" not in out


def test_build_loop_argv(tmp_path):
    task = {
        "epochs": 14, "lr": 0.0001, "lora_rank": 32, "lora_alpha": 32,
        "max_len": 1024, "val_split": 0.1, "seed": 0,
        "adapter_file": "a.safetensors", "save_every_min": 30,
        "val_every": 1,
    }
    argv = kaggle_train_driver.build_loop_argv(
        tmp_path / "kaggle_train_loop.py", task, tmp_path / "out", "16"
    )
    assert argv[1] == str(tmp_path / "kaggle_train_loop.py")
    assert argv[2] == str(tmp_path / "out" / "train.jsonl")
    assert "--batch-size" in argv and "16" in argv
    assert "--save-every-min" in argv and "30" in argv
    assert "--val-every" in argv and "1" in argv
    assert "--checkpoint-dir" in argv
    assert str(tmp_path / "out" / "checkpoints") in argv


def test_build_stage_argv_no_extra(tmp_path):
    argv = kaggle_train_driver.build_stage_argv(
        tmp_path / "s.py", tmp_path / "t.toml", []
    )
    assert argv[1:] == [str(tmp_path / "s.py"), "--task", str(tmp_path / "t.toml")]


def test_build_stage_argv_uses_current_interpreter(tmp_path):
    import sys

    argv = kaggle_train_driver.build_stage_argv(
        tmp_path / "s.py", tmp_path / "t.toml", []
    )
    assert argv[0] == sys.executable


def test_build_stage_argv_preserves_extra_order(tmp_path):
    argv = kaggle_train_driver.build_stage_argv(
        tmp_path / "s.py", tmp_path / "t.toml", ["--b", "2", "--a", "1"]
    )
    assert argv[-4:] == ["--b", "2", "--a", "1"]
