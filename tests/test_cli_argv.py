"""Tests for scripts/finetune.py and scripts/build.py argv builders."""

import sys
import threading
import time

import build
import finetune


def make_config(tmp_path):
    return (
        {
            "task": {"out_dir": "out"},
            "data": {"train_file": "train.jsonl"},
            "train": {
                "epochs": 20,
                "batch_size": 16,
                "lr": 0.0001,
                "lora_rank": 16,
                "lora_alpha": 32,
                "max_len": 1024,
                "val_split": 0.1,
                "seed": 0,
                "adapter_file": "adapter.safetensors",
            },
            "build": {"layers": 20, "weights_file": "w.cact"},
        },
        tmp_path,
    )


def fake_cactus_env(tmp_path, monkeypatch):
    """Fake interpreter dir containing an executable cactus binary."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "cactus"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))


def test_finetune_argv(tmp_path, monkeypatch):
    fake_cactus_env(tmp_path, monkeypatch)
    config, task_dir = make_config(tmp_path)
    argv = finetune.build_argv(config, task_dir)
    assert argv[0].endswith("/cactus") and argv[1] == "finetune"
    assert "--epochs" in argv and "20" in argv
    assert "--batch-size" in argv and "16" in argv
    assert "--lora-rank" in argv and "--lora-alpha" in argv
    assert argv[-2:] == ["--out", str(task_dir / "out" / "adapter.safetensors")]
    assert str(task_dir / "out" / "train.jsonl") in argv


def test_build_argv(tmp_path, monkeypatch):
    fake_cactus_env(tmp_path, monkeypatch)
    config, task_dir = make_config(tmp_path)
    argv = build.build_argv(config, task_dir)
    assert argv[0].endswith("/cactus") and argv[1] == "build"
    assert "--layers" in argv and "20" in argv
    assert "--lora" in argv and "--out" in argv


def backend_config(value):
    config = {"train": {}}
    if value != "missing":
        config["train"]["backend"] = value
    return config


def test_train_env_cpu_hides_gpus(tmp_path):
    assert finetune.train_env(backend_config("cpu"), tmp_path) == {
        "CUDA_VISIBLE_DEVICES": ""
    }


def test_train_env_gpu_auto_missing_inherit(tmp_path):
    assert finetune.train_env(backend_config("gpu"), tmp_path) == {}
    assert finetune.train_env(backend_config("auto"), tmp_path) == {}
    assert finetune.train_env(backend_config("missing"), tmp_path) == {}


def test_train_env_unknown_dies(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        finetune.train_env(backend_config("tpu"), tmp_path)


def test_apply_overrides():
    config = {"train": {"backend": "cpu", "batch_size": 4}}
    assert finetune.apply_overrides(config, None, None)["train"] == {
        "backend": "cpu",
        "batch_size": 4,
    }
    assert finetune.apply_overrides(config, "gpu", "16")["train"] == {
        "backend": "gpu",
        "batch_size": "16",
    }
    # Original untouched.
    assert config["train"] == {"backend": "cpu", "batch_size": 4}


def test_train_env_cache_dir_sets_cache(tmp_path):
    config = {"train": {"backend": "cpu", "jax_cache_dir": "out/jax_cache"}}
    env = finetune.train_env(config, tmp_path)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(
        tmp_path / "out" / "jax_cache"
    )
    assert (tmp_path / "out" / "jax_cache").is_dir()


def test_train_env_existing_cache_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/keep")
    config = {"train": {"backend": "cpu", "jax_cache_dir": "out/jax_cache"}}
    env = finetune.train_env(config, tmp_path)
    assert "JAX_COMPILATION_CACHE_DIR" not in env
    assert not (tmp_path / "out" / "jax_cache").exists()


def test_run_tee_streams_before_exit(tmp_path, capsys):
    """First line must reach the log while the child still runs.

    Regression test: iterating the pipe yields nothing until the 8 KB
    readahead buffer fills or the child exits, hiding all progress.
    """
    log_path = tmp_path / "tee.log"
    argv = [
        sys.executable,
        "-c",
        "print('first', flush=True); "
        "import time; time.sleep(5); print('second', flush=True)",
    ]
    thread = threading.Thread(
        target=finetune.run_tee, args=(argv, log_path), daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 3.0
        while thread.is_alive() and time.monotonic() < deadline:
            if log_path.exists() and "first" in log_path.read_text():
                break
            time.sleep(0.05)
        assert log_path.exists()
        assert "first" in log_path.read_text(), "no output before child exit"
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert "second" in log_path.read_text()
