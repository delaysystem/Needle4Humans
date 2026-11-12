"""Tests for scripts/kaggle_push.py metadata and argv."""

import json

import kaggle_push


def make_config():
    return {
        "task": {"name": "mini", "out_dir": "out"},
        "kaggle": {
            "username": "someone",
            "input_dataset": "mini-input",
            "kernel": "mini-train",
            "accelerator": "NvidiaTeslaT4",
            "timeout": 3600,
        },
    }


def test_kernel_ref():
    assert kaggle_push.kernel_ref(make_config()) == "someone/mini-train"


def test_render_kernel_metadata():
    assert kaggle_push.render_kernel_metadata(make_config()) == {
        "id": "someone/mini-train",
        "title": "mini train",
        "code_file": "kaggle_train_driver.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["someone/mini-input"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def test_render_kernel_metadata_cpu():
    config = make_config()
    config["kaggle"]["gpu"] = False
    metadata = kaggle_push.render_kernel_metadata(config)
    assert metadata["enable_gpu"] == "false"
    assert metadata["machine_shape"] == ""


def test_render_kernel_metadata_resume_mounts_checkpoints():
    config = make_config()
    config["kaggle"]["checkpoints_dataset"] = "mini-checkpoints"
    assert kaggle_push.render_kernel_metadata(
        config, resume=True
    )["dataset_sources"] == ["someone/mini-input", "someone/mini-checkpoints"]
    assert kaggle_push.render_kernel_metadata(
        config
    )["dataset_sources"] == ["someone/mini-input"]


def test_render_kernel_metadata_resume_needs_slug():
    import pytest

    with pytest.raises(SystemExit):
        kaggle_push.render_kernel_metadata(make_config(), resume=True)


def test_stage_kernel_copies_driver_and_metadata(tmp_path):
    root = tmp_path / "root"
    task_dir = root / "tasks" / "t"
    (root / "scripts").mkdir(parents=True)
    task_dir.mkdir(parents=True)
    (root / "scripts" / "kaggle_train_driver.py").write_text(
        "print(1)\n", encoding="utf-8"
    )
    staging = tmp_path / "staging"
    kaggle_push.stage_kernel(make_config(), task_dir, staging)
    assert (staging / "kaggle_train_driver.py").is_file()
    with open(staging / "kernel-metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["id"] == "someone/mini-train"
    assert metadata["dataset_sources"] == ["someone/mini-input"]


def test_build_push_argv(tmp_path):
    assert kaggle_push.build_push_argv("kaggle", tmp_path, 3600) == [
        "kaggle", "kernels", "push", "-p", str(tmp_path), "-t", "3600",
    ]
