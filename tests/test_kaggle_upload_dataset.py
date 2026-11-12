"""Tests for scripts/_kaggle.py, kaggle_upload_dataset.py, and staging."""

import json

import pytest

import _common
import _kaggle
import kaggle_upload_dataset


def make_config(**overrides):
    config = {
        "task": {
            "name": "mini",
            "tools": "tools.json",
            "slot_banks": "banks.json",
            "seed_bank": "train_seeds.jsonl",
            "eval_bank": "eval_seeds.jsonl",
            "out_dir": "out",
        },
        "kaggle": {
            "username": "someone",
            "input_dataset": "mini-input",
            "output_dataset": "mini-outputs",
            "kernel": "mini-train",
        },
    }
    config.update(overrides)
    return config


def make_task_tree(tmp_path):
    """Repo-like tree: <root>/tasks/t + <root>/scripts/*.py."""
    root = tmp_path / "root"
    task_dir = root / "tasks" / "t"
    scripts_dir = root / "scripts"
    task_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    for name in (
        "task.toml",
        "tools.json",
        "slot_banks.json",
        "train_seeds.jsonl",
        "eval_seeds.jsonl",
    ):
        (task_dir / name).write_text("{}", encoding="utf-8")
    (scripts_dir / "_common.py").write_text("x = 1\n", encoding="utf-8")
    (scripts_dir / "generate_data.py").write_text("y = 2\n", encoding="utf-8")
    return task_dir


def test_dataset_ref():
    config = make_config()
    assert _kaggle.dataset_ref(config, "input_dataset") == "someone/mini-input"
    assert (
        _kaggle.dataset_ref(config, "output_dataset") == "someone/mini-outputs"
    )


def test_render_dataset_metadata():
    assert _kaggle.render_dataset_metadata("o/s", "Title") == {
        "title": "Title",
        "id": "o/s",
        "licenses": [{"name": "CC0-1.0"}],
    }


def test_title_from_slug():
    assert _kaggle.title_from_slug("mini-input") == "mini input"
    assert _kaggle.title_from_slug("a-b-c") == "a b c"


def test_push_dataset_argv_create_and_version(tmp_path):
    create = _kaggle.push_dataset_argv("kaggle", False, tmp_path, "msg")
    assert create == ["kaggle", "datasets", "create", "-p", str(tmp_path)]
    version = _kaggle.push_dataset_argv("kaggle", True, tmp_path, "msg")
    assert version == [
        "kaggle", "datasets", "version",
        "-p", str(tmp_path), "-m", "msg",
    ]


def test_run_or_print_dry_run(capsys):
    assert _kaggle.run_or_print(["kaggle", "x"], True) == 0
    assert "kaggle x" in capsys.readouterr().out


def test_stage_input_dataset_copies_bundle(tmp_path):
    task_dir = make_task_tree(tmp_path)
    staging = tmp_path / "staging"
    kaggle_upload_dataset.stage_input_dataset(
        make_config(), task_dir, staging
    )
    names = sorted(path.name for path in staging.iterdir())
    assert names == [
        "_common.py",
        "dataset-metadata.json",
        "eval_seeds.jsonl",
        "generate_data.py",
        "slot_banks.json",
        "task.toml",
        "tools.json",
        "train_seeds.jsonl",
    ]
    with open(staging / "dataset-metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["id"] == "someone/mini-input"
    assert metadata["title"] == "mini input"
