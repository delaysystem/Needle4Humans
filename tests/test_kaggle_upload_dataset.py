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


def test_dataset_last_updated_parses(monkeypatch):
    import datetime

    rows = [
        {"ref": "o/other", "lastUpdated": "2026-11-13 00:00:00.000000"},
        {"ref": "o/s", "lastUpdated": "2026-11-13 02:23:26.810000"},
        {"ref": "o/bad", "lastUpdated": "not-a-date"},
    ]
    monkeypatch.setattr(_kaggle, "_fetch_dataset_rows", lambda k: rows)
    assert _kaggle.dataset_last_updated("kaggle", "o/s") == datetime.datetime(
        2026, 11, 13, 2, 23, 26
    )
    assert _kaggle.dataset_last_updated("kaggle", "o/missing") is None
    assert _kaggle.dataset_last_updated("kaggle", "o/bad") is None


def test_wait_for_dataset_updated_advance(monkeypatch):
    import datetime

    before = datetime.datetime(2026, 11, 13, 2, 0, 0)
    after = datetime.datetime(2026, 11, 13, 2, 23, 26)
    calls = iter([before, before, after])
    monkeypatch.setattr(
        _kaggle, "dataset_last_updated", lambda k, r: next(calls)
    )
    sleeps = []
    _kaggle.wait_for_dataset_updated(
        "kaggle", "o/s", before, timeout=60, sleep_fn=sleeps.append
    )
    assert sleeps[-1] == _kaggle.READY_GRACE_S
    assert len(sleeps) == 3


def test_wait_for_dataset_updated_timeout(monkeypatch):
    import datetime

    before = datetime.datetime(2026, 11, 13, 2, 0, 0)
    monkeypatch.setattr(
        _kaggle, "dataset_last_updated", lambda k, r: before
    )
    with pytest.raises(SystemExit):
        _kaggle.wait_for_dataset_updated(
            "kaggle", "o/s", before, timeout=0.05,
            sleep_fn=lambda s: None,
        )


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
