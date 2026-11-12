"""Tests for scripts/kaggle_publish_outputs.py staging."""

import json

import pytest

import kaggle_publish_outputs


def make_config():
    return {
        "task": {"name": "mini", "out_dir": "out"},
        "kaggle": {
            "username": "someone",
            "output_dataset": "mini-outputs",
            "publish_files": ["a.bin", "report.json"],
        },
    }


def test_stage_output_dataset(tmp_path):
    out_dir = tmp_path / "t" / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "a.bin").write_text("x", encoding="utf-8")
    (out_dir / "report.json").write_text("{}", encoding="utf-8")
    staging = tmp_path / "staging"
    kaggle_publish_outputs.stage_output_dataset(
        make_config(), tmp_path / "t", staging
    )
    assert (staging / "a.bin").is_file()
    assert (staging / "report.json").is_file()
    with open(staging / "dataset-metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["id"] == "someone/mini-outputs"
    assert metadata["title"] == "mini outputs"


def test_stage_output_dataset_missing_dies(tmp_path):
    out_dir = tmp_path / "t" / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "a.bin").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        kaggle_publish_outputs.stage_output_dataset(
            make_config(), tmp_path / "t", tmp_path / "staging"
        )
