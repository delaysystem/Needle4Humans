"""Tests for scripts/_common.py helpers."""

import sys

import _common


def test_cactus_bin_prefers_venv_sibling(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "cactus"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    resolved = _common.cactus_bin()
    assert resolved == str(binary)


def test_cactus_bin_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    assert _common.cactus_bin() == "cactus"


def test_kaggle_bin_found(monkeypatch):
    monkeypatch.setattr(
        _common.shutil, "which", lambda _: "/home/u/.local/bin/kaggle"
    )
    assert _common.kaggle_bin() == "/home/u/.local/bin/kaggle"


def test_kaggle_bin_missing_dies(monkeypatch):
    import pytest

    monkeypatch.setattr(_common.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit):
        _common.kaggle_bin()


def test_check_no_overwrite_passes_when_absent(tmp_path):
    _common.check_no_overwrite(
        [tmp_path / "a.json", tmp_path / "b.json"], False
    )


def test_check_no_overwrite_dies_naming_files(tmp_path, capsys):
    import pytest

    existing = tmp_path / "report.json"
    existing.write_text("old", encoding="utf-8")
    with pytest.raises(SystemExit):
        _common.check_no_overwrite(
            [existing, tmp_path / "missing.json"], False
        )
    err = capsys.readouterr().err
    assert "report.json" in err
    assert "missing.json" not in err
    assert "--overwrite" in err


def test_check_no_overwrite_flag_allows(tmp_path):
    existing = tmp_path / "report.json"
    existing.write_text("old", encoding="utf-8")
    _common.check_no_overwrite([existing], True)
    assert existing.read_text(encoding="utf-8") == "old"


def test_cactus_bin_ignores_non_executable_sibling(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "cactus"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o644)
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    assert _common.cactus_bin() == "cactus"


def test_cactus_bin_ignores_directory_sibling(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "cactus").mkdir()
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    assert _common.cactus_bin() == "cactus"


def test_cactus_bin_returns_string(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    assert isinstance(_common.cactus_bin(), str)


def test_kaggle_bin_missing_names_fix(monkeypatch, capsys):
    import pytest

    monkeypatch.setattr(_common.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as exc:
        _common.kaggle_bin()
    assert exc.value.code == 1
    assert "uv tool install kaggle" in capsys.readouterr().err


def test_kaggle_bin_empty_string_path(monkeypatch):
    monkeypatch.setattr(_common.shutil, "which", lambda _: "")
    assert _common.kaggle_bin() == ""


def test_die_default_code_and_stderr(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        _common.die("boom")
    assert exc.value.code == 1
    assert "error: boom" in capsys.readouterr().err


def test_die_custom_code():
    import pytest

    with pytest.raises(SystemExit) as exc:
        _common.die("nope", code=3)
    assert exc.value.code == 3


def test_load_task_config_ok(tmp_path):
    task_file = tmp_path / "task.toml"
    task_file.write_text('[task]\nname = "x"\n', encoding="utf-8")
    config, task_dir = _common.load_task_config(str(task_file))
    assert config["task"]["name"] == "x"
    assert task_dir == tmp_path


def test_load_task_config_missing_dies(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        _common.load_task_config(str(tmp_path / "nope.toml"))


def test_load_task_config_invalid_toml_raises(tmp_path):
    import pytest
    import tomllib

    task_file = tmp_path / "task.toml"
    task_file.write_text("[[[broken", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        _common.load_task_config(str(task_file))


def test_load_task_config_directory_dies(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        _common.load_task_config(str(tmp_path))


def test_task_relative_path(tmp_path):
    assert _common.task_relative_path(tmp_path, "a/b.json") == tmp_path / "a" / "b.json"
    assert _common.task_relative_path(tmp_path / "sub", "..") == tmp_path
    assert _common.task_relative_path(tmp_path, ".") == tmp_path


def test_task_relative_path_resolves_symlink(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    assert _common.task_relative_path(link, "f") == target / "f"


def test_read_json_ok_and_invalid(tmp_path):
    import json
    import pytest

    path = tmp_path / "a.json"
    path.write_text('{"k": [1, 2]}', encoding="utf-8")
    assert _common.read_json(path) == {"k": [1, 2]}
    path.write_text("{nope", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _common.read_json(path)


def test_read_json_missing_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        _common.read_json(tmp_path / "nope.json")


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"a": 1}\n\n   \n\t\n{"b": 2}\n', encoding="utf-8")
    assert _common.read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_empty_file(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text("", encoding="utf-8")
    assert _common.read_jsonl(path) == []


def test_read_jsonl_invalid_line_raises(tmp_path):
    import json
    import pytest

    path = tmp_path / "a.jsonl"
    path.write_text('{"a": 1}\nnope\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _common.read_jsonl(path)


def test_write_jsonl_round_trip_unicode(tmp_path):
    rows = [{"q": "ünïcodé✓", "n": 1}, {"q": "plain"}]
    path = tmp_path / "deep" / "nest" / "a.jsonl"
    _common.write_jsonl(path, rows)
    raw = path.read_bytes()
    assert "ünïcodé✓".encode("utf-8") in raw
    assert _common.read_jsonl(path) == rows


def test_write_jsonl_empty_rows(tmp_path):
    path = tmp_path / "a.jsonl"
    _common.write_jsonl(path, [])
    assert path.read_text(encoding="utf-8") == ""


def test_add_task_argument_requires_value():
    import argparse
    import pytest

    parser = _common.add_task_argument(argparse.ArgumentParser())
    assert parser.parse_args(["--task", "t.toml"]).task == "t.toml"
    with pytest.raises(SystemExit):
        parser.parse_args([])
