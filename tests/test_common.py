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
