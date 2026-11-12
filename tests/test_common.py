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
