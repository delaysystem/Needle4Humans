"""Tests for scripts/kaggle_train_loop.py checkpoint format (numpy-only)."""

import json

import numpy as np
import pytest

import kaggle_train_loop


def make_payload():
    return (
        {
            "lora/layers/0/A": np.ones((4, 2), dtype=np.float32),
            "lora/layers/0/B": np.zeros((2, 4), dtype=np.float32),
            "opt/0": np.full((4, 2), 0.5, dtype=np.float32),
            "split_order": np.arange(10),
            "epoch_order": np.arange(9, -1, -1),
        },
        {
            "epoch": 3,
            "start": 8,
            "step": 100,
            "total_steps": 500,
            "seed": 0,
            "rng_state": np.random.default_rng(0).bit_generator.state,
        },
    )


def test_save_load_round_trip(tmp_path):
    arrays, meta = make_payload()
    kaggle_train_loop.save_checkpoint(tmp_path, 100, arrays, meta)
    loaded = kaggle_train_loop.load_latest_checkpoint(tmp_path)
    assert loaded is not None
    step, got_arrays, got_meta = loaded
    assert step == 100
    assert set(got_arrays) == set(arrays)
    for key in arrays:
        np.testing.assert_array_equal(got_arrays[key], arrays[key])
    assert got_meta["epoch"] == 3
    assert got_meta["start"] == 8
    assert got_meta["total_steps"] == 500
    assert got_meta["rng_state"]["bit_generator"] == "PCG64"


def test_rng_state_json_round_trip():
    """PCG64 bit-generator state must survive JSON (checkpoint meta)."""
    state = np.random.default_rng(0).bit_generator.state
    assert json.loads(json.dumps(state)) == state


def test_prune_keeps_latest_two(tmp_path):
    arrays, meta = make_payload()
    for step in (10, 20, 30):
        meta = dict(meta, step=step)
        kaggle_train_loop.save_checkpoint(tmp_path, step, arrays, meta)
    assert not (tmp_path / "ckpt-000010.npz").exists()
    assert not (tmp_path / "ckpt-000010.json").exists()
    assert (tmp_path / "ckpt-000020.npz").exists()
    assert (tmp_path / "ckpt-000030.npz").exists()
    step, _, _ = kaggle_train_loop.load_latest_checkpoint(tmp_path)
    assert step == 30


def test_load_falls_back_past_half_written(tmp_path):
    arrays, meta = make_payload()
    kaggle_train_loop.save_checkpoint(tmp_path, 10, arrays, dict(meta, step=10))
    kaggle_train_loop.save_checkpoint(tmp_path, 20, arrays, dict(meta, step=20))
    # Simulate a kill between the npz write and the meta write.
    (tmp_path / "ckpt-000020.json").unlink()
    (tmp_path / "latest.json").write_text(json.dumps({"step": 20}))
    step, _, got_meta = kaggle_train_loop.load_latest_checkpoint(tmp_path)
    assert step == 10
    assert got_meta["step"] == 10


def test_load_empty_is_none(tmp_path):
    assert kaggle_train_loop.load_latest_checkpoint(tmp_path) is None
    tmp_path.joinpath("latest.json").write_text(json.dumps({"step": 5}))
    assert kaggle_train_loop.load_latest_checkpoint(tmp_path) is None


def test_advance_past_completed_epoch():
    advance = kaggle_train_loop.advance_past_completed_epoch
    # Mid-epoch checkpoint resumes in place.
    assert advance(3, 8, 16) == (3, 8, False)
    # Boundary checkpoint advances to the next epoch.
    assert advance(3, 16, 16) == (4, 0, True)
    # Boundary on the final epoch yields an empty range (straight to save).
    assert advance(9, 16, 16) == (10, 0, True)
