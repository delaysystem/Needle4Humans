"""Tests for scripts/kaggle_train_loop.py checkpoint format (numpy-only)."""

import json

import numpy as np
import pytest

import kaggle_train_loop


def test_resolve_devices():
    assert kaggle_train_loop.resolve_devices("auto", 2) == 2
    assert kaggle_train_loop.resolve_devices("auto", 1) == 1
    assert kaggle_train_loop.resolve_devices("1", 8) == 1
    with pytest.raises(SystemExit):
        kaggle_train_loop.resolve_devices("2", 2)


def test_check_backend():
    kaggle_train_loop.check_backend("auto", "cpu")
    kaggle_train_loop.check_backend("tpu", "tpu")
    kaggle_train_loop.check_backend("gpu", "gpu")
    with pytest.raises(SystemExit):
        kaggle_train_loop.check_backend("tpu", "cpu")
    with pytest.raises(SystemExit):
        kaggle_train_loop.check_backend("gpu", "cpu")
    with pytest.raises(SystemExit):
        kaggle_train_loop.check_backend("bogus", "cpu")


def test_build_parser_defaults():
    base = [
        "data.jsonl", "--epochs", "1", "--batch-size", "2", "--lr", "0.1",
        "--lora-rank", "4", "--lora-alpha", "4", "--max-len", "128",
        "--val-split", "0.1", "--seed", "0", "--out", "a.safetensors",
        "--checkpoint-dir", "ckpts", "--save-every-min", "30",
    ]
    args = kaggle_train_loop.build_parser().parse_args(base)
    assert (args.devices, args.init_adapter, args.val_every) == (
        "auto", None, 1)
    assert args.expect_backend == "auto"
    args = kaggle_train_loop.build_parser().parse_args(
        base + ["--devices", "1", "--val-every", "0"]
    )
    assert (args.devices, args.val_every) == ("1", 0)


def test_apply_backend_tuning():
    from types import SimpleNamespace

    cuda = SimpleNamespace(flash=True, remat=True, scan_unroll=1,
                           num_layers=20)
    kaggle_train_loop.apply_backend_tuning(cuda, "cuda")
    # CUDA keeps vendor memory settings: no flash hardware on T4, so
    # remat stays on (v14 OOMed with it off).
    assert (cuda.flash, cuda.remat, cuda.scan_unroll) == (True, True, 1)
    metal = SimpleNamespace(flash=True, remat=True, scan_unroll=1,
                            num_layers=20)
    kaggle_train_loop.apply_backend_tuning(metal, "metal")
    assert (metal.flash, metal.remat, metal.scan_unroll) == (False, False, 20)


def test_shard_batch_even_split():
    ids = np.arange(32 * 8, dtype=np.int32).reshape(32, 8)
    masks = np.ones((32, 8), dtype=np.float32)
    sh_ids, sh_masks = kaggle_train_loop.shard_batch(ids, masks, 2)
    assert sh_ids.shape == (2, 16, 8)
    assert sh_masks.shape == (2, 16, 8)
    assert sh_ids.dtype == np.int32
    # Row order preserved across the split.
    assert (sh_ids.reshape(32, 8) == ids).all()


def test_shard_batch_tail_pads_with_zero_mask():
    ids = np.arange(31 * 8, dtype=np.int32).reshape(31, 8) + 1
    masks = np.ones((31, 8), dtype=np.float32)
    sh_ids, sh_masks = kaggle_train_loop.shard_batch(ids, masks, 2)
    assert sh_ids.shape == (2, 16, 8)
    # First 31 rows intact, last row padded with mask 0 (loss-neutral).
    flat_ids = sh_ids.reshape(32, 8)
    flat_masks = sh_masks.reshape(32, 8)
    assert (flat_ids[:31] == ids).all()
    assert (flat_ids[31:] == 0).all()
    assert (flat_masks[:31] == 1).all()
    assert (flat_masks[31:] == 0).all()


def test_shard_batch_single_row_single_device():
    ids = np.ones((5, 4), dtype=np.int32)
    masks = np.ones((5, 4), dtype=np.float32)
    sh_ids, sh_masks = kaggle_train_loop.shard_batch(ids, masks, 1)
    assert sh_ids.shape == (1, 5, 4)
    assert (sh_masks == 1).all()


def make_warm_shapes():
    return {
        ("a", "b"): {"A": (4, 2), "B": (2, 4)},
        ("c",): {"A": (3, 2), "B": (2, 3)},
    }


def make_warm_adapter():
    return {
        "lora": {
            "a/b": {
                "A": np.ones((4, 2), dtype=np.float32),
                "B": np.zeros((2, 4), dtype=np.float32),
            },
            "c": {
                "A": np.ones((3, 2), dtype=np.float32),
                "B": np.zeros((2, 3), dtype=np.float32),
            },
        }
    }


def test_warm_lora_from_adapter_ok():
    warm = kaggle_train_loop.warm_lora_from_adapter(
        make_warm_adapter(), make_warm_shapes()
    )
    assert set(warm) == {("a", "b"), ("c",)}
    assert warm[("a", "b")]["A"].shape == (4, 2)
    assert (warm[("c",)]["B"] == 0).all()


def test_warm_lora_from_adapter_refuses_mismatch():
    shapes = make_warm_shapes()
    bad_shape = make_warm_adapter()
    bad_shape["lora"]["c"]["B"] = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(SystemExit):
        kaggle_train_loop.warm_lora_from_adapter(bad_shape, shapes)
    extra = make_warm_adapter()
    extra["lora"]["zzz"] = {
        "A": np.ones((1, 1), dtype=np.float32),
        "B": np.ones((1, 1), dtype=np.float32),
    }
    with pytest.raises(SystemExit):
        kaggle_train_loop.warm_lora_from_adapter(extra, shapes)
    missing = make_warm_adapter()
    del missing["lora"]["c"]
    with pytest.raises(SystemExit):
        kaggle_train_loop.warm_lora_from_adapter(missing, shapes)


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


def test_checkpoint_paths_format_steps(tmp_path):
    loop = kaggle_train_loop
    assert loop._npz_path(tmp_path, 0).name == "ckpt-000000.npz"
    assert loop._meta_path(tmp_path, 7).name == "ckpt-000007.json"
    assert loop._latest_path(tmp_path).name == "latest.json"


def test_checkpoint_paths_large_step_not_truncated(tmp_path):
    name = kaggle_train_loop._npz_path(tmp_path, 1234567).name
    assert name == "ckpt-1234567.npz"


def test_checkpoint_paths_join_dir(tmp_path):
    sub = tmp_path / "ckpts"
    assert kaggle_train_loop._npz_path(sub, 3).parent == sub


@pytest.mark.parametrize("available", [0, 1, 8])
def test_resolve_devices_auto_bounds(available):
    assert kaggle_train_loop.resolve_devices("auto", available) == max(1, available)


def test_resolve_devices_one_ignores_available():
    assert kaggle_train_loop.resolve_devices("1", 0) == 1


@pytest.mark.parametrize("bad", ["0", "all", "", "2 ", "auto ", "AUTO"])
def test_resolve_devices_rejects(bad):
    with pytest.raises(SystemExit):
        kaggle_train_loop.resolve_devices(bad, 2)


def test_check_backend_auto_skips_unknown_actual():
    kaggle_train_loop.check_backend("auto", "weird-backend")


def test_check_backend_cpu_match():
    kaggle_train_loop.check_backend("cpu", "cpu")


def test_advance_past_completed_epoch_overrun():
    advance = kaggle_train_loop.advance_past_completed_epoch
    assert advance(3, 20, 16) == (4, 0, True)


def test_advance_past_completed_epoch_start_zero():
    advance = kaggle_train_loop.advance_past_completed_epoch
    assert advance(0, 0, 16) == (0, 0, False)
