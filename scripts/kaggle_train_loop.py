#!/usr/bin/env python3
"""LoRA training loop with periodic saves and crash resume.

Same training math as ``cactus finetune`` — same model, numerics, LoRA
targets, optimizer, schedule, and data pipeline, imported from the
installed ``cactus-compute`` package (pinned by the driver) rather than
reimplemented — plus what the stock trainer lacks: the adapter,
optimizer state, RNG state, and loop position are saved every
``--save-every-min`` minutes (crash-safe via write-temp-then-rename,
latest two kept), and a rerun automatically resumes from the latest
complete checkpoint instead of restarting from step 0.

Two speedups keep single-device math exactly: ``--devices auto`` (the
default) shards each batch across every local device with ``pmap``
(the per-shard loss normalizes by the cross-device mask sum, combined
with psum, so updates match the 1-device path), and ``--init-adapter``
warm-starts the LoRA weights from a prior run's adapter.

Checkpoint layout under ``--checkpoint-dir``::

    ckpt-<step:06d>.npz    lora A/B, optimizer leaves, order arrays
    ckpt-<step:06d>.json   epoch, position, step, rng state, versions
    latest.json            pointer at the newest complete checkpoint

Only numpy/json are imported at module level so the checkpoint format
stays unit-testable; jax/cactus/optax import lazily inside ``train``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Origin: task choice; matches write_adapter's lora/<path>/A convention.
LORA_KEY_PREFIX = "lora/"
OPT_KEY_PREFIX = "opt/"

# Origin: task choice; two checkpoints bound disk while surviving one
# corrupt write (latest.json always points at a complete pair).
KEEP_CHECKPOINTS = 2


def _npz_path(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"ckpt-{step:06d}.npz"


def _meta_path(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"ckpt-{step:06d}.json"


def _latest_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "latest.json"


def save_checkpoint(ckpt_dir: Path, step: int, arrays: dict, meta: dict) -> Path:
    """Write one checkpoint pair plus the latest pointer, crash-safely.

    ``arrays`` maps npz key to array (lora weights, optimizer leaves,
    order arrays). Files land under temp names and are renamed last, so
    a kill mid-save never corrupts the previous checkpoint; only the
    newest ``KEEP_CHECKPOINTS`` pairs are retained.
    """
    import numpy as np

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # savez appends .npz unless present, so the temp name keeps the suffix.
    npz_tmp = ckpt_dir / f"ckpt-{step:06d}.tmp.npz"
    meta_tmp = _meta_path(ckpt_dir, step).with_suffix(".json.tmp")
    np.savez(npz_tmp, **{key: np.asarray(value) for key, value in arrays.items()})
    os.replace(npz_tmp, _npz_path(ckpt_dir, step))
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(meta_tmp, _meta_path(ckpt_dir, step))
    latest_tmp = _latest_path(ckpt_dir).with_suffix(".json.tmp")
    latest_tmp.write_text(
        json.dumps({"step": step}, indent=2), encoding="utf-8"
    )
    os.replace(latest_tmp, _latest_path(ckpt_dir))
    pairs = sorted(
        int(path.stem.split("-")[1])
        for path in ckpt_dir.glob("ckpt-*.npz")
        if not path.name.endswith(".tmp.npz")
    )
    for old in pairs[:-KEEP_CHECKPOINTS]:
        _npz_path(ckpt_dir, old).unlink(missing_ok=True)
        _meta_path(ckpt_dir, old).unlink(missing_ok=True)
    return _npz_path(ckpt_dir, step)


def _load_pair(ckpt_dir: Path, step: int):
    """Load one checkpoint pair, or None when incomplete."""
    import numpy as np

    npz_path = _npz_path(ckpt_dir, step)
    meta_path = _meta_path(ckpt_dir, step)
    if not npz_path.is_file() or not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return arrays, meta


def load_latest_checkpoint(ckpt_dir: Path):
    """Load the newest complete checkpoint, else an older one, else None.

    Falls back past half-written pairs (a kill between the npz and the
    pointer update) instead of failing the resume.
    """
    if not _latest_path(ckpt_dir).is_file():
        return None
    with open(_latest_path(ckpt_dir), encoding="utf-8") as handle:
        latest = json.load(handle).get("step")
    candidates = sorted(
        (
            int(path.stem.split("-")[1])
            for path in ckpt_dir.glob("ckpt-*.npz")
            if not path.name.endswith(".tmp.npz")
        ),
        reverse=True,
    )
    if latest in candidates:
        candidates.remove(latest)
        candidates.insert(0, latest)
    for step in candidates:
        pair = _load_pair(ckpt_dir, step)
        if pair is not None:
            return step, pair[0], pair[1]
    return None


def _lora_to_numpy(lora):
    """LoRA {path: {A, B}} device arrays to flat npz-ready dict."""
    import numpy as np

    flat = {}
    for path, adapter in lora.items():
        name = "/".join(path)
        flat[f"{LORA_KEY_PREFIX}{name}/A"] = np.asarray(adapter["A"])
        flat[f"{LORA_KEY_PREFIX}{name}/B"] = np.asarray(adapter["B"])
    return flat


def _lora_from_numpy(arrays):
    """Rebuild {path-tuple: {A, B}} device arrays from an npz dict."""
    import jax.numpy as jnp

    names = {
        key[len(LORA_KEY_PREFIX):-len("/A")]
        for key in arrays
        if key.startswith(LORA_KEY_PREFIX) and key.endswith("/A")
    }
    lora = {}
    for name in names:
        path = tuple(name.split("/"))
        lora[path] = {
            "A": jnp.asarray(arrays[f"{LORA_KEY_PREFIX}{name}/A"]),
            "B": jnp.asarray(arrays[f"{LORA_KEY_PREFIX}{name}/B"]),
        }
    return lora


def _opt_leaves_to_numpy(opt_state):
    import jax
    import numpy as np

    return [np.asarray(leaf) for leaf in jax.tree.leaves(opt_state)]


def _opt_leaves_from_numpy(optimizer, lora, saved):
    """Fresh optimizer state with saved leaves overlaid.

    The tree structure comes from a fresh init (deterministic given the
    same optimizer and adapter shapes); only leaf values cross the
    save boundary, so a structural mismatch fails loudly here instead
    of silently corrupting training.
    """
    import jax
    import jax.numpy as jnp

    fresh = optimizer.init(lora)
    leaves, treedef = jax.tree.flatten(fresh)
    if len(leaves) != len(saved):
        raise SystemExit(
            f"optimizer state has {len(saved)} saved leaves but fresh "
            f"init has {len(leaves)}; refusing to resume across versions"
        )
    return jax.tree.unflatten(
        treedef, [jnp.asarray(saved[f"{OPT_KEY_PREFIX}{i}"]) for i in range(len(saved))]
    )


def _snapshot(ckpt_dir, step_i, lora, opt_state, order_full, order, epoch,
              start, total_steps, seed, rng):
    """Save one complete checkpoint and prune old ones."""
    arrays = _lora_to_numpy(lora)
    for i, leaf in enumerate(_opt_leaves_to_numpy(opt_state)):
        arrays[f"{OPT_KEY_PREFIX}{i}"] = leaf
    arrays["split_order"] = order_full
    arrays["epoch_order"] = order
    meta = {
        "epoch": epoch,
        "start": start,
        "step": step_i,
        "total_steps": total_steps,
        "seed": seed,
        "rng_state": rng.bit_generator.state,
    }
    save_checkpoint(ckpt_dir, step_i, arrays, meta)


def resolve_devices(requested: str, available: int) -> int:
    """Data-parallel device count for training.

    ``auto`` uses every local device; ``1`` forces the single-device
    path (identical math, easier debugging). Anything else dies:
    silently ignoring a device request would fake a promised speedup.
    """
    if requested == "1":
        return 1
    if requested == "auto":
        return max(1, available)
    raise SystemExit(f"unknown --devices: {requested!r} (want auto or 1)")


def shard_batch(ids, masks, n_devices):
    """Pad one batch to a device multiple and split its leading axis.

    ``ids``/``masks`` are host numpy row batches; padding rows carry
    mask 0 so they contribute nothing to the masked loss. Returns
    ``(D, per, L)`` arrays. The pmap loss normalizes by the cross-device
    mask sum, so training math matches the single-device path exactly.
    """
    import numpy as np

    count = len(ids)
    per = max(1, -(-count // n_devices))
    pad = per * n_devices - count
    if pad:
        ids = np.concatenate(
            [ids, np.zeros((pad, ids.shape[1]), ids.dtype)]
        )
        masks = np.concatenate(
            [masks, np.zeros((pad, masks.shape[1]), masks.dtype)]
        )
    width = ids.shape[1]
    return (
        np.asarray(ids).reshape(n_devices, per, width),
        np.asarray(masks).reshape(n_devices, per, width),
    )


def warm_lora_from_adapter(adapter: dict, fresh_shapes: dict) -> dict:
    """Warm-start LoRA weights from a ``read_adapter()`` dict.

    ``fresh_shapes`` maps path-tuple to ``{"A": shape, "B": shape}``
    from a fresh ``init_lora``. Every adapter entry must match its
    target shape exactly, and the adapter must cover every target: a
    partial warm start would mix trained and random weights silently.
    Returns ``{path-tuple: {"A", "B"}}`` host-numpy arrays.
    """
    import numpy as np

    lora = {}
    for name, mats in adapter["lora"].items():
        path = tuple(name.split("/"))
        if path not in fresh_shapes:
            raise SystemExit(
                f"warm-start adapter has unexpected target {name!r}"
            )
        entry = {}
        for key in ("A", "B"):
            want = tuple(fresh_shapes[path][key])
            got = tuple(np.shape(mats[key]))
            if got != want:
                raise SystemExit(
                    f"warm-start adapter {name}/{key} shape {got} "
                    f"does not match fresh init {want}"
                )
            entry[key] = np.asarray(mats[key])
        lora[path] = entry
    missing = [path for path in fresh_shapes if path not in lora]
    if missing:
        names = sorted("/".join(path) for path in missing)
        raise SystemExit(
            f"warm-start adapter lacks targets {names}; refusing a "
            "partial warm start"
        )
    return lora


def apply_backend_tuning(config, backend: str) -> None:
    """Speed toggles per backend; training math stays identical.

    Metal keeps the stock finetune path (no flash, no remat, full
    unroll). CUDA keeps the vendor memory settings: the T4 has no
    flash-attention hardware (Turing, sm_75 — needs Ampere+), so
    attention materializes full seq^2 score matrices and remat is
    the only thing keeping them out of memory (disabling it OOMs
    at 84 GB on the first step; measured on v14 2026-11-13).
    """
    if backend == "metal":
        config.flash = False
        config.remat = False
        config.scan_unroll = config.num_layers


def advance_past_completed_epoch(start_epoch, start_pos, count):
    """Resume position, skipping an epoch that already completed.

    A checkpoint written on an epoch's final batch records start == count;
    resuming must continue with the NEXT epoch (whose order the caller
    draws fresh, exactly as an uninterrupted run would), not re-run an
    empty epoch or print a fabricated loss line.
    """
    if start_pos >= count:
        return start_epoch + 1, 0, True
    return start_epoch, start_pos, False


def _cactus_version():
    from importlib.metadata import version

    try:
        return version("cactus-compute")
    except Exception:  # noqa: BLE001 - best effort label
        return "unknown"


def train(args) -> int:
    """Train a LoRA adapter, saving periodically, resuming when present."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from cactus.model.architecture import SimpleAttentionNetwork
    from cactus.model.checkpoints import write_adapter
    from cactus.model.finetune import (
        DEFAULT_BASE,
        fit_max_len,
        init_lora,
        load_jsonl,
        lora_target_paths,
        merge_lora,
    )
    from cactus.model.quantize import WEIGHT_BITS, configure_deploy, cq_ste_params
    from cactus.model.run import load_checkpoint
    from cactus.model.tokenizer import get_tokenizer

    def emit(msg):
        print(msg, flush=True)

    emit(f"  versions  cactus-compute {_cactus_version()}  jax {jax.__version__}")
    base_path = args.checkpoint or DEFAULT_BASE
    ckpt_dir = Path(args.checkpoint_dir)
    resumed = load_latest_checkpoint(ckpt_dir)
    if resumed is not None:
        _, saved_arrays, saved_meta = resumed
        emit(f"  resume    step {saved_meta['step']} of {saved_meta['total_steps']}")
    if resumed is not None and args.init_adapter:
        raise SystemExit(
            "resume restores weights from the checkpoint; --init-adapter "
            "is only for fresh runs"
        )

    params, config = load_checkpoint(base_path)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    backend = jax.default_backend().lower()
    apply_backend_tuning(config, backend)
    params = jax.device_put(params)
    emit(f"  {'backend':<9} {backend}  float32")
    emit(f"  {'tuning':<9} remat {config.remat}  flash {config.flash}  "
         f"unroll {config.scan_unroll}")
    emit(f"  {'depth':<9} {config.num_layers} layers")
    tokenizer = get_tokenizer(config.vocab_size)
    max_len = fit_max_len(args.jsonl_path, tokenizer, args.max_len)
    seqs, masks = load_jsonl(args.jsonl_path, tokenizer, max_len)
    if len(seqs) == 0:
        raise SystemExit("no usable examples in " + args.jsonl_path)
    emit(f"  {'data':<9} {len(seqs)} examples  seq_len {max_len}  cap {args.max_len}")

    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=getattr(config, "act_bits", 8),
                     kv_bits=getattr(config, "kv_bits", 8))
    emit(f"  {'numerics':<9} CQ W{WEIGHT_BITS} STE + A8 (matches export)")
    paths = lora_target_paths(params)
    scale = args.lora_alpha / args.lora_rank
    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    batch, epochs = args.batch_size, args.epochs
    n_val = min(int(len(seqs) * args.val_split), len(seqs) - 1)
    if resumed is None:
        lora = init_lora(params, paths, args.lora_rank, jax.random.PRNGKey(seed))
        emit(f"  {'lora':<9} rank {args.lora_rank}  alpha {args.lora_alpha:g}  "
             f"{len(paths)} weight groups")
        if args.init_adapter:
            from cactus.model.checkpoints import read_adapter

            fresh_shapes = {
                path: {"A": value["A"].shape, "B": value["B"].shape}
                for path, value in lora.items()
            }
            warm = warm_lora_from_adapter(
                read_adapter(args.init_adapter), fresh_shapes
            )
            lora = jax.tree.map(jnp.asarray, warm)
            emit(f"  {'warm':<9} start from {args.init_adapter}")
        if n_val > 0:
            order_full = rng.permutation(len(seqs))
            seqs, masks = seqs[order_full], masks[order_full]
            val_seqs, val_masks = seqs[:n_val], masks[:n_val]
            seqs, masks = seqs[n_val:], masks[n_val:]
            emit(f"  {'holdout':<9} {n_val} examples for validation")
        else:
            order_full = np.arange(len(seqs))
            val_seqs, val_masks = seqs[:0], masks[:0]
        count = len(seqs)
        steps_per_epoch = -(-count // batch)
        total_steps = epochs * steps_per_epoch
        start_epoch, start_pos, step_i = 0, 0, 0
        epoch_order = rng.permutation(count)
    else:
        lora = _lora_from_numpy(saved_arrays)
        order_full = np.asarray(saved_arrays["split_order"])
        epoch_order = np.asarray(saved_arrays["epoch_order"])
        seqs, masks = seqs[order_full], masks[order_full]
        if n_val > 0:
            val_seqs, val_masks = seqs[:n_val], masks[:n_val]
            seqs, masks = seqs[n_val:], masks[n_val:]
        else:
            val_seqs, val_masks = seqs[:0], masks[:0]
        count = len(seqs)
        total_steps = saved_meta["total_steps"]
        start_epoch = saved_meta["epoch"]
        start_pos = saved_meta["start"]
        step_i = saved_meta["step"]
        rng.bit_generator.state = saved_meta["rng_state"]
        start_epoch, start_pos, skipped = advance_past_completed_epoch(
            start_epoch, start_pos, count
        )
        if skipped:
            epoch_order = rng.permutation(count)
        emit(f"  {'lora':<9} restored {len(lora)} adapters at step {step_i}")
    warmup = min(max(1, total_steps // 20), total_steps - 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=warmup, decay_steps=total_steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    if resumed is None:
        opt_state = optimizer.init(lora)
    else:
        saved_leaves = {
            key: value for key, value in saved_arrays.items()
            if key.startswith(OPT_KEY_PREFIX)
        }
        opt_state = _opt_leaves_from_numpy(optimizer, lora, saved_leaves)
    emit(f"  {'schedule':<9} {total_steps} steps  warmup {warmup}  cosine decay  "
         f"clip 1.0  (compiling...)")
    n_devices = resolve_devices(args.devices, jax.device_count())
    use_pmap = n_devices > 1
    if use_pmap:
        lora = jax.tree.map(lambda x: jnp.stack([x] * n_devices), lora)
        opt_state = jax.tree.map(lambda x: jnp.stack([x] * n_devices), opt_state)
        emit(f"  {'devices':<9} {n_devices} x {backend}  data-parallel")
    else:
        emit(f"  {'devices':<9} 1 x {backend}")

    def first(tree):
        """Device-0 replica back on host (checkpoints stay unreplicated)."""
        return jax.tree.map(lambda x: np.asarray(x)[0], tree)

    def loss_fn(lora, ids, mask):
        merged = cq_ste_params(merge_lora(params, lora, scale), WEIGHT_BITS)
        logits = model.apply({"params": merged}, ids, quant=True)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    def ploss_fn(lora, ids, mask):
        """Per-shard loss normalized by the cross-device mask sum.

        Each device holds sum(shard)/M with M the whole-batch mask
        total, so a psum over devices reproduces the single-device
        loss and grads EXACTLY (a pmean would shrink both by 1/D).
        """
        merged = cq_ste_params(merge_lora(params, lora, scale), WEIGHT_BITS)
        logits = model.apply({"params": merged}, ids, quant=True)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        total = jax.lax.psum(mask.sum(), "shard")
        return (ce * mask).sum() / jnp.maximum(total, 1.0)

    @jax.jit
    def train_step(lora, opt_state, ids, mask):
        loss, grads = jax.value_and_grad(loss_fn)(lora, ids, mask)
        updates, opt_state = optimizer.update(grads, opt_state, lora)
        return optax.apply_updates(lora, updates), opt_state, loss

    def _ptrain_step(lora, opt_state, ids, mask):
        # psum, not pmean: per-shard grads are already divided by the
        # whole-batch total, so the sum (not the mean) matches the
        # single-device update. Identical grads on every device keep
        # the clip norm and the replicas in sync.
        loss, grads = jax.value_and_grad(ploss_fn)(lora, ids, mask)
        grads = jax.lax.psum(grads, "shard")
        updates, opt_state = optimizer.update(grads, opt_state, lora)
        updated = optax.apply_updates(lora, updates)
        return updated, opt_state, jax.lax.psum(loss, "shard")

    ptrain_step = jax.pmap(_ptrain_step, axis_name="shard")
    eval_step = jax.jit(loss_fn)
    every = max(1, total_steps // 50)
    save_every_s = args.save_every_min * 60
    last_save = time.monotonic()
    for epoch in range(start_epoch, epochs):
        if epoch == start_epoch:
            order = epoch_order
        else:
            order = rng.permutation(count)
        first = start_pos if epoch == start_epoch else 0
        last = 0.0
        for start in range(first, count, batch):
            idx = order[start:start + batch]
            if use_pmap:
                sh_ids, sh_masks = shard_batch(seqs[idx], masks[idx], n_devices)
                lora, opt_state, loss = ptrain_step(
                    lora, opt_state, jnp.asarray(sh_ids), jnp.asarray(sh_masks))
            else:
                lora, opt_state, loss = train_step(lora, opt_state,
                                                   jnp.asarray(seqs[idx]),
                                                   jnp.asarray(masks[idx]))
            last = float(np.asarray(loss).reshape(-1)[0])
            step_i += 1
            if step_i % every == 0:
                emit(f"  {'step':<9} {step_i}/{total_steps}  loss {last:.4f}")
            if time.monotonic() - last_save >= save_every_s:
                snap_lora = first(lora) if use_pmap else lora
                snap_opt = first(opt_state) if use_pmap else opt_state
                _snapshot(ckpt_dir, step_i, snap_lora, snap_opt, order_full,
                          order, epoch, start + batch, total_steps, seed, rng)
                last_save = time.monotonic()
                emit(f"  {'save':<9} checkpoint at step {step_i}")
        score_lora = first(lora) if use_pmap else lora
        if n_val > 0:
            val = np.mean([float(eval_step(score_lora, jnp.asarray(val_seqs[i:i + batch]),
                                           jnp.asarray(val_masks[i:i + batch])))
                           for i in range(0, n_val, batch)])
            emit(f"  {'epoch':<9} {epoch + 1}/{epochs}  loss {last:.4f}  val {val:.4f}")
        else:
            emit(f"  {'epoch':<9} {epoch + 1}/{epochs}  loss {last:.4f}")

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if use_pmap:
        lora = first(lora)
    write_adapter(out, {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])}
                 for p, v in lora.items()},
        "scale": float(scale),
        "base": base_path,
        "rank": args.lora_rank,
        "seed": seed,
    })
    emit(f"  {'adapter':<9} {out}")
    emit("  note      held-out scoring left to evaluate.py")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", help="Path to JSONL training data")
    parser.add_argument("--checkpoint", default=None,
                        help="Base model checkpoint (auto-downloads if omitted)")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=float, required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--val-split", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output adapter path")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Periodic checkpoint directory")
    parser.add_argument("--save-every-min", type=float, required=True,
                        help="Minutes between checkpoints (0 disables)")
    parser.add_argument("--devices", default="auto",
                        help="auto uses every local device (pmap "
                        "data-parallel); 1 forces single-device")
    parser.add_argument("--init-adapter", default=None,
                        help="Warm-start LoRA weights from a prior "
                        ".safetensors adapter (fresh runs only)")
    args = parser.parse_args(argv)
    if args.save_every_min == 0:
        args.save_every_min = float("inf")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
