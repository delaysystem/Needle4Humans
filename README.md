# Needle 3 finetuning

Reusable scripts to finetune a Cactus Needle 3 model locally, plus a
complete test task: human-handoff detection in 7 languages (EN, DE, FR,
ES, IT, NL, PL) with sarcasm, implicit requests, multi-sentence context,
and edge cases.

How it works: every stage is one script, and every task-specific value
lives in the task's `task.toml`. A new finetune needs a new task
directory, not new code.

## Layout

- `scripts/check_env.py` — verify Python, the `cactus` package/CLI, the
  JAX backend, and GPU VRAM.
- `scripts/generate_data.py` — expand a task's seed banks into
  `train.jsonl`/`eval.jsonl` plus a manifest. Deterministic slot
  expansion, no external services.
- `scripts/validate_data.py` — schema, grounding, quota, negative-ratio,
  train/eval-overlap, and length checks.
- `scripts/finetune.py` — run `cactus finetune` (local LoRA) from config.
- `scripts/build.py` — run `cactus build` to export the `.cact`.
- `scripts/evaluate.py` — score base vs tuned weights on the frozen eval
  set (exact-calls-in-order).
- `scripts/kaggle_upload_dataset.py`, `kaggle_push.py`, `kaggle_poll.py`,
  `kaggle_publish_outputs.py` (+ `kaggle_train_driver.py`, which runs on
  Kaggle) — versioned datasets in/out, GPU kernel runs, status polling.
- `tasks/human_handoff/` — the test task: `task.toml`, `tools.json`,
  `slot_banks.json`, `seeds_train.jsonl`, `seeds_eval.jsonl`.
- `tests/` — unit tests mirroring the scripts.

## Setup

```sh
uv venv
. .venv/bin/activate
uv pip install 'cactus-compute[train,gpu]' pytest   # NVIDIA
# uv pip install 'cactus-compute[train]' pytest     # CPU / other
```

## Run (human-handoff task)

```sh
. .venv/bin/activate
export TASK=tasks/human_handoff/task.toml
python scripts/check_env.py
python scripts/generate_data.py --task $TASK
python scripts/validate_data.py --task $TASK
python scripts/finetune.py --task $TASK
python scripts/build.py --task $TASK
python scripts/evaluate.py --task $TASK
```

`finetune.py` and `build.py` accept `--dry-run` to print the `cactus`
argv without executing. `evaluate.py --limit N` smoke-tests the first
N eval cases.

## Run on Kaggle GPUs

Compile + training run in a private GPU kernel; datasets carry the
inputs and the binaries. Needs the CLI (`uv tool install kaggle`) and
a token at `~/.kaggle/access_token`:

```sh
export TASK=tasks/human_handoff/task.toml
python scripts/kaggle_upload_dataset.py --task $TASK   # version inputs
python scripts/kaggle_push.py --task $TASK             # push GPU run
python scripts/kaggle_poll.py --task $TASK             # wait + download
python scripts/kaggle_publish_outputs.py --task $TASK  # version binaries
python scripts/evaluate.py --task $TASK                # local, runtime-only
```

All four Kaggle scripts accept `--dry-run`; `kaggle_poll.py --once`
prints a single status line. Kernel + dataset titles derive from their
slugs (Kaggle remaps mismatches). The on-Kaggle driver
(`scripts/kaggle_train_driver.py`) runs generate → validate → finetune
→ build, then prunes the GPU compile cache and flattens artifacts so
downloads stay small. Every wait is bounded: per-call eval timeouts,
per-stage kernel timeouts, poller status/download timeouts, and 429
backoff that never burns the failure budget.

## Verification

Per-repo rules live in `REPO_RULES.md`. The gates:

```sh
python -m pytest tests/ -q
python -m compileall -q scripts tests
```

## Notes

- Local training only; the hosted platform path is out of scope.
- No `cactus generate-data`: all dataset content is authored in this
  repo (seeds plus deterministic expansion).
- `[train] backend` picks the JAX device (`cpu` hides GPUs, `gpu`/`auto`
  inherit the environment). The handoff task trains on CPU because the
  20-layer fp32 step OOMs on a 4 GB GTX 1650.
- On small/old machines, cap training so it can't take the box down
  (uncapped all-core XLA compile rebooted this host, and batch-16 compile
  ate 8.6 GB until the kernel OOM-killer fired). Pinned to 2 cores, low
  priority, 6 GB RAM ceiling so the cgroup killer fires before global
  OOM, detached:
  `setsid -f systemd-run --user --scope -p MemoryMax=6G taskset -c 0,1
  nice -n 10 .venv/bin/python scripts/finetune.py --task <task.toml>`.
- Locally tuned archives report `confidence=None` (the confidence head
  is untouched by local LoRA); route on eval accuracy instead.
