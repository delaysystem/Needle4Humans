# Needle 3 finetuning

Reusable scripts to finetune Cactus Needle 3 models, plus two complete
tasks: human-handoff detection in 7 languages (EN, DE, FR, ES, IT, NL,
PL) with sarcasm, implicit requests, multi-sentence context, and edge
cases; and moderation (toxicity plus AI-exploit detection on both user
messages and assistant output, EN-only v1). Each task trains its own
separate adapter.

How it works: every stage is one script, and every task-specific value
lives in the task's `task.toml`. A new finetune needs a new task
directory, not new code.

The tuned model ships in the companion repo
([Needle4Workers](https://github.com/delaysystem/Needle4Workers)): a
production Cloudflare Worker with guardrails, metrics, and a
release-gating eval harness.

## Results (human handoff)

Frozen 252-case eval, exact name+args+order, 7 languages × 6 categories:

| Round | Train rows | Exact | Routing |
|---|---|---|---|
| v1 (base tuned) | 1281 | 69.0% (174) | 85.3% |
| v2 | 1862 | 86.1% (217) | 96.0% |
| v3 | 2282 | 91.3% (230) | 98.8% |
| v4 | 2814 | 94.8% (239) | 98.8% |
| v5 | 3124 | 94.4% (238) | 97.6% |

Untuned base: 15.5% (39). v5 per-category: edge 100%, implicit 100%,
sarcastic 97.6%, explicit 95.2%, negative 90.5%, multisentence 83.3%.
Per-language parity: worst 33/36 (de/fr/it), best 36/36 (es).
Target 250/252 (99.2%) is set in `tasks/human_handoff/rounds.json`,
which also logs every round; round 6 (3257 rows, miss-targeted) is next.

## Results (moderation)

Frozen 36-case eval, exact name+args+order, EN × 6 categories
(benign, toxic_user, toxic_llm, exploit_user, exploit_llm, edge):

| Round | Train rows | Exact | Routing |
|---|---|---|---|
| v1 | 158 | pending TPU probe | pending |

Target 34/36 (94.4%) is set in `tasks/moderation/rounds.json`.
Training runs on Kaggle GPU: TPU was tried first (probe v1 starved
QUEUED 7+ hours on 2026-11-16 while GPU kernels completed), so v1
follows the GPU fallback with the same driver and loop the handoff
task proved over 11 rounds.

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
  set (exact-calls-in-order, plus a tool-name-only routing diagnostic).
- `scripts/kaggle_upload_dataset.py`, `kaggle_push.py`, `kaggle_poll.py`,
  `kaggle_publish_outputs.py` (+ `kaggle_train_driver.py`, which runs on
  Kaggle) — versioned datasets in/out, GPU or TPU kernel runs, status
  polling. TPU tasks can push a hardware probe (`[kaggle] probe=true`)
  before burning quota on training.
- `tasks/human_handoff/` — the handoff task: `task.toml`, `tools.json`,
  `slot_banks.json`, `seeds_train.jsonl`, `seeds_eval.jsonl`.
- `tasks/moderation/` — the moderation task: same layout, `flag_content`
  tool (violation + source enums), EN-only v1.
- `tests/` — unit tests mirroring the scripts.

## Setup

```sh
uv venv --python 3.11
. .venv/bin/activate
uv pip install -r requirements.txt
```

## Run (either task)

```sh
. .venv/bin/activate
export TASK=tasks/human_handoff/task.toml  # or tasks/moderation/task.toml
python scripts/check_env.py
python scripts/generate_data.py --task $TASK
python scripts/validate_data.py --task $TASK
python scripts/finetune.py --task $TASK
python scripts/build.py --task $TASK
python scripts/evaluate.py --task $TASK
```

`finetune.py` and `build.py` accept `--dry-run` to print the `cactus`
argv without executing. `evaluate.py --limit N` smoke-tests the first
N eval cases. No script overwrites existing outputs: `generate_data.py`,
`finetune.py`, `build.py`, `evaluate.py`, and `kaggle_poll.py` all
refuse when their targets exist and need an explicit `--overwrite` to
replace them (poll downloads stage first, then move into place).

## Run on Kaggle (GPU or TPU)

Compile + training run in a private GPU or TPU kernel; datasets carry
the inputs and the binaries. Needs the CLI (`uv tool install kaggle`)
and a token at `~/.kaggle/access_token`:

```sh
export TASK=tasks/human_handoff/task.toml  # or tasks/moderation/task.toml
python scripts/kaggle_upload_dataset.py --task $TASK   # version inputs
python scripts/kaggle_push.py --task $TASK             # push kernel run
python scripts/kaggle_poll.py --task $TASK             # wait + download
python scripts/kaggle_publish_outputs.py --task $TASK  # version binaries
python scripts/evaluate.py --task $TASK                # local, runtime-only
```

All four Kaggle scripts accept `--dry-run`; `kaggle_poll.py --once`
prints a single status line. Kernel + dataset titles derive from their
slugs (Kaggle remaps mismatches). The on-Kaggle driver
(`scripts/kaggle_train_driver.py`) runs generate → validate → finetune
→ build, then prunes the compile cache and flattens artifacts so
downloads stay small. Every wait is bounded: per-call eval timeouts,
per-stage kernel timeouts, poller status/download timeouts, and 429
backoff that never burns the failure budget.

TPU tasks (`tpu=true` in `[kaggle]`) push a hardware probe first:
with `[kaggle] probe=true` the driver reports JAX backend/devices and
exits without installing or training. Past TPU requests silently
landed on CPU (see `STATE.md`), so flip `probe` to false only after
the probe log shows TPU devices. The training loop also dies on a
backend mismatch (`--expect-backend`, wired from the same flags), so a
misplaced kernel can never burn a full run quietly.

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
- Forks: `[kaggle] username` in each `task.toml` is owner-specific
  (`blonded2` here) — point it at your own account plus your
  `~/.kaggle/` credentials before pushing.
- Each task's `out/` deliberately tracks `train.jsonl`, `eval.jsonl`,
  `manifest.json`, and the kernel log (byte-reproducible inputs and
  run records); binaries (`.cact`, `.safetensors`, caches, staging
  dirs) stay ignored.

## License

Business Source License 1.0, see [LICENSE](LICENSE). Converts to
Apache-2.0 on 2030-01-01.
