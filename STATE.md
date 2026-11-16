# Migration state — Needle 3 human-handoff finetune

## Status addendum 2026-11-16 (moderation task + TPU path; handoff notes below stand)

Repo is now multi-task. New `tasks/moderation/` finetune (separate
adapter, owner decision): `flag_content` tool with `violation`
(toxicity, jailbreak, prompt_injection, tool_abuse, manipulation) and
`source` (user, llm) enums, EN-only v1, 158 train + 36 eval rows, all
seeds authored in-repo, validated 0 errors. Target 34/36 exact
(`tasks/moderation/rounds.json`). No PII content anywhere by owner
decision. Scripts generalized task-neutrally: `evaluate.py` prints
neutral `[]` labels plus a tool-name-only routing diagnostic;
`kaggle_push.py` renders TPU kernels (`[kaggle] tpu` + `tpu_shape`);
the driver resolves its pip stack from `[kaggle] train_package`,
supports `[kaggle] probe` hardware probes, and passes
`--expect-backend` so the loop dies on a backend mismatch instead of
mistraining. Moderation TPU attempt failed louder than Nov-12's
silent-CPU: probe v1 (2026-11-16) starved QUEUED 7+ hours (poll
timeout) while the handoff GPU kernel completed, so v1 trains on the
recorded GPU fallback (batch 32, T4 fit validated by that same
completed handoff run: 1358 steps, 2xT4, returncode 0). The queued
TPU probe v1 was left in place; if it ever runs it is bonus data only.

## Status addendum 2026-11-14 03:55 UTC (supersedes the notes below)

Training moved to Kaggle GPU (account `blonded2`). Round 5 finished on
kernel v17: 238/252 exact (target 250), loss converged flat — data-limited.
Round 6 (kernel v18, 553 seeds / 3257 rows targeting v17's 14 misses) is
RUNNING as of 03:51 UTC on the refreshed quota (30h GPU). When it
completes: poller downloads to `out/`, run `evaluate.py`, publish outputs
if improved. Progress tracker: `tasks/human_handoff/rounds.json`.

Repo renamed 2026-11-14: `delaysystem/cactus` → `delaysystem/Needle4Humans`
(GitHub keeps a redirect); local dir is now `Music/Needle4Humans`. The
worker repo is `Needle4Workers` (`Music/tuff/Needle4Workers`,
`delaysystem/Needle4Workers` on GitHub).

Written 2026-11-10 on the old box (i5-4570S, 15 GB RAM, GTX 1650 SUPER 4 GB)
for a move to a new PC. Code and dataset move; environments, caches, and
build artifacts do not.

## What this is

A reusable local-LoRA finetuning rig for Cactus Needle 3, plus a complete
test task: human-handoff detection in 7 languages (EN, DE, FR, ES, IT, NL,
PL) covering explicit asks, implicit frustration, sarcasm, multi-sentence
context, and edge cases (typos, one-word, ALL-CAPS). Every stage is one
config-driven script; a new finetune needs a new task directory, not new
code. Details in `README.md`; repo rules in `REPO_RULES.md`.

## What to copy

Copy (source + data):

- `scripts/` — `_common.py`, `check_env.py`, `generate_data.py`,
  `validate_data.py`, `finetune.py`, `build.py`, `evaluate.py`
- `tasks/human_handoff/` — `task.toml`, `tools.json`, `slot_banks.json`,
  `seeds_train.jsonl` (161 seeds), `seeds_eval.jsonl` (252 seeds)
- `tasks/human_handoff/out/train.jsonl`, `eval.jsonl`, `manifest.json`
  (optional: byte-reproducible via `generate_data.py`, same seed)
- `tests/` — `conftest.py`, `test_common.py`, `test_generate_data.py`,
  `test_validate_data.py`, `test_evaluate.py`, `test_cli_argv.py`
- `README.md`, `REPO_RULES.md`, `agents2.md`, this file

Do NOT copy (machine-local; the new box rebuilds them):

- `.venv/` (Python 3.14 via uv on the old box; torch-free,
  `cactus-compute[train,gpu]` + `pytest`)
- `tasks/human_handoff/out/train.log`, `out/jax_cache/`
- `checkpoints/` (trainer default dir), `~/.cache/cactus/`
  (engine + base weights, re-downloaded automatically)
- No adapter or `.cact` was ever produced (see below), so there is
  nothing to carry over even optionally.

Note: the old repo has no commits yet (`git status` shows everything
untracked). Commit on the new box or keep copying files.

## Status: where each stage stands

| Stage | State |
|---|---|
| Env + install | Done on old box; redo on new box (§Resume) |
| `check_env.py` | Done, green on old box |
| Tool definition | Done (`request_human_handoff`, reason enum + details) |
| Dataset | DONE, validated: 1281 train + 252 eval rows, 0 errors |
| Unit tests | DONE: 37 passing |
| Training | DONE on Kaggle T4 (94 min, val 0.82→0.0104, held-out 123/128) |
| `build.py` (20-layer `.cact`) | DONE on Kaggle: `handoff_20l.cact` (63 MB) |
| `evaluate.py` (base vs tuned) | DONE locally: base 15.5% → tuned 69.0% exact, 85.3% routing |

Dataset composition per language: explicit 60 (incl. 36 with `reason`),
implicit 27, sarcastic 27, multisentence 27 (incl. 18 with `details`),
edge 20, negative 22 (~12% `[]` refusals). Eval is 6 per
language×category cell (252 total), phrased distinctly from train, zero
query overlap (validator-enforced).

## TPU verdict (2026-11-12): not provisionable via API

Four probes (script+string shape, script+bool, correct Tpu1VmV38 shape,
notebook kernel) all landed on CPU despite enable_tpu and a 20h TPU
quota balance. Code audit says the model is TPU-clean (single backend
check falls back for fp32; no CUDA-only ops), so this is account or
capacity gating with silent CPU fallback, not a code problem. Options:
check the website notebook UI for a TPU accelerator choice, or stay on
GPU (5.25h left, refreshes Nov 14). No TPU code was merged.

## Outcome on Kaggle (2026-11-11/12, account blonded2)

- Input dataset `blonded2/needle3-handoff-input` (private, v1), kernel
  `blonded2/needle3-handoff-train` (script, T4 ×2, internet on).
- v1 failed in seconds: datasets mount at
  `/kaggle/input/datasets/<owner>/<slug>/`, driver looked one level up
  (fixed + test-covered).
- v2 completed: pip 49s, generate + validate instant, finetune 94 min
  (1460 steps, batch 16), build 26s. Loss 2.72→0.01, val 0.82→0.0104,
  no overfitting; 123/128 held-out exact on W4 weights.
- Binaries dataset `blonded2/needle3-handoff-outputs` (private, v1):
  adapter, 20-layer `.cact`, eval report, train log, data, manifest.
- Local eval (runtime-only package): base 39/252 (15.5%), tuned 174/252
  (69.0%) exact-call, 215/252 (85.3%) routing-correct, 0 timeouts.
  Multisentence exact is 9/42 only because the model adds a `details`
  arg the gold `{}` omits (31/33 misses are that pattern).
- Eval hardening found mid-run: `reset()` between independent queries
  (else context accumulates and calls slow/hang), flushed progress,
  120 s per-call timeout recorded in the report.

## Round 2 (2026-11-12): miss-driven improvements

- Diagnosed v1 misses: question-form asks, transfer verbs, hello-checks,
  repeat-page/wasting-time implicits, greeting + relative negatives all
  missing from training; multi golds wrongly `{}` (model rightly added
  `details`); details always on sentence 1 (positional bias).
- Corrected 42 multi eval golds (details spans; e03 + reason=billing);
  v1 rescored 75.0% exact on corrected golds (multi 57.1%).
- Added 91 seeds (all 7 langs), 1862 train rows, rank 16→32, input
  dataset v2, kernel v3 (129 min). Val 0.81→0.016, held-out 168/186.
- v2 eval: 217/252 (86.1%) exact, 242/252 (96.0%) routing, 0 timeouts.
  Per category: edge 100, explicit 92.9, implicit 97.6, multi 52.4
  exact (routing 100 — misses are span/reason choice), negative 100,
  sarcastic 73.8 (routing 85.7; -3 cases vs v1, precision/recall trade
  against negatives going 76→100).
- Outputs dataset v2 published with v2 adapter (rank 32), `.cact`,
  report, logs, data.
- Hang-proofing pass: poller status/download/log timeouts, 429 backoff
  outside the failure budget, driver per-stage timeout (3h) from
  `[kaggle] stage_timeout` recorded in STATUS.json. 73 tests pass.

## Training attempt history (evidence, not lore)

1. **GPU, batch 16** → `RESOURCE_EXHAUSTED ... CUDA_ERROR_OUT_OF_MEMORY`
   at `jit_train_step` load. Reproduced on a 32-example subset at batch 4.
   Memory-fraction 0.5 and platform-allocator probes also failed. JAX
   itself on the GPU is healthy (test matmul passed). Verdict: the
   20-layer fp32 step does not fit ~3.3 GB free VRAM. → `backend = "cpu"`.
2. **CPU, batch 16** → grew past 8.6 GB RSS; kernel global-OOM killed it.
   An earlier uncapped all-core run rebooted the box outright.
3. **CPU, batch 8, 6 GB cgroup** → exceeded the 6 GB ceiling;
   cgroup-scoped OOM kill, box untouched. Safety caps proven working.
4. **CPU, batch 4, 6 GB cgroup, 2 cores, nice** → alive ~50 CPU-min,
   RSS flat at 5.88 GB, temps 62–67°C — then the box rebooted AGAIN
   (~21:24) despite the caps. Nothing was produced; `train.log` stayed
   empty (see buffering note). Takeaway: this old box is unstable under
   any sustained load (age/PSU/RAM, not our settings) — moving is right.

Supporting measurements: all 1281 rows are 294–449 tokens with the real
tokenizer (tool schema ≈ 280 tokens), so `--max-len 256` would truncate
everything — ruled out. Weights are already 4-bit in training
(`CQ W4 STE + A8`); only compute is fp32, hardcoded by Cactus, no flag.
GTX 1650 (sm_75) has no fp8 hardware anyway.

Two script fixes came out of this, both test-covered: `cactus` binary
resolution without an activated venv, and truly streaming `run_tee`
(readline loop + per-line flush; the old version hid all output until
8 KB or process exit — which is why blind runs showed empty logs).

## Resume procedure on the new PC

```sh
uv venv
. .venv/bin/activate
uv pip install 'cactus-compute[train,gpu]' pytest   # NVIDIA; else [train]
export TASK=tasks/human_handoff/task.toml
python scripts/check_env.py
python -m pytest tests/ -q
python scripts/generate_data.py --task $TASK
python scripts/validate_data.py --task $TASK
```

Then, sized to the new hardware:

- **Capable GPU (8 GB+ VRAM recommended):** set `backend = "gpu"` (or
  delete the key for `auto`) and `batch_size = 16` in `task.toml`, then
  `python scripts/finetune.py --task $TASK` in the foreground or under
  `setsid`/tmux. First run downloads the base checkpoint + tokenizer.
- **Weak box again:** keep `backend = "cpu"`, `batch_size = 4`, and use
  the capped launch from `README.md` (2 cores, low priority, 6 GB
  ceiling). Expect many hours for 5780 steps.

Then:

```sh
python scripts/build.py --task $TASK        # 20-layer .cact
python scripts/evaluate.py --task $TASK     # base vs tuned, JSON report
```

Facts that shape the run: the trainer saves the adapter ONCE at the end
(no checkpoints, no resume — a crash restarts from step 0, same seed so
deterministic); `jax_cache_dir` in `task.toml` persists XLA executables
so a restarted run skips recompilation; loss starts near ~1.0 and only
the trend matters; locally tuned archives report `confidence=None`, so
route on eval accuracy. Monitors used here were session-scoped and are
dead; re-arm equivalents if wanted.

## Open items / known limitations

- `batch_size = 4` in `task.toml` is a weak-box survival setting, not a
  quality choice — raise to 16 on capable hardware.
- Eval answers use `{}` (routing-only); `reason`/`details` grounding is
  trained (252 reason + 126 details examples) but not separately scored.
- Seeds are parallel translations across languages by design (same
  template structure, native phrasing) — good for parity, weaker for
  per-language idiom coverage.
