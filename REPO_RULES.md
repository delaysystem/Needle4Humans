# Per-repo rules: cactus Needle 3 finetuning

Companion to `agents2.md`. This repo is greenfield (no upstream, no CI); the
commands below are defined here unless a source is named.

## Environment

- In-project env only: `.venv/` at the repo root (see `README.md`).
- Runtime + training install, from the
  [Fine-tuning Needle guide](https://cactuscompute.com/blog/finetuning-needle):
  `pip install "cactus-compute[train,gpu]"` on this NVIDIA machine,
  `"cactus-compute[train]"` elsewhere.
- Test runner: `pip install pytest` (defined here; stdlib-only otherwise).

## Verification definition (agents2.md Rule 3)

| Check | Command | Source |
|---|---|---|
| Unit tests | `python -m pytest tests/ -q` from repo root | defined here |
| Syntax | `python -m compileall -q scripts tests` | defined here |
| Literal scan (Rule 1) | `grep -rnE '(/home/\|/tmp/\|http://\|https://\|:[0-9]{2,5}\b)' scripts/ \| grep -v '^.*#'` reviewed by reading the diff | agents2.md Rule 1 |
| End-to-end | `scripts/check_env.py && scripts/generate_data.py && scripts/validate_data.py && scripts/finetune.py && scripts/build.py && scripts/evaluate.py` with the task config | the plan |

- Affected tests: `tests/test_<script>.py` mirrors `scripts/<script>.py`; the
  suite is small, so every change runs the full `tests/` directory.
- CI: none (no remotes, no CI config as of 2026-11-10); local runs are the gate.
- Mocks/stubs/fakes: not used. Pure logic (validation, expansion, scoring) is
  tested directly; CLI-wrapping scripts are tested on argument building, and
  the real `cactus` CLI plus a real train/build/eval run is the proof.
- Cannot run here: the hosted platform path is out of scope (local only, no
  `CACTUS_API_KEY`); `cactus generate-data` is out of scope per owner decision.
- Small changes: affected `tests/test_<script>.py` plus the syntax check; the
  full e2e run is required when data format, CLI flags, or eval scoring change.

## Submission

No upstream: work stays on the local branch, uncommitted until the owner asks
for a commit. No pushes (no remotes configured).
