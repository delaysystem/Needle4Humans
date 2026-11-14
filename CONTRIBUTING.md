# Contributing

## Setup

```sh
uv venv --python 3.11
. .venv/bin/activate
uv pip install -r requirements.txt
```

## Gates (must pass before any PR)

```sh
python -m pytest tests/ -q
python -m compileall -q scripts tests
```

Rules live in `REPO_RULES.md` and `agents2.md`. The short version:

- Task values live in task config, never in code (no task literals in `scripts/`).
- Tests mirror scripts: `tests/test_<script>.py` covers `scripts/<script>.py`.
- No mocks: pure logic is tested directly; CLIs are tested on argv
  building; the real `cactus` CLI plus real runs are the proof.
- Datasets are authored in-repo (seeds + deterministic expansion).
  Never train on the frozen eval set; new train queries must not
  duplicate eval queries verbatim.

## License

Contributions are under the repo's Business Source License 1.0
(see LICENSE).
