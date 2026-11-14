# Security Policy

## Reporting

Report vulnerabilities privately to DelaySystem@tubot.lat with a
description and, if possible, a reproduction. Do not open a public
issue for unpatched security bugs.

## Scope notes

- This repo contains no production services; the sensitive surface is
  the Kaggle pipeline (`scripts/kaggle_*.py`), which shells out to the
  `kaggle` CLI using your own `~/.kaggle/` credentials. Never commit
  credentials, tokens, or browser session material.
- Model binaries (`.cact`, `.safetensors`) are build artifacts and stay
  out of git; only small JSONL/JSON run records are tracked.
