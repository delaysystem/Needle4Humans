"""First-match rule engine for pre-model routing.

General interpreter: rule SPECS are task data (JSON), this module holds no
task literals. Each rule names a regex plus the call list to emit when it
matches (possibly empty, meaning abstain). First match in file order wins;
no match returns None so the caller falls through to the model.

Safety direction: a false POSITIVE routes a chat to a human (graceful
degradation); a false ABSTAIN withholds help. Abstain rules must stay
tight; adversarial no-fire cases in the task's cases file prove it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def load_rules(path: str | Path) -> list[dict]:
    """Load and validate rule specs, dying loud on any defect."""
    raw = _common.read_json(path)
    rules = raw.get("rules", None) if isinstance(raw, dict) else None
    if not isinstance(rules, list) or not rules:
        _common.die(f"fastpath spec needs a non-empty 'rules' list: {path}")
    seen = set()
    for index, rule in enumerate(rules):
        where = f"{path} rule #{index}"
        if not isinstance(rule, dict):
            _common.die(f"{where} is not an object")
        ident = rule.get("id")
        if not ident or not isinstance(ident, str):
            _common.die(f"{where} needs a string 'id'")
        if ident in seen:
            _common.die(f"{where} duplicates rule id '{ident}'")
        seen.add(ident)
        pattern = rule.get("pattern")
        if not pattern or not isinstance(pattern, str):
            _common.die(f"{where} ('{ident}') needs a 'pattern' string")
        try:
            flags = 0 if rule.get("case") == "sensitive" else re.IGNORECASE
            rule["_compiled"] = re.compile(pattern, flags)
        except re.error as exc:
            _common.die(f"{where} ('{ident}') bad regex: {exc}")
        calls = rule.get("calls", None)
        if not isinstance(calls, list):
            _common.die(f"{where} ('{ident}') needs a 'calls' list")
        for call in calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments", {}), dict)
            ):
                _common.die(
                    f"{where} ('{ident}') calls need "
                    "{name, arguments} objects"
                )
    return rules


def apply_rules(
    rules: list[dict], query: str
) -> list | None:
    """First matching rule's call list, or None when nothing matches."""
    for rule in rules:
        if rule["_compiled"].search(query):
            return [
                {
                    "name": call["name"],
                    "arguments": dict(call.get("arguments", {})),
                }
                for call in rule["calls"]
            ]
    return None


def run_cases(rules: list[dict], cases: list[dict]) -> list[str]:
    """Check {query, calls|null} cases; returns human-readable failures."""
    failures = []
    for index, case in enumerate(cases):
        query = case.get("query", "")
        want = case.get("calls", None)
        got = apply_rules(rules, query)
        if want is None:
            if got is not None:
                failures.append(
                    f"case #{index} should fall through but fired: {query!r}"
                )
        elif got != want:
            failures.append(
                f"case #{index} want {json.dumps(want)} "
                f"got {json.dumps(got)}: {query!r}"
            )
    return failures
