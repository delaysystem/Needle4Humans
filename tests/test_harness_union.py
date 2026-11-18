"""Harness gates: rule precision on frozen eval + 16L union target.

Uses frozen artifacts (eval.jsonl, eval_report_16l.json) with the live
rule spec, so rule edits that lose coverage or precision fail loudly.
"""

import json
from pathlib import Path

import fastpath
from evaluate import calls_match

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "human_handoff"


def _load():
    rules = fastpath.load_rules(TASK_DIR / "fastpath.json")
    evals = {}
    for line in (TASK_DIR / "out" / "eval.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            evals[row["id"]] = row
    report = json.loads((TASK_DIR / "out" / "eval_report_16l.json").read_text(encoding="utf-8"))
    return rules, evals, report["cases"]


def test_eval_rule_fires_are_all_correct():
    rules, evals, _ = _load()
    wrong = [
        row_id
        for row_id, row in evals.items()
        if (fired := fastpath.apply_rules(rules, row["query"])) is not None
        and not calls_match(row["answers"], fired)
    ]
    assert wrong == [], f"rules misfire on frozen eval: {wrong}"


def test_16l_engine_plus_rules_union_reaches_240():
    rules, evals, cases = _load()
    assert len(cases) == 252
    union = 0
    for case in cases:
        row = evals[case["id"]]
        fired = fastpath.apply_rules(rules, row["query"])
        if fired is not None:
            union += calls_match(row["answers"], fired)
        else:
            union += case["tuned_hit"]
    assert union >= 240, f"16L+rules union is {union}/252, need >= 240"
