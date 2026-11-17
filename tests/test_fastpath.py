"""Tests for scripts/fastpath.py (general rule engine)."""

import pytest

import fastpath


def write_spec(tmp_path, rules):
    import json

    path = tmp_path / "fastpath.json"
    path.write_text(json.dumps({"rules": rules}), encoding="utf-8")
    return path


def test_first_match_wins(tmp_path):
    rules = fastpath.load_rules(
        write_spec(
            tmp_path,
            [
                {"id": "a", "pattern": "hi", "calls": [{"name": "t", "arguments": {}}]},
                {"id": "b", "pattern": "hi", "calls": []},
            ],
        )
    )
    assert fastpath.apply_rules(rules, "hi there") == [
        {"name": "t", "arguments": {}}
    ]


def test_no_match_returns_none(tmp_path):
    rules = fastpath.load_rules(
        write_spec(
            tmp_path,
            [{"id": "a", "pattern": "zzz", "calls": []}],
        )
    )
    assert fastpath.apply_rules(rules, "hi there") is None


def test_default_case_insensitive_explicit_sensitive(tmp_path):
    rules = fastpath.load_rules(
        write_spec(
            tmp_path,
            [
                {"id": "i", "pattern": "stop", "calls": []},
                {"id": "s", "pattern": "Go", "calls": [], "case": "sensitive"},
            ],
        )
    )
    assert fastpath.apply_rules(rules, "STOP!") == []
    assert fastpath.apply_rules(rules, "go now") is None
    assert fastpath.apply_rules(rules, "Go now") == []


def test_calls_copied_not_aliased(tmp_path):
    rules = fastpath.load_rules(
        write_spec(
            tmp_path,
            [
                {
                    "id": "a",
                    "pattern": "x",
                    "calls": [{"name": "t", "arguments": {"r": 1}}],
                }
            ],
        )
    )
    first = fastpath.apply_rules(rules, "x")
    first[0]["arguments"]["r"] = 999
    second = fastpath.apply_rules(rules, "x")
    assert second == [{"name": "t", "arguments": {"r": 1}}]


def test_bad_specs_die_loud(tmp_path):
    import json

    bad_specs = [
        {"rules": []},
        {"rules": [{"pattern": "x", "calls": []}]},
        {"rules": [{"id": "a", "calls": []}]},
        {"rules": [{"id": "a", "pattern": "(", "calls": []}]},
        {"rules": [{"id": "a", "pattern": "x"}]},
        {
            "rules": [
                {"id": "a", "pattern": "x", "calls": []},
                {"id": "a", "pattern": "y", "calls": []},
            ]
        },
        {
            "rules": [
                {
                    "id": "a",
                    "pattern": "x",
                    "calls": [{"name": 5, "arguments": {}}],
                }
            ]
        },
    ]
    for index, spec in enumerate(bad_specs):
        path = tmp_path / f"bad{index}.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        with pytest.raises(SystemExit):
            fastpath.load_rules(path)


def test_run_cases_reports_failures_only(tmp_path):
    rules = fastpath.load_rules(
        write_spec(
            tmp_path,
            [{"id": "a", "pattern": "hi", "calls": [{"name": "t", "arguments": {}}]}],
        )
    )
    assert fastpath.run_cases(
        rules,
        [
            {"query": "hi", "calls": [{"name": "t", "arguments": {}}]},
            {"query": "bye", "calls": None},
        ],
    ) == []
    failures = fastpath.run_cases(
        rules,
        [
            {"query": "bye", "calls": [{"name": "t", "arguments": {}}]},
            {"query": "hi", "calls": None},
        ],
    )
    assert len(failures) == 2
