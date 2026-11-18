"""Tests for scripts/evaluate.py scoring."""

import pytest

import evaluate


def test_calls_match_exact():
    expected = [{"name": "t", "arguments": {"a": 1}}]
    assert evaluate.calls_match(expected, [dict(expected[0])])


def test_calls_match_empty():
    assert evaluate.calls_match([], [])
    assert not evaluate.calls_match([], [{"name": "t", "arguments": {}}])


def test_calls_match_order_matters():
    expected = [
        {"name": "a", "arguments": {}},
        {"name": "b", "arguments": {}},
    ]
    actual = [
        {"name": "b", "arguments": {}},
        {"name": "a", "arguments": {}},
    ]
    assert not evaluate.calls_match(expected, actual)


def test_calls_match_arguments_must_agree():
    expected = [{"name": "t", "arguments": {"reason": "billing"}}]
    assert not evaluate.calls_match(
        expected, [{"name": "t", "arguments": {"reason": "technical"}}]
    )
    assert not evaluate.calls_match(expected, [{"name": "t", "arguments": {}}])


def test_calls_match_missing_arguments_is_empty():
    assert evaluate.calls_match(
        [{"name": "t", "arguments": {}}], [{"name": "t"}]
    )


def test_routing_match_ignores_arguments():
    expected = [{"name": "t", "arguments": {"a": 1}}]
    assert evaluate.routing_match(expected, [{"name": "t"}])
    assert evaluate.routing_match(
        expected, [{"name": "t", "arguments": {"a": 2}}]
    )
    assert not evaluate.routing_match(
        expected, [{"name": "other", "arguments": {"a": 1}}]
    )


def test_routing_match_order_matters_and_empty():
    assert evaluate.routing_match([], [])
    assert not evaluate.routing_match([], [{"name": "t"}])
    expected = [{"name": "a"}, {"name": "b"}]
    assert evaluate.routing_match(
        expected, [{"name": "a"}, {"name": "b", "arguments": {}}]
    )
    assert not evaluate.routing_match(
        expected, [{"name": "b"}, {"name": "a"}]
    )


def test_summarize_buckets():
    results = [
        {"id": "a", "hit": True, "expected": [{"name": "t"}]},
        {"id": "b", "hit": False, "expected": [{"name": "t"}]},
        {"id": "c", "hit": True, "expected": []},
    ]
    manifest = {
        "a": {"lang": "en", "category": "explicit"},
        "b": {"lang": "de", "category": "explicit"},
        "c": {"lang": "de", "category": "negative"},
    }
    summary = evaluate.summarize(results, manifest)
    assert (summary["hits"], summary["total"]) == (2, 3)
    assert summary["by_language"]["en"] == {"total": 1, "hits": 1}
    assert summary["by_language"]["de"] == {"total": 2, "hits": 1}
    assert summary["by_category"]["explicit"] == {"total": 2, "hits": 1}
    assert (summary["negative_hits"], summary["negatives"]) == (1, 1)


def test_strip_required_keys():
    tools = [
        {
            "name": "handoff",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        }
    ]
    stripped = evaluate.strip_required_keys(tools)
    assert stripped[0]["parameters"] == {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    }
    # Original untouched (deep copy).
    assert tools[0]["parameters"]["required"] == ["reason"]


def test_rate_guards_empty():
    assert evaluate.rate(0, 0) == 0.0
    assert evaluate.rate(1, 2) == 0.5


def test_complete_with_timeout_fast():
    import time
    from concurrent.futures import ThreadPoolExecutor

    class Agent:
        def complete(self, query):
            return {"function_calls": [{"name": "t"}]}

    with ThreadPoolExecutor(max_workers=1) as pool:
        response, timed_out = evaluate.complete_with_timeout(
            pool, Agent(), "hi"
        )
    assert timed_out is False
    assert response == {"function_calls": [{"name": "t"}]}


def test_complete_with_timeout_slow(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    class Agent:
        def complete(self, query):
            time.sleep(1)
            return {"function_calls": []}

    monkeypatch.setattr(evaluate, "CALL_TIMEOUT_S", 0.05)
    with ThreadPoolExecutor(max_workers=1) as pool:
        calls, timed_out = evaluate.complete_with_timeout(
            pool, Agent(), "hi"
        )
    assert timed_out is True
    assert calls == []


def test_main_refuses_overwrite_without_flag(tmp_path):
    import pytest

    task_dir = tmp_path / "task"
    (task_dir / "out").mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "[task]\nname = \"mini\"\nout_dir = \"out\"\n"
        "[eval]\nreport_file = \"report.json\"\n",
        encoding="utf-8",
    )
    report = task_dir / "out" / "report.json"
    report.write_text("sentinel", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluate.main(["--task", str(task_dir / "task.toml")])
    assert report.read_text(encoding="utf-8") == "sentinel"


def test_run_agent_resets_and_reports_progress(monkeypatch, capsys):
    import cactus

    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.resets = 0
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def complete(self, query):
            return {"function_calls": [{"name": "t"}]}

        def reset(self):
            self.resets += 1

    monkeypatch.setattr(cactus, "Cactus", FakeAgent)
    results = evaluate.run_agent([], None, ["a", "b"], "base")
    assert results == [
        ([{"name": "t"}], False),
        ([{"name": "t"}], False),
    ]
    assert created[0].resets == 2
    assert "[base] 2/2" in capsys.readouterr().out


def test_majority_vote_picks_most_common():
    a = [{"name": "t", "arguments": {"reason": "billing"}}]
    b = [{"name": "t", "arguments": {"reason": "general"}}]
    assert evaluate.majority_vote([a, b, a]) == a
    assert evaluate.majority_vote([b, a, a]) == a


def test_majority_vote_ties_go_first():
    a = [{"name": "t", "arguments": {}}]
    b = []
    assert evaluate.majority_vote([a, b]) == a
    assert evaluate.majority_vote([b, a]) == b


def test_majority_vote_ignores_key_order():
    first = [{"arguments": {"reason": "x"}, "name": "t"}]
    second = [{"name": "t", "arguments": {"reason": "x"}}]
    assert evaluate.majority_vote([first, second, []]) == first


def test_majority_vote_empty():
    assert evaluate.majority_vote([]) == []


def test_load_prime_absent_key_returns_none(tmp_path):
    assert evaluate.load_prime({"eval": {}}, tmp_path) is None
    assert evaluate.load_prime({}, tmp_path) is None


def test_load_prime_reads_file(tmp_path):
    (tmp_path / "prime.txt").write_text(" route! ", encoding="utf-8")
    config = {"eval": {"prime_file": "prime.txt"}}
    assert evaluate.load_prime(config, tmp_path) == "route!"


def test_load_prime_missing_file_dies(tmp_path):
    import pytest

    config = {"eval": {"prime_file": "nope.txt"}}
    with pytest.raises(SystemExit):
        evaluate.load_prime(config, tmp_path)


def test_run_agent_votes_take_majority(monkeypatch):
    import cactus

    answers = [
        {"function_calls": [{"name": "t", "arguments": {"r": 1}}]},
        {"function_calls": []},
        {"function_calls": [{"name": "t", "arguments": {"r": 1}}]},
    ]

    class FakeAgent:
        def __init__(self, **kwargs):
            self.resets = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def complete(self, query):
            return answers.pop(0)

        def reset(self):
            self.resets += 1

    created = []
    orig_init = FakeAgent.__init__

    def tracking_init(self, **kwargs):
        orig_init(self, **kwargs)
        created.append(self)

    monkeypatch.setattr(FakeAgent, "__init__", tracking_init)
    monkeypatch.setattr(cactus, "Cactus", FakeAgent)
    results = evaluate.run_agent([], None, ["q"], "base", votes=3)
    assert results == [
        ([{"name": "t", "arguments": {"r": 1}}], False),
    ]
    assert created[0].resets == 3


def test_run_agent_passes_system(monkeypatch):
    import cactus

    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def complete(self, query):
            return {"function_calls": []}

        def reset(self):
            pass

    monkeypatch.setattr(cactus, "Cactus", FakeAgent)
    evaluate.run_agent([], None, ["q"], "base", system="prime!")
    assert seen.get("system") == "prime!"
    seen.clear()
    evaluate.run_agent([], None, ["q"], "base")
    assert "system" not in seen


# ---------------------------------------------------------------------------
# Hard coverage: calls_match matrix
# ---------------------------------------------------------------------------
def test_calls_match_nested_arguments():
    good = [{"name": "t", "arguments": {"a": {"b": [1, {"c": 2}]}}}]
    assert evaluate.calls_match(good, [dict(good[0])])
    bad = [{"name": "t", "arguments": {"a": {"b": [1, {"c": 3}]}}}]
    assert not evaluate.calls_match(good, bad)


def test_calls_match_argument_key_order_irrelevant():
    expected = [{"name": "t", "arguments": {"a": 1, "b": 2}}]
    actual = [{"name": "t", "arguments": {"b": 2, "a": 1}}]
    assert evaluate.calls_match(expected, actual)


def test_calls_match_extra_actual_key_fails():
    expected = [{"name": "t", "arguments": {"a": 1}}]
    actual = [{"name": "t", "arguments": {"a": 1, "b": 2}}]
    assert not evaluate.calls_match(expected, actual)


def test_calls_match_missing_actual_key_fails():
    expected = [{"name": "t", "arguments": {"a": 1, "b": 2}}]
    actual = [{"name": "t", "arguments": {"a": 1}}]
    assert not evaluate.calls_match(expected, actual)


def test_calls_match_name_case_sensitive():
    assert not evaluate.calls_match([{"name": "t"}], [{"name": "T"}])


def test_calls_match_none_arguments_is_empty():
    assert evaluate.calls_match(
        [{"name": "t", "arguments": None}], [{"name": "t", "arguments": {}}]
    )
    assert not evaluate.calls_match(
        [{"name": "t", "arguments": None}],
        [{"name": "t", "arguments": {"a": 1}}],
    )


def test_calls_match_int_vs_string():
    assert not evaluate.calls_match(
        [{"name": "t", "arguments": {"a": 1}}],
        [{"name": "t", "arguments": {"a": "1"}}],
    )


def test_calls_match_list_arguments_order_matters():
    expected = [{"name": "t", "arguments": {"items": [1, 2]}}]
    assert evaluate.calls_match(
        expected, [{"name": "t", "arguments": {"items": [1, 2]}}]
    )
    assert not evaluate.calls_match(
        expected, [{"name": "t", "arguments": {"items": [2, 1]}}]
    )


# ---------------------------------------------------------------------------
# Hard coverage: routing_match matrix
# ---------------------------------------------------------------------------
def test_routing_match_missing_names_equal_none():
    assert evaluate.routing_match([{}], [{}])
    assert not evaluate.routing_match([{}], [{"name": "t"}])


def test_routing_match_duplicate_names():
    expected = [{"name": "a"}, {"name": "a"}]
    assert evaluate.routing_match(expected, [{"name": "a"}, {"name": "a"}])
    assert not evaluate.routing_match(expected, [{"name": "a"}])


def test_routing_match_length_mismatch():
    assert not evaluate.routing_match(
        [{"name": "a"}], [{"name": "a"}, {"name": "a"}]
    )


def test_routing_match_unicode_names():
    assert evaluate.routing_match(
        [{"name": "tâche"}], [{"name": "tâche"}]
    )
    assert not evaluate.routing_match(
        [{"name": "tâche"}], [{"name": "tache"}]
    )


def test_routing_match_middle_swap_fails():
    expected = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    actual = [{"name": "a"}, {"name": "c"}, {"name": "b"}]
    assert not evaluate.routing_match(expected, actual)


def test_routing_match_deeply_nested_arguments_ignored():
    expected = [{"name": "t", "arguments": {"a": {"b": [1]}}}]
    actual = [{"name": "t", "arguments": {"z": {"y": [2]}}}]
    assert evaluate.routing_match(expected, actual)


# ---------------------------------------------------------------------------
# Hard coverage: summarize
# ---------------------------------------------------------------------------
def test_summarize_empty_results():
    summary = evaluate.summarize([], {})
    assert (summary["hits"], summary["total"]) == (0, 0)
    assert summary["by_language"] == {}
    assert (summary["negative_hits"], summary["negatives"]) == (0, 0)


def test_summarize_all_miss():
    results = [
        {"id": "a", "hit": False, "expected": [{"name": "t"}]},
        {"id": "b", "hit": False, "expected": []},
    ]
    manifest = {
        "a": {"lang": "en", "category": "explicit"},
        "b": {"lang": "en", "category": "negative"},
    }
    summary = evaluate.summarize(results, manifest)
    assert (summary["hits"], summary["total"]) == (0, 2)
    assert (summary["negative_hits"], summary["negatives"]) == (0, 1)


def test_summarize_negatives_keyed_on_expected_only():
    results = [
        {"id": "a", "hit": False, "expected": [{"name": "t"}]},
        {"id": "b", "hit": True, "expected": []},
    ]
    manifest = {
        "a": {"lang": "en", "category": "explicit"},
        "b": {"lang": "en", "category": "negative"},
    }
    summary = evaluate.summarize(results, manifest)
    assert (summary["negative_hits"], summary["negatives"]) == (1, 1)


def test_summarize_mixed_buckets():
    results = [
        {"id": "a", "hit": True, "expected": []},
        {"id": "b", "hit": True, "expected": []},
        {"id": "c", "hit": False, "expected": []},
    ]
    manifest = {
        "a": {"lang": "en", "category": "x"},
        "b": {"lang": "en", "category": "y"},
        "c": {"lang": "de", "category": "x"},
    }
    summary = evaluate.summarize(results, manifest)
    assert summary["by_language"]["en"] == {"total": 2, "hits": 2}
    assert summary["by_category"]["x"] == {"total": 2, "hits": 1}


# ---------------------------------------------------------------------------
# Hard coverage: strip_required_keys
# ---------------------------------------------------------------------------
def test_strip_required_keys_deep_nesting():
    node = {"a": {"b": {"required": [1], "keep": 2}}}
    assert evaluate.strip_required_keys(node) == {"a": {"b": {"keep": 2}}}


def test_strip_required_keys_inside_lists():
    node = [{"required": 1, "x": 1}, {"y": [{"required": 2, "z": 3}]}]
    assert evaluate.strip_required_keys(node) == [
        {"x": 1},
        {"y": [{"z": 3}]},
    ]


@pytest.mark.parametrize("value", [None, 5, "s", True])
def test_strip_required_keys_scalar_passthrough(value):
    assert evaluate.strip_required_keys(value) is value


# ---------------------------------------------------------------------------
# Hard coverage: majority_vote extras
# ---------------------------------------------------------------------------
def test_majority_vote_five_runs():
    a = [{"name": "t"}]
    b = []
    assert evaluate.majority_vote([a, b, b, a, b]) == b


def test_majority_vote_all_different_returns_first():
    a, b, c = [[{"n": 1}]], [[{"n": 2}]], []
    assert evaluate.majority_vote([a, b, c]) == a


def test_majority_vote_single_run():
    a = [{"name": "t"}]
    assert evaluate.majority_vote([a]) == a


def test_majority_vote_all_empty():
    assert evaluate.majority_vote([[], [], []]) == []


# ---------------------------------------------------------------------------
# Hard coverage: load_prime + rate extras
# ---------------------------------------------------------------------------
def test_load_prime_whitespace_only_returns_none(tmp_path):
    (tmp_path / "prime.txt").write_text("  \n ", encoding="utf-8")
    config = {"eval": {"prime_file": "prime.txt"}}
    assert evaluate.load_prime(config, tmp_path) is None


def test_load_prime_config_without_eval():
    assert evaluate.load_prime({"task": {}}, ".") is None


def test_rate_zero_hits():
    assert evaluate.rate(0, 5) == 0.0
