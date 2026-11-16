"""Tests for scripts/evaluate.py scoring."""

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
