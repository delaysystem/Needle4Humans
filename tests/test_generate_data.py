"""Tests for scripts/generate_data.py (pure expansion plus tmp-task runs)."""

import json

import pytest

import _common
import generate_data


BANKS = {"ask": {"en": ["talk to a human", "see an agent"]}}
TOOLS = [{"name": "request_human_handoff", "parameters": {}}]


def make_seed(**overrides):
    seed = {
        "id": "en-explicit-01",
        "lang": "en",
        "category": "explicit",
        "query": "I want to {ask}.",
        "slots": {"ask": "@ask"},
        "answers": [{"name": "request_human_handoff", "arguments": {}}],
        "reasoning": "'{ask}' -> handoff",
    }
    seed.update(overrides)
    return seed


def test_expand_seed_cartesian_product():
    seed = make_seed(
        slots={"a": ["x", "y"], "b": ["1", "2", "3"]},
        query="{a} {b}",
        reasoning="{a}{b}",
        answers=[],
    )
    variants = generate_data.expand_seed(seed, {}, 7, 12)
    assert len(variants) == 6
    queries = sorted(query for query, _, _ in variants)
    assert queries == ["x 1", "x 2", "x 3", "y 1", "y 2", "y 3"]


def test_expand_seed_deterministic_and_capped():
    seed = make_seed(
        slots={"a": [f"v{i}" for i in range(10)]},
        query="{a}",
        reasoning="{a}",
    )
    first = generate_data.expand_seed(seed, {}, 7, 4)
    second = generate_data.expand_seed(seed, {}, 7, 4)
    assert first == second
    assert len(first) == 4


def test_expand_seed_bank_reference():
    variants = generate_data.expand_seed(make_seed(), BANKS, 7, 12)
    assert sorted(query for query, _, _ in variants) == [
        "I want to see an agent.",
        "I want to talk to a human.",
    ]


def test_expand_seed_unknown_bank_dies():
    with pytest.raises(SystemExit):
        generate_data.expand_seed(make_seed(slots={"ask": "@nope"}), BANKS, 7, 12)


def test_expand_seed_bad_placeholder_dies():
    with pytest.raises(SystemExit):
        generate_data.expand_seed(make_seed(query="{missing}"), BANKS, 7, 12)


def test_expand_seed_missing_key_dies():
    seed = make_seed()
    del seed["reasoning"]
    with pytest.raises(SystemExit):
        generate_data.expand_seed(seed, BANKS, 7, 12)


def write_task(tmp_path, seeds_train, seeds_eval):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[task]\n"
        'name = "mini"\n'
        'tools = "tools.json"\n'
        'slot_banks = "banks.json"\n'
        'seed_bank = "train_seeds.jsonl"\n'
        'eval_bank = "eval_seeds.jsonl"\n'
        'out_dir = "out"\n'
        "[data]\n"
        'languages = ["en"]\n'
        'categories = ["explicit", "negative"]\n'
        "train_per_cell = 2\n"
        "eval_per_cell = 1\n"
        "negative_ratio_min = 0.1\n"
        "negative_ratio_max = 0.9\n"
        "train_seed = 7\n"
        "max_variants_per_seed = 12\n"
        'train_file = "train.jsonl"\n'
        'eval_file = "eval.jsonl"\n'
        'manifest_file = "manifest.json"\n'
        "max_len_tokens = 1024\n"
        "fragmentation_warning = 1.7\n",
        encoding="utf-8",
    )
    (task_dir / "tools.json").write_text(json.dumps(TOOLS), encoding="utf-8")
    (task_dir / "banks.json").write_text(json.dumps(BANKS), encoding="utf-8")
    _common.write_jsonl(task_dir / "train_seeds.jsonl", seeds_train)
    _common.write_jsonl(task_dir / "eval_seeds.jsonl", seeds_eval)
    return task_dir


def test_main_writes_train_eval_manifest(tmp_path):
    seeds_train = [
        make_seed(),
        make_seed(
            id="en-negative-01",
            category="negative",
            query="What time is it?",
            slots={},
            answers=[],
            reasoning="routine -> []",
        ),
        make_seed(
            id="en-negative-02",
            category="negative",
            query="Thanks!",
            slots={},
            answers=[],
            reasoning="thanks -> []",
        ),
    ]
    seeds_eval = [
        make_seed(
            id="en-explicit-e01",
            slots={},
            query="I want to talk.",
            reasoning="'talk' -> handoff",
        ),
        make_seed(
            id="en-negative-e01",
            category="negative",
            query="Hello.",
            slots={},
            answers=[],
            reasoning="greeting -> []",
        ),
    ]
    task_dir = write_task(tmp_path, seeds_train, seeds_eval)
    assert generate_data.main(["--task", str(task_dir / "task.toml")]) == 0
    train = _common.read_jsonl(task_dir / "out" / "train.jsonl")
    assert len(train) == 4
    assert {row["id"] for row in train} == {f"tr-{n:05d}" for n in range(4)}
    assert all(set(row) == {"id", "query", "tools", "answers", "reasoning"}
               for row in train)
    eval_rows = _common.read_jsonl(task_dir / "out" / "eval.jsonl")
    assert len(eval_rows) == 2
    with open(task_dir / "out" / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert len(manifest["rows"]) == 6


def test_main_short_cell_dies(tmp_path):
    seeds_train = [make_seed(id="en-explicit-e01", slots={},
                             query="I want to talk.")]
    seeds_eval = [make_seed(id="en-explicit-e01", slots={},
                            query="I want to talk.")]
    task_dir = write_task(tmp_path, seeds_train, seeds_eval)
    with pytest.raises(SystemExit):
        generate_data.main(["--task", str(task_dir / "task.toml"),
                            "--split", "train"])


def test_output_paths_for_split(tmp_path):
    data = {
        "train_file": "train.jsonl",
        "eval_file": "eval.jsonl",
        "manifest_file": "manifest.json",
    }
    out = tmp_path / "out"
    assert generate_data.output_paths_for_split(data, out, "train") == [
        out / "train.jsonl", out / "manifest.json",
    ]
    assert generate_data.output_paths_for_split(data, out, "eval") == [
        out / "eval.jsonl", out / "manifest.json",
    ]
    assert generate_data.output_paths_for_split(data, out, "both") == [
        out / "train.jsonl", out / "eval.jsonl", out / "manifest.json",
    ]


def test_main_refuses_overwrite_without_flag(tmp_path):
    seeds_train = [
        make_seed(),
        make_seed(
            id="en-negative-01",
            category="negative",
            query="What time is it?",
            slots={},
            answers=[],
            reasoning="routine -> []",
        ),
        make_seed(
            id="en-negative-02",
            category="negative",
            query="Thanks!",
            slots={},
            answers=[],
            reasoning="thanks -> []",
        ),
    ]
    seeds_eval = [
        make_seed(
            id="en-explicit-e01",
            slots={},
            query="I want to talk.",
            reasoning="'talk' -> handoff",
        ),
        make_seed(
            id="en-negative-e01",
            category="negative",
            query="Hello.",
            slots={},
            answers=[],
            reasoning="greeting -> []",
        ),
    ]
    task_dir = write_task(tmp_path, seeds_train, seeds_eval)
    argv = ["--task", str(task_dir / "task.toml")]
    assert generate_data.main(argv) == 0
    train_path = task_dir / "out" / "train.jsonl"
    train_path.write_text("sentinel", encoding="utf-8")
    with pytest.raises(SystemExit):
        generate_data.main(argv)
    assert train_path.read_text(encoding="utf-8") == "sentinel"
    assert generate_data.main(argv + ["--overwrite"]) == 0
    assert train_path.read_text(encoding="utf-8") != "sentinel"
