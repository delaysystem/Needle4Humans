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


# ---------------------------------------------------------------------------
# Hard coverage: resolve_slots
# ---------------------------------------------------------------------------
def test_resolve_slots_inline_lists_copied():
    seed = make_seed(slots={"a": ["x", "y"]})
    resolved = generate_data.resolve_slots(seed, {})
    assert resolved == {"a": ["x", "y"]}
    assert resolved["a"] is not seed["slots"]["a"]


def test_resolve_slots_bank_per_language():
    banks = {"ask": {"en": ["a"], "de": ["b", "c"]}}
    assert generate_data.resolve_slots(make_seed(), banks)["ask"] == ["a"]
    seed = make_seed(lang="de")
    assert generate_data.resolve_slots(seed, banks)["ask"] == ["b", "c"]


def test_resolve_slots_unknown_bank_dies():
    with pytest.raises(SystemExit):
        generate_data.resolve_slots(make_seed(slots={"a": "@nope"}), BANKS)


def test_resolve_slots_bank_missing_language_dies():
    with pytest.raises(SystemExit):
        generate_data.resolve_slots(make_seed(lang="xx"), BANKS)


def test_resolve_slots_no_slots_key():
    seed = make_seed()
    del seed["slots"]
    assert generate_data.resolve_slots(seed, {}) == {}


def test_resolve_slots_empty_slots():
    assert generate_data.resolve_slots(make_seed(slots={}), {}) == {}


def test_resolve_slots_bare_string_splits_chars():
    # Current behavior: a non-@ string slot becomes its character list.
    resolved = generate_data.resolve_slots(make_seed(slots={"a": "xy"}), {})
    assert resolved == {"a": ["x", "y"]}


# ---------------------------------------------------------------------------
# Hard coverage: substitute
# ---------------------------------------------------------------------------
def test_substitute_plain_and_repeated():
    assert generate_data.substitute("no-op", {}, "s") == "no-op"
    assert (
        generate_data.substitute("{a} and {a}", {"a": "x"}, "s") == "x and x"
    )


def test_substitute_missing_key_dies():
    with pytest.raises(SystemExit):
        generate_data.substitute("{nope}", {}, "s-1")


def test_substitute_positional_placeholder_dies():
    with pytest.raises(SystemExit):
        generate_data.substitute("{}", {"a": "x"}, "s-1")


def test_substitute_nested_structures():
    node = {"q": ["{a}!", {"deep": "{b}?"}], "r": "{a}{b}"}
    assert generate_data.substitute(node, {"a": "x", "b": "y"}, "s") == {
        "q": ["x!", {"deep": "y?"}],
        "r": "xy",
    }


@pytest.mark.parametrize("value", [5, 3.5, None, True, False])
def test_substitute_non_string_passthrough(value):
    assert generate_data.substitute(value, {"a": "x"}, "s") is value


def test_substitute_escaped_braces():
    assert generate_data.substitute("{{a}} {a}", {"a": "x"}, "s") == "{a} x"


# ---------------------------------------------------------------------------
# Hard coverage: expand_seed
# ---------------------------------------------------------------------------
def test_expand_seed_no_slots_single_variant():
    variants = generate_data.expand_seed(
        make_seed(slots={}, query="Q", reasoning="r"), {}, 7, 12)
    assert len(variants) == 1
    query, answers, reasoning = variants[0]
    assert (query, reasoning) == ("Q", "r")


def test_expand_seed_cap_zero_yields_nothing():
    assert generate_data.expand_seed(make_seed(), BANKS, 7, 0) == []


def test_expand_seed_cap_above_combos_keeps_all():
    seed = make_seed(slots={"a": ["x", "y"]}, query="{a}", reasoning="{a}")
    assert len(generate_data.expand_seed(seed, {}, 7, 99)) == 2


def test_expand_seed_different_base_seed_same_set():
    seed = make_seed(
        slots={"a": [f"v{i}" for i in range(8)]}, query="{a}", reasoning="{a}"
    )
    first = generate_data.expand_seed(seed, {}, 7, 8)
    second = generate_data.expand_seed(seed, {}, 8, 8)
    assert sorted(first) == sorted(second)


def test_expand_seed_answers_substituted_per_combo():
    seed = make_seed(
        slots={"r": ["billing", "general"]},
        query="Q {r}",
        answers=[{"name": "t", "arguments": {"reason": "{r}"}}],
        reasoning="{r}",
    )
    reasons = sorted(
        answers[0]["arguments"]["reason"]
        for _, answers, _ in generate_data.expand_seed(seed, {}, 7, 12)
    )
    assert reasons == ["billing", "general"]


@pytest.mark.parametrize("key", ["id", "lang", "category", "query", "answers"])
def test_expand_seed_each_missing_key_dies(key):
    seed = make_seed()
    del seed[key]
    with pytest.raises(SystemExit):
        generate_data.expand_seed(seed, {}, 7, 12)


# ---------------------------------------------------------------------------
# Hard coverage: expand_bank + check_cell_counts
# ---------------------------------------------------------------------------
def test_expand_bank_empty_seeds():
    assert generate_data.expand_bank([], {}, TOOLS, 7, 12) == []


def test_expand_bank_provenance_and_order():
    seeds = [
        make_seed(id="s1", slots={}, query="Q1", reasoning="r1"),
        make_seed(id="s2", slots={}, query="Q2", reasoning="r2"),
    ]
    rows = generate_data.expand_bank(seeds, {}, TOOLS, 7, 12)
    assert [row["_meta"]["seed"] for row in rows] == ["s1", "s2"]
    assert rows[0]["_meta"] == {
        "seed": "s1",
        "lang": "en",
        "category": "explicit",
        "variant": 0,
    }
    assert all(row["tools"] is TOOLS for row in rows)


def test_expand_bank_variant_indices():
    seed = make_seed(
        slots={"a": ["x", "y", "z"]}, query="{a}", reasoning="{a}"
    )
    rows = generate_data.expand_bank([seed], {}, TOOLS, 7, 12)
    assert sorted(row["_meta"]["variant"] for row in rows) == [0, 1, 2]


def _meta_row(seed, lang, category):
    return {"_meta": {"seed": seed, "lang": lang, "category": category}}


def test_check_cell_counts_tallies():
    rows = [
        _meta_row("a", "en", "explicit"),
        _meta_row("b", "en", "explicit"),
        _meta_row("c", "de", "negative"),
    ]
    assert generate_data.check_cell_counts(
        rows, ["en", "de"], ["explicit", "negative"], "train"
    ) == {("en", "explicit"): 2, ("de", "negative"): 1}


def test_check_cell_counts_empty():
    assert generate_data.check_cell_counts([], ["en"], ["explicit"], "t") == {}


def test_check_cell_counts_unknown_language_dies():
    with pytest.raises(SystemExit):
        generate_data.check_cell_counts(
            [_meta_row("a", "xx", "explicit")], ["en"], ["explicit"], "train"
        )


def test_check_cell_counts_unknown_category_dies():
    with pytest.raises(SystemExit):
        generate_data.check_cell_counts(
            [_meta_row("a", "en", "nope")], ["en"], ["explicit"], "eval"
        )


def test_main_eval_split_only(tmp_path):
    task_dir = write_task(
        tmp_path,
        [make_seed(),
         make_seed(id="en-negative-01", category="negative",
                   query="T?", slots={}, answers=[], reasoning="r"),
         make_seed(id="en-negative-02", category="negative",
                   query="U?", slots={}, answers=[], reasoning="r")],
        [make_seed(id="e1", slots={}, query="Q?", reasoning="r"),
         make_seed(id="e2", category="negative", slots={}, query="H?",
                   answers=[], reasoning="r")],
    )
    argv = ["--task", str(task_dir / "task.toml"), "--split", "eval"]
    assert generate_data.main(argv) == 0
    assert (task_dir / "out" / "eval.jsonl").is_file()
    assert not (task_dir / "out" / "train.jsonl").is_file()
