"""Tests for scripts/validate_data.py row checks."""

import validate_data


TOOLS = {
    "request_human_handoff": {
        "name": "request_human_handoff",
        "parameters": {
            "properties": {
                "reason": {"enum": ["billing", "technical"]},
                "details": {"type": "string"},
                "count": {"type": "integer"},
            }
        },
    }
}


def make_row(**overrides):
    row = {
        "id": "tr-00000",
        "query": "My bill is wrong. I want to talk to a human.",
        "tools": [{"name": "request_human_handoff"}],
        "answers": [
            {
                "name": "request_human_handoff",
                "arguments": {"reason": "billing"},
            }
        ],
        "reasoning": "'bill is wrong' -> billing; 'talk to a human' -> handoff",
    }
    row.update(overrides)
    return row


def test_valid_row():
    assert validate_data.validate_row(make_row(), TOOLS) == []


def test_missing_required_argument():
    tools = {
        "request_human_handoff": {
            "name": "request_human_handoff",
            "parameters": {
                "properties": {"reason": {"enum": ["billing", "general"]}},
                "required": ["reason"],
            },
        }
    }
    row = make_row(answers=[{"name": "request_human_handoff", "arguments": {}}])
    errors = validate_data.validate_row(row, tools)
    assert len(errors) == 1 and "required" in errors[0]
    row["answers"][0]["arguments"] = {"reason": "general"}
    assert validate_data.validate_row(row, tools) == []


def test_array_arguments():
    tools = {
        "handoff": {
            "name": "handoff",
            "parameters": {
                "type": "array",
                "items": {"type": "string", "enum": ["billing", "general"]},
                "minItems": 1,
                "maxItems": 1,
            },
        }
    }
    ok = make_row(
        tools=[{"name": "handoff"}],
        answers=[{"name": "handoff", "arguments": ["billing"]}],
    )
    assert validate_data.validate_row(ok, tools) == []
    for bad_args in ([], ["billing", "general"], ["bogus"], [7]):
        row = make_row(
            tools=[{"name": "handoff"}],
            answers=[{"name": "handoff", "arguments": bad_args}],
        )
        assert validate_data.validate_row(row, tools) != [], bad_args
    obj_tools = {
        "handoff": {
            "name": "handoff",
            "parameters": {"type": "object", "properties": {}},
        }
    }
    row = make_row(
        tools=[{"name": "handoff"}],
        answers=[{"name": "handoff", "arguments": ["billing"]}],
    )
    assert validate_data.validate_row(row, obj_tools) != []


def test_valid_negative_row():
    row = make_row(
        query="What are your hours?",
        answers=[],
        reasoning="routine -> []",
    )
    assert validate_data.validate_row(row, TOOLS) == []


def test_missing_key():
    row = make_row()
    del row["reasoning"]
    assert validate_data.validate_row(row, TOOLS) != []


def test_empty_reasoning():
    assert validate_data.validate_row(make_row(reasoning="  "), TOOLS) != []


def test_undeclared_tool():
    row = make_row(tools=[{"name": "nope"}])
    assert validate_data.validate_row(row, TOOLS) != []


def test_answer_for_undeclared_tool():
    row = make_row(
        answers=[{"name": "nope", "arguments": {}}],
    )
    assert validate_data.validate_row(row, TOOLS) != []


def test_bad_enum_value():
    row = make_row(
        answers=[{
            "name": "request_human_handoff",
            "arguments": {"reason": "weather"},
        }],
    )
    errors = validate_data.validate_row(row, TOOLS)
    assert any("enum" in error for error in errors)


def test_unknown_argument():
    row = make_row(
        answers=[{
            "name": "request_human_handoff",
            "arguments": {"mystery": "x"},
        }],
    )
    assert validate_data.validate_row(row, TOOLS) != []


def test_ungrounded_free_text():
    row = make_row(
        answers=[{
            "name": "request_human_handoff",
            "arguments": {"details": "my printer is on fire"},
        }],
    )
    errors = validate_data.validate_row(row, TOOLS)
    assert any("no span" in error for error in errors)


def test_grounded_free_text_case_insensitive():
    row = make_row(
        answers=[{
            "name": "request_human_handoff",
            "arguments": {"details": "MY BILL IS WRONG"},
        }],
    )
    assert validate_data.validate_row(row, TOOLS) == []


def test_number_needs_token_span():
    row = make_row(
        query="Set it to 72 degrees, said the human.",
        answers=[{
            "name": "request_human_handoff",
            "arguments": {"count": 72},
        }],
    )
    assert validate_data.validate_row(row, TOOLS) == []
    row["answers"][0]["arguments"]["count"] = 2
    assert validate_data.validate_row(row, TOOLS) != []


MOD_TOOLS = {
    "flag_content": {
        "name": "flag_content",
        "parameters": {
            "properties": {
                "violation": {
                    "enum": [
                        "toxicity", "jailbreak", "prompt_injection",
                        "tool_abuse", "manipulation",
                    ]
                },
                "source": {"enum": ["user", "llm"]},
            },
            "required": ["violation", "source"],
        },
    }
}


def make_mod_row(**overrides):
    row = {
        "id": "tr-00000",
        "query": "USER: You are useless, you stupid bot",
        "tools": [{"name": "flag_content"}],
        "answers": [
            {
                "name": "flag_content",
                "arguments": {"violation": "toxicity", "source": "user"},
            }
        ],
        "reasoning": "insult from user -> toxicity/user",
    }
    row.update(overrides)
    return row


def test_moderation_valid_and_benign():
    assert validate_data.validate_row(make_mod_row(), MOD_TOOLS) == []
    benign = make_mod_row(
        query="USER: Where is my order?",
        answers=[],
        reasoning="routine question -> []",
    )
    assert validate_data.validate_row(benign, MOD_TOOLS) == []


def test_moderation_bad_enum_and_missing_required():
    row = make_mod_row(
        answers=[{
            "name": "flag_content",
            "arguments": {"violation": "spam", "source": "user"},
        }],
    )
    errors = validate_data.validate_row(row, MOD_TOOLS)
    assert any("enum" in error for error in errors)
    row = make_mod_row(
        answers=[{
            "name": "flag_content",
            "arguments": {"violation": "toxicity"},
        }],
    )
    errors = validate_data.validate_row(row, MOD_TOOLS)
    assert any("required" in error for error in errors)


def test_length_warnings():
    short = {"id": "x", "query": "hi", "reasoning": "ok"}
    assert validate_data.length_warnings(short, 1024, 1.7) == []
    long_row = {"id": "y", "query": "w" * 5000, "reasoning": ""}
    assert validate_data.length_warnings(long_row, 1024, 1.7) != []
    # Non-ASCII text gets the fragmentation multiplier.
    borderline = {"id": "z", "query": "ü" * 3000, "reasoning": ""}
    assert validate_data.length_warnings(borderline, 1024, 1.7) != []


# ---------------------------------------------------------------------------
# Hard coverage: check_item matrix
# ---------------------------------------------------------------------------
import pytest


@pytest.mark.parametrize(
    "value,schema,n_errors",
    [
        ("billing", {"type": "string"}, 0),
        (5, {"type": "string"}, 1),
        (None, {"type": "string"}, 1),
        (["x"], {"type": "string"}, 1),
        ("billing", {"enum": ["billing", "general"]}, 0),
        ("zzz", {"enum": ["billing", "general"]}, 1),
        ("Billing", {"enum": ["billing"]}, 1),
        (1, {"enum": [1, 2]}, 0),
        (3, {"enum": [1, 2]}, 1),
        (True, {"enum": [True, False]}, 0),
        (1, {"enum": [True]}, 0),  # 1 == True in Python: documents equality
        ("anything", {}, 0),
        ({"a": 1}, {"type": "object"}, 0),  # unhandled types pass silently
    ],
)
def test_check_item_matrix(value, schema, n_errors):
    errors = []
    validate_data.check_item("tool", value, schema, errors)
    assert len(errors) == n_errors


def test_check_item_non_dict_schema_passes():
    errors = []
    validate_data.check_item("tool", "x", None, errors)
    validate_data.check_item("tool", "x", "string", errors)
    assert errors == []


# ---------------------------------------------------------------------------
# Hard coverage: argument_errors on objects
# ---------------------------------------------------------------------------
PROPS = {
    "reason": {"enum": ["billing", "general"]},
    "details": {"type": "string"},
    "count": {"type": "integer"},
}


@pytest.mark.parametrize(
    "arguments,required,n_errors",
    [
        ({"reason": "billing"}, ["reason"], 0),
        ({"reason": "billing"}, ["reason", "details"], 1),
        ({}, ["reason", "details"], 2),
        ({"reason": "billing"}, [], 0),
        ({"reason": "billing"}, None, 0),
        ({"bogus": 1}, [], 1),
        ({"a": 1, "b": 2}, [], 2),
        ({"reason": "Billing"}, [], 1),
        ({"reason": "billing", "bogus": 1}, [], 1),
    ],
)
def test_argument_errors_object_shape(arguments, required, n_errors):
    errors = validate_data.argument_errors(
        "t", arguments, PROPS, "My bill is wrong.", required
    )
    assert len(errors) == n_errors


@pytest.mark.parametrize(
    "value,query,n_errors",
    [
        ("my bill", "My bill is wrong.", 0),
        ("MY BILL", "My bill is wrong.", 0),
        ("refund", "My bill is wrong.", 1),
        ("", "My bill is wrong.", 0),  # empty string is always a substring
        ("bill is wrong. And more", "My bill is wrong.", 1),
        ("ünïcodé", "Say ünïcodé now.", 0),
        ("ÜNÏCODÉ", "Say ünïcodé now.", 0),
        ("UNÏCODÉ", "Say ünïcodé now.", 1),
    ],
)
def test_argument_errors_free_text_span(value, query, n_errors):
    errors = validate_data.argument_errors(
        "t", {"details": value}, PROPS, query
    )
    assert len(errors) == n_errors


@pytest.mark.parametrize(
    "value,query,n_errors",
    [
        (72, "Set it to 72 degrees.", 0),
        (2, "Set it to 72 degrees.", 1),
        (7, "Set it to 72 degrees.", 1),  # substring of token is not a span
        (72.5, "Set it to 72.5 degrees.", 0),
        (-3, "Down by -3 points.", 0),
        (0, "Zero 0 tolerance.", 0),
        (True, "True statement here.", 0),  # bools skip the number check
        (False, "Nothing relevant.", 0),
    ],
)
def test_argument_errors_number_span(value, query, n_errors):
    errors = validate_data.argument_errors(
        "t", {"count": value}, PROPS, query
    )
    assert len(errors) == n_errors


# ---------------------------------------------------------------------------
# Hard coverage: argument_errors on arrays
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "arguments,items,min_items,max_items,n_errors",
    [
        (["a"], {"type": "string"}, None, None, 0),
        (["a", 5], {"type": "string"}, None, None, 1),
        (["x"], None, None, None, 1),
        ([], {"type": "string"}, None, None, 0),
        (["a", "b"], {"type": "string"}, 2, None, 0),
        (["a"], {"type": "string"}, 2, None, 1),
        (["a", "b", "c"], {"type": "string"}, None, 2, 1),
        (["a", "b"], {"type": "string"}, None, 2, 0),
        (["a", "b"], {"type": "string"}, 1, 3, 0),
        (["ok", "bad"], {"enum": ["ok"]}, None, None, 1),
        (["ok"], {"enum": ["ok"]}, 1, 1, 0),
        ([], {"type": "string"}, 1, 1, 1),
    ],
)
def test_argument_errors_array_matrix(
    arguments, items, min_items, max_items, n_errors
):
    errors = validate_data.argument_errors(
        "t", arguments, {}, "query unused for arrays",
        items_schema=items, min_items=min_items, max_items=max_items,
    )
    assert len(errors) == n_errors


# ---------------------------------------------------------------------------
# Hard coverage: validate_row structure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["query", "tools", "answers", "reasoning"])
def test_validate_row_each_missing_key(missing):
    row = make_row()
    del row[missing]
    errors = validate_data.validate_row(row, TOOLS)
    assert len(errors) == 1 and missing in errors[0]


def test_validate_row_missing_keys_short_circuit():
    row = make_row()
    del row["query"]
    del row["answers"]
    errors = validate_data.validate_row(row, TOOLS)
    assert len(errors) == 2


@pytest.mark.parametrize("query", ["", "   ", "\t\n", 123, None, []])
def test_validate_row_bad_query(query):
    assert validate_data.validate_row(make_row(query=query), TOOLS) != []


@pytest.mark.parametrize("reasoning", ["", "   ", 42, None])
def test_validate_row_bad_reasoning(reasoning):
    assert validate_data.validate_row(make_row(reasoning=reasoning), TOOLS) != []


def test_validate_row_tool_name_none():
    row = make_row(tools=[{"name": None}])
    errors = validate_data.validate_row(row, TOOLS)
    assert any("undeclared tool" in e for e in errors)


def test_validate_row_answer_missing_name():
    row = make_row(answers=[{"arguments": {}}])
    errors = validate_data.validate_row(row, TOOLS)
    assert any("undeclared tool" in e for e in errors)


@pytest.mark.parametrize("arguments", ["x", 5, 3.5, True])
def test_validate_row_arguments_wrong_type(arguments):
    row = make_row(
        answers=[{"name": "request_human_handoff", "arguments": arguments}]
    )
    errors = validate_data.validate_row(row, TOOLS)
    assert any("not an object or array" in e for e in errors)


def test_validate_row_list_args_against_object_type():
    row = make_row(
        answers=[{"name": "request_human_handoff", "arguments": ["billing"]}]
    )
    errors = validate_data.validate_row(row, TOOLS)
    assert any("takes object arguments" in e for e in errors)


def test_validate_row_object_args_against_array_type_passes():
    # Current behavior: only the list-against-object direction is gated.
    tools = {
        "t": {"name": "t", "parameters": {"type": "array", "properties": {}}}
    }
    row = make_row(
        tools=[{"name": "t"}],
        answers=[{"name": "t", "arguments": {}}],
    )
    assert validate_data.validate_row(row, tools) == []


def test_validate_row_multiple_answers_each_checked():
    row = make_row(
        answers=[
            {"name": "request_human_handoff", "arguments": {"reason": "billing"}},
            {"name": "request_human_handoff", "arguments": {"reason": "nope"}},
        ]
    )
    errors = validate_data.validate_row(row, TOOLS)
    assert len(errors) == 1 and "enum" in errors[0]


def test_validate_row_error_uses_fallback_id():
    row = make_row()
    del row["id"]
    del row["query"]
    errors = validate_data.validate_row(row, TOOLS)
    assert any("row ?:" in e for e in errors)


# ---------------------------------------------------------------------------
# Hard coverage: length_warnings math
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query_len,reasoning_len,max_len,frag,warns",
    [
        (100, 100, 1024, 1.7, False),
        (4096, 0, 1024, 1.7, False),  # exactly at max: strict >
        (4100, 0, 1024, 1.7, True),
        (2000, 2000, 1024, 1.7, False),  # split across fields still sums
        (2000, 2200, 1024, 1.7, True),
        (0, 0, 1024, 1.7, False),
    ],
)
def test_length_warnings_ascii_math(query_len, reasoning_len, max_len, frag, warns):
    row = {"id": "x", "query": "w" * query_len, "reasoning": "w" * reasoning_len}
    assert (validate_data.length_warnings(row, max_len, frag) != []) == warns


@pytest.mark.parametrize(
    "text,frag,max_len,warns",
    [
        ("ü" * 2000, 1.7, 1024, False),  # 500 * 1.7 = 850
        ("ü" * 3000, 1.7, 1024, True),  # 750 * 1.7 = 1275
        ("ü", 1.7, 1024, False),
        ("🎉" * 100, 1.7, 1024, False),  # emoji count as one char each
        ("w" * 4095 + "ü", 1.7, 1024, True),  # one non-ascii flips multiplier
    ],
)
def test_length_warnings_unicode_math(text, frag, max_len, warns):
    row = {"id": "x", "query": text, "reasoning": ""}
    assert (validate_data.length_warnings(row, max_len, frag) != []) == warns


def test_length_warnings_message_format():
    row = {"id": "tr-007", "query": "w" * 5000, "reasoning": ""}
    (message,) = validate_data.length_warnings(row, 1024, 1.7)
    assert "tr-007" in message and "1024" in message and "1250" in message


def test_length_warnings_missing_keys_default_empty():
    assert validate_data.length_warnings({"id": "x"}, 1024, 1.7) == []


def test_length_warnings_zero_max_len():
    assert validate_data.length_warnings({"id": "x"}, 0, 1.7) == []


def _write_validate_task(tmp_path, train_rows, eval_rows, manifest_rows):
    import json as _json

    task_dir = tmp_path / "task"
    out_dir = task_dir / "out"
    out_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "[task]\nname = \"mini\"\nout_dir = \"out\"\ntools = \"tools.json\"\n"
        "[data]\nlanguages = [\"en\"]\ncategories = [\"explicit\"]\n"
        "train_file = \"train.jsonl\"\neval_file = \"eval.jsonl\"\n"
        "manifest_file = \"manifest.json\"\n"
        "negative_ratio_min = 0.1\nnegative_ratio_max = 0.9\n"
        "max_len_tokens = 1024\nfragmentation_warning = 1.7\n",
        encoding="utf-8",
    )
    tools = [
        {
            "name": "request_human_handoff",
            "parameters": {"properties": {"reason": {"enum": ["general"]}}},
        }
    ]
    (task_dir / "tools.json").write_text(_json.dumps(tools), encoding="utf-8")
    import _common

    _common.write_jsonl(out_dir / "train.jsonl", train_rows)
    _common.write_jsonl(out_dir / "eval.jsonl", eval_rows)
    (out_dir / "manifest.json").write_text(
        _json.dumps({"rows": manifest_rows}), encoding="utf-8"
    )
    return task_dir / "task.toml"


def _mini_row(query, answers):
    return {
        "id": "tr-0",
        "query": query,
        "tools": [{"name": "request_human_handoff"}],
        "answers": answers,
        "reasoning": "r",
    }


def _mini_call(reason):
    return [{"name": "request_human_handoff", "arguments": {"reason": reason}}]


def test_main_clean_task_passes(tmp_path, capsys):
    task = _write_validate_task(
        tmp_path,
        [
            _mini_row("Talk to a person.", _mini_call("general")),
            _mini_row("What time is it?", []),
        ],
        [_mini_row("Give me a human.", _mini_call("general"))],
        {"tr-0": {"lang": "en", "category": "explicit"}},
    )
    assert validate_data.main(["--task", str(task)]) == 0
    assert "0 errors" in capsys.readouterr().out


def test_main_bad_row_fails(tmp_path):
    task = _write_validate_task(
        tmp_path,
        [
            _mini_row("Talk to a person.", _mini_call("general")),
            _mini_row("What time is it?", []),
            {"id": "bad", "query": "", "tools": [], "answers": [], "reasoning": "r"},
        ],
        [_mini_row("Give me a human.", _mini_call("general"))],
        {"tr-0": {"lang": "en", "category": "explicit"}},
    )
    assert validate_data.main(["--task", str(task)]) == 1


def test_main_ratio_violation_fails(tmp_path):
    rows = [_mini_row(f"Ask {i} for a person.", _mini_call("general")) for i in range(10)]
    task = _write_validate_task(
        tmp_path,
        rows,
        [_mini_row("Give me a human.", _mini_call("general"))],
        {"tr-0": {"lang": "en", "category": "explicit"}},
    )
    assert validate_data.main(["--task", str(task)]) == 1


def test_main_eval_train_overlap_fails(tmp_path):
    task = _write_validate_task(
        tmp_path,
        [
            _mini_row("Talk to a person.", _mini_call("general")),
            _mini_row("What time is it?", []),
        ],
        [_mini_row("Talk to a person.", _mini_call("general"))],
        {"tr-0": {"lang": "en", "category": "explicit"}},
    )
    assert validate_data.main(["--task", str(task)]) == 1


def test_main_manifest_gates_fail(tmp_path):
    task = _write_validate_task(
        tmp_path,
        [
            _mini_row("Talk to a person.", _mini_call("general")),
            _mini_row("What time is it?", []),
        ],
        [_mini_row("Give me a human.", _mini_call("general"))],
        {"tr-0": {"lang": "xx", "category": "nope"}},
    )
    assert validate_data.main(["--task", str(task)]) == 1


def test_main_ratio_boundaries_pass(tmp_path):
    rows = [_mini_row(f"Ask {i} for a person.", _mini_call("general")) for i in range(9)]
    rows.append(_mini_row("What time is it?", []))
    task = _write_validate_task(
        tmp_path,
        rows,
        [_mini_row("Give me a human.", _mini_call("general"))],
        {"tr-0": {"lang": "en", "category": "explicit"}},
    )
    assert validate_data.main(["--task", str(task)]) == 0
