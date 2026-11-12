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


def test_length_warnings():
    short = {"id": "x", "query": "hi", "reasoning": "ok"}
    assert validate_data.length_warnings(short, 1024, 1.7) == []
    long_row = {"id": "y", "query": "w" * 5000, "reasoning": ""}
    assert validate_data.length_warnings(long_row, 1024, 1.7) != []
    # Non-ASCII text gets the fragmentation multiplier.
    borderline = {"id": "z", "query": "ü" * 3000, "reasoning": ""}
    assert validate_data.length_warnings(borderline, 1024, 1.7) != []
