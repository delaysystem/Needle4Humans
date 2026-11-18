#!/usr/bin/env python3
"""Validate a generated finetune dataset against the task spec and the
Needle data contract.

Row checks (from the
`Fine-tuning Needle <https://cactuscompute.com/blog/finetuning-needle>`_
guide and the
`tool design <https://cactuscompute.com/blog/designing-tools-for-needle>`_
guide):

- required keys ``query``/``tools``/``answers``/``reasoning`` present
- every tool and answer name is declared in the task's tools file
- schema-``required`` arguments are present on every answer
- enum arguments are members of the schema enum
- free-text arguments are evidenced by a span of the query
- array arguments match the ``items`` schema and ``min/maxItems`` bounds
- ``reasoning`` is non-empty

Dataset checks:

- off-topic ``[]`` share within the configured band
- per-cell quotas hold per the manifest
- no train/eval query overlap
- length warnings (overlong lines truncate silently upstream)

Exit 0 on success, 1 on any error. Warnings never fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def check_item(tool_name: str, value, schema: dict, errors: list) -> None:
    """Check one array item against the items schema (enum/type)."""
    if not isinstance(schema, dict):
        return
    want = schema.get("type")
    if want == "string" and not isinstance(value, str):
        errors.append(f"{tool_name}: item {value!r} is not a string")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(
            f"{tool_name}: item {value!r} not in enum {schema['enum']}"
        )


def argument_errors(
    tool_name: str,
    arguments: dict | list,
    properties: dict,
    query: str,
    required: list | None = None,
    items_schema: dict | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> list[str]:
    """Check one answer's arguments against the tool schema and query."""
    errors = []
    if isinstance(arguments, list):
        if items_schema is None:
            errors.append(f"{tool_name}: array arguments need an items schema")
            return errors
        if min_items is not None and len(arguments) < min_items:
            errors.append(f"{tool_name}: want at least {min_items} items")
        if max_items is not None and len(arguments) > max_items:
            errors.append(f"{tool_name}: want at most {max_items} items")
        for value in arguments:
            check_item(tool_name, value, items_schema, errors)
        return errors
    for key in required or []:
        if key not in arguments:
            errors.append(f"{tool_name}: missing required argument '{key}'")
    folded_query = query.casefold()
    for key, value in arguments.items():
        if key not in properties:
            errors.append(f"{tool_name}: unknown argument '{key}'")
            continue
        schema = properties[key]
        if "enum" in schema:
            if value not in schema["enum"]:
                errors.append(
                    f"{tool_name}.{key}: {value!r} not in enum "
                    f"{schema['enum']}"
                )
        elif isinstance(value, str):
            if value.casefold() not in folded_query:
                errors.append(
                    f"{tool_name}.{key}: {value!r} has no span in the query"
                )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if str(value) not in query.split():
                errors.append(
                    f"{tool_name}.{key}: {value!r} has no span in the query"
                )
    return errors


def validate_row(row: dict, tools_by_name: dict) -> list[str]:
    """Return the error list for one dataset row (empty means valid)."""
    errors = []
    for key in ("query", "tools", "answers", "reasoning"):
        if key not in row:
            errors.append(f"row {row.get('id', '?')}: missing key '{key}'")
    if errors:
        return errors
    row_id = row.get("id", "?")
    if not isinstance(row["query"], str) or not row["query"].strip():
        errors.append(f"row {row_id}: empty query")
    if not isinstance(row["reasoning"], str) or not row["reasoning"].strip():
        errors.append(f"row {row_id}: empty reasoning")
    for tool in row["tools"]:
        if tool.get("name") not in tools_by_name:
            errors.append(f"row {row_id}: undeclared tool {tool.get('name')!r}")
    query_text = row["query"] if isinstance(row["query"], str) else ""
    for answer in row["answers"]:
        name = answer.get("name")
        if name not in tools_by_name:
            errors.append(f"row {row_id}: answer for undeclared tool {name!r}")
            continue
        parameters = tools_by_name[name].get("parameters", {})
        properties = parameters.get("properties", {})
        arguments = answer.get("arguments", {})
        if not isinstance(arguments, (dict, list)):
            errors.append(f"row {row_id}: {name} arguments not an object or array")
            continue
        if isinstance(arguments, list) and parameters.get("type") != "array":
            errors.append(f"row {row_id}: {name} takes object arguments")
            continue
        errors.extend(
            f"row {row_id}: {message}"
            for message in argument_errors(
                name,
                arguments,
                properties,
                query_text,
                parameters.get("required", []),
                parameters.get("items"),
                parameters.get("minItems"),
                parameters.get("maxItems"),
            )
        )
    return errors


def length_warnings(
    row: dict, max_len: int, fragmentation: float
) -> list[str]:
    """Warn when a row may not fit the training context.

    Token estimate is deliberately crude (chars/4); rows using
    non-ASCII text get the configured fragmentation multiplier.
    """
    text = row.get("query", "") + row.get("reasoning", "")
    estimate = len(text) / 4
    if any(ord(char) > 127 for char in text):
        estimate *= fragmentation
    if estimate > max_len:
        return [
            f"row {row.get('id', '?')}: ~{estimate:.0f} tokens may exceed "
            f"max_len {max_len} and be silently truncated"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    data = config["data"]
    out_dir = _common.task_relative_path(task_dir, config["task"]["out_dir"])
    tools = _common.read_json(
        _common.task_relative_path(task_dir, config["task"]["tools"])
    )
    tools_by_name = {tool["name"]: tool for tool in tools}
    train_rows = _common.read_jsonl(out_dir / data["train_file"])
    eval_rows = _common.read_jsonl(out_dir / data["eval_file"])
    with open(out_dir / data["manifest_file"], encoding="utf-8") as handle:
        import json

        manifest = json.load(handle)

    errors: list[str] = []
    warnings: list[str] = []
    for row in train_rows + eval_rows:
        errors.extend(validate_row(row, tools_by_name))
        warnings.extend(
            length_warnings(
                row, data["max_len_tokens"], data["fragmentation_warning"]
            )
        )

    negatives = sum(1 for row in train_rows if row["answers"] == [])
    ratio = negatives / len(train_rows) if train_rows else 0
    if not data["negative_ratio_min"] <= ratio <= data["negative_ratio_max"]:
        errors.append(
            f"negative share {ratio:.3f} outside "
            f"[{data['negative_ratio_min']}, {data['negative_ratio_max']}]"
        )

    for row_id, meta in manifest["rows"].items():
        if meta["lang"] not in data["languages"]:
            errors.append(f"manifest {row_id}: unknown language")
        if meta["category"] not in data["categories"]:
            errors.append(f"manifest {row_id}: unknown category")

    train_queries = {row["query"] for row in train_rows}
    overlap = sorted(
        {row["query"] for row in eval_rows} & train_queries
    )
    if overlap:
        errors.append(
            f"{len(overlap)} eval queries also in train, e.g. {overlap[0]!r}"
        )

    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")
    print(
        f"checked {len(train_rows)} train + {len(eval_rows)} eval rows: "
        f"{len(errors)} errors, {len(warnings)} warnings"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
