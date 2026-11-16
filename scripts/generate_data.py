#!/usr/bin/env python3
"""Expand a task's seed banks into train/eval JSONL for ``cactus finetune``.

Each seed carries a ``query`` template with ``{slot}`` placeholders, an
optional ``slots`` map (literal value lists or ``"@bank"`` references into
the task's slot banks file, resolved per seed language), an ``answers``
template, and a ``reasoning`` template. Expansion is the cartesian product
of the slots in sorted-slot order, deterministically shuffled and capped,
so the same seed bank plus ``train_seed`` always yields the same dataset.

Fails loudly when a (language, category) cell is short of its quota: add
seeds, never silently duplicate rows. Refuses to overwrite existing
outputs unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def resolve_slots(seed: dict, banks: dict) -> dict[str, list[str]]:
    """Resolve a seed's slots to {name: values}, following @bank refs."""
    resolved: dict[str, list[str]] = {}
    for name in sorted(seed.get("slots", {})):
        value = seed["slots"][name]
        if isinstance(value, str) and value.startswith("@"):
            bank = value[1:]
            lang = seed["lang"]
            try:
                resolved[name] = list(banks[bank][lang])
            except KeyError:
                _common.die(
                    f"seed {seed.get('id', '?')}: unknown bank ref "
                    f"'{value}' for language '{lang}'"
                )
        else:
            resolved[name] = list(value)
    return resolved


def substitute(node, mapping: dict[str, str], seed_id: str):
    """Recursively apply {slot} substitution to strings in a structure."""
    if isinstance(node, str):
        try:
            return node.format(**mapping)
        except (KeyError, IndexError) as exc:
            _common.die(f"seed {seed_id}: bad placeholder in {node!r}: {exc}")
    if isinstance(node, list):
        return [substitute(item, mapping, seed_id) for item in node]
    if isinstance(node, dict):
        return {
            key: substitute(item, mapping, seed_id) for key, item in node.items()
        }
    return node


def expand_seed(
    seed: dict, banks: dict, base_seed: int, cap: int
) -> list[tuple[str, list, str]]:
    """Expand one seed into (query, answers, reasoning) variants."""
    for key in ("id", "lang", "category", "query", "answers", "reasoning"):
        if key not in seed:
            _common.die(f"seed {seed.get('id', '?')}: missing key '{key}'")
    resolved = resolve_slots(seed, banks)
    names = sorted(resolved)
    combos = list(itertools.product(*(resolved[name] for name in names))) or [()]
    rng = random.Random(f"{base_seed}:{seed['id']}")
    rng.shuffle(combos)
    variants = []
    for mapping_values in combos[:cap]:
        mapping = dict(zip(names, mapping_values))
        variants.append(
            (
                substitute(seed["query"], mapping, seed["id"]),
                substitute(seed["answers"], mapping, seed["id"]),
                substitute(seed["reasoning"], mapping, seed["id"]),
            )
        )
    return variants


def expand_bank(
    seeds: list[dict],
    banks: dict,
    tools: list,
    base_seed: int,
    cap: int,
) -> list[dict]:
    """Expand every seed, attaching provenance metadata per row."""
    rows = []
    for seed in seeds:
        for index, (query, answers, reasoning) in enumerate(
            expand_seed(seed, banks, base_seed, cap)
        ):
            rows.append(
                {
                    "query": query,
                    "tools": tools,
                    "answers": answers,
                    "reasoning": reasoning,
                    "_meta": {
                        "seed": seed["id"],
                        "lang": seed["lang"],
                        "category": seed["category"],
                        "variant": index,
                    },
                }
            )
    return rows


def check_cell_counts(
    rows: list[dict], languages: list[str], categories: list[str], kind: str
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        meta = row["_meta"]
        lang, category = meta["lang"], meta["category"]
        if lang not in languages:
            _common.die(f"{kind} row from seed {meta['seed']}: "
                        f"language '{lang}' not in task config")
        if category not in categories:
            _common.die(f"{kind} row from seed {meta['seed']}: "
                        f"category '{category}' not in task config")
        counts[(lang, category)] = counts.get((lang, category), 0) + 1
    return counts


def output_paths_for_split(
    data: dict, out_dir: Path, split: str
) -> list[Path]:
    """Output files a generate run with this split will write."""
    paths = []
    if split in ("train", "both"):
        paths.append(out_dir / data["train_file"])
    if split in ("eval", "both"):
        paths.append(out_dir / data["eval_file"])
    paths.append(out_dir / data["manifest_file"])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval", "both"],
        default="both",
        help="Which split(s) to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files instead of refusing.",
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    data = config["data"]
    out_dir = _common.task_relative_path(task_dir, config["task"]["out_dir"])
    _common.check_no_overwrite(
        output_paths_for_split(data, out_dir, args.split), args.overwrite
    )
    tools = _common.read_json(
        _common.task_relative_path(task_dir, config["task"]["tools"])
    )
    banks = _common.read_json(
        _common.task_relative_path(task_dir, config["task"]["slot_banks"])
    )
    languages = data["languages"]
    categories = data["categories"]
    cap = data["max_variants_per_seed"]
    manifest: dict = {
        "task": config["task"]["name"],
        "train_seed": data["train_seed"],
        "rows": {},
    }

    if args.split in ("train", "both"):
        seeds = _common.read_jsonl(
            _common.task_relative_path(task_dir, config["task"]["seed_bank"])
        )
        rows = expand_bank(seeds, banks, tools, data["train_seed"], cap)
        counts = check_cell_counts(rows, languages, categories, "train")
        short = {
            cell: counts.get(cell, 0)
            for cell in itertools.product(languages, categories)
            if counts.get(cell, 0) < data["train_per_cell"]
        }
        if short:
            detail = ", ".join(
                f"{lang}/{cat} has {found}, need {data['train_per_cell']}"
                for (lang, cat), found in sorted(short.items())
            )
            _common.die(f"train cells short of quota: {detail}")
        random.Random(data["train_seed"]).shuffle(rows)
        train_rows = []
        for number, row in enumerate(rows):
            row_id = f"tr-{number:05d}"
            manifest["rows"][row_id] = row.pop("_meta")
            row["id"] = row_id
            train_rows.append(row)
        train_path = out_dir / data["train_file"]
        _common.write_jsonl(train_path, train_rows)
        print(f"wrote {len(train_rows)} train rows to {train_path}")

    if args.split in ("eval", "both"):
        seeds = _common.read_jsonl(
            _common.task_relative_path(task_dir, config["task"]["eval_bank"])
        )
        rows = expand_bank(seeds, banks, tools, data["train_seed"], cap)
        counts = check_cell_counts(rows, languages, categories, "eval")
        wrong = {
            cell: counts.get(cell, 0)
            for cell in itertools.product(languages, categories)
            if counts.get(cell, 0) != data["eval_per_cell"]
        }
        if wrong:
            detail = ", ".join(
                f"{lang}/{cat} has {found}, need exactly "
                f"{data['eval_per_cell']}"
                for (lang, cat), found in sorted(wrong.items())
            )
            _common.die(f"eval cells off quota: {detail}")
        eval_rows = []
        for number, row in enumerate(rows):
            row_id = f"ev-{number:05d}"
            manifest["rows"][row_id] = row.pop("_meta")
            row["id"] = row_id
            eval_rows.append(row)
        eval_path = out_dir / data["eval_file"]
        _common.write_jsonl(eval_path, eval_rows)
        print(f"wrote {len(eval_rows)} eval rows to {eval_path}")

    manifest_path = out_dir / data["manifest_file"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
