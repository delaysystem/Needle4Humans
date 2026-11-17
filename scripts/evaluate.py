#!/usr/bin/env python3
"""Score base vs tuned weights on the task's frozen eval set.

Runs ``Cactus.complete()`` for every eval query under the base model and
under the tuned ``.cact``, scoring DroidCall-style exact-calls-in-order:
the call names, order, and argument objects must match ``answers`` exactly.
Prints accuracy overall, per language, and per category, plus the exact
rate on ``[]`` negatives and a tool-name-only routing diagnostic, and
writes a JSON report. Refuses to overwrite an existing report unless
``--overwrite`` is passed.

``cactus`` is imported lazily so the pure scoring helpers stay testable
without the package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common
import fastpath


def calls_match(expected: list, actual: list) -> bool:
    """Exact ordered match of tool calls (names, order, arguments)."""
    if len(expected) != len(actual):
        return False
    for want, got in zip(expected, actual):
        if want.get("name") != got.get("name"):
            return False
        if (want.get("arguments") or {}) != (got.get("arguments") or {}):
            return False
    return True


def routing_match(expected: list, actual: list) -> bool:
    """Tool-name-only ordered match (routing diagnostic, not the target)."""
    return [call.get("name") for call in expected] == [
        call.get("name") for call in actual
    ]


def summarize(
    results: list[dict], manifest_rows: dict
) -> dict:
    """Aggregate per-case hits into overall/per-language/per-category rates."""
    summary: dict = {
        "total": 0,
        "hits": 0,
        "by_language": {},
        "by_category": {},
        "negatives": 0,
        "negative_hits": 0,
    }

    def bucket(store: dict, key: str) -> dict:
        return store.setdefault(key, {"total": 0, "hits": 0})

    for result in results:
        meta = manifest_rows[result["id"]]
        hit = result["hit"]
        summary["total"] += 1
        summary["hits"] += hit
        lang = bucket(summary["by_language"], meta["lang"])
        lang["total"] += 1
        lang["hits"] += hit
        cat = bucket(summary["by_category"], meta["category"])
        cat["total"] += 1
        cat["hits"] += hit
        if result["expected"] == []:
            summary["negatives"] += 1
            summary["negative_hits"] += hit
    return summary


def rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def majority_vote(call_runs: list[list]) -> list:
    """Most common call list across repeated runs; ties go to the first.

    Each run is canonicalized (sorted JSON) so key order never splits a
    majority. Empty input yields no calls.
    """
    if not call_runs:
        return []
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    originals: dict[str, list] = {}
    for index, calls in enumerate(call_runs):
        key = json.dumps(calls, sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = index
            originals[key] = calls
    best = max(counts, key=lambda k: (counts[k], -first_seen[k]))
    return originals[best]


def print_summary(label: str, summary: dict) -> None:
    print(f"--- {label} ---")
    print(
        f"overall: {summary['hits']}/{summary['total']} "
        f"({rate(summary['hits'], summary['total']):.1%})"
    )
    print(
        f"negatives ([]) exact: {summary['negative_hits']}/"
        f"{summary['negatives']} "
        f"({rate(summary['negative_hits'], summary['negatives']):.1%})"
    )
    for scope in ("by_language", "by_category"):
        print(f"{scope}:")
        for key in sorted(summary[scope]):
            cell = summary[scope][key]
            print(
                f"  {key}: {cell['hits']}/{cell['total']} "
                f"({rate(cell['hits'], cell['total']):.1%})"
            )


# Task choice: ~50x the measured ~2.4 s/call on this box; a call slower
# than this is hung, not slow.
CALL_TIMEOUT_S = 120
# Task choice: progress granularity for long local eval runs.
PROGRESS_EVERY = 25


def strip_required_keys(node):
    """Deep-copy a tools payload minus every `required` key (engine quirk)."""
    if isinstance(node, dict):
        return {
            key: strip_required_keys(value)
            for key, value in node.items()
            if key != "required"
        }
    if isinstance(node, list):
        return [strip_required_keys(item) for item in node]
    return node


def load_prime(config: dict, task_dir) -> str | None:
    """System-prime text from the task's optional eval prime_file."""
    name = config.get("eval", {}).get("prime_file")
    if not name:
        return None
    path = _common.task_relative_path(task_dir, name)
    if not path.is_file():
        _common.die(f"eval prime_file not found: {path}")
    return path.read_text(encoding="utf-8").strip() or None


def complete_with_timeout(pool, agent, query: str):
    """One complete() call: (function_calls, timed_out)."""
    from concurrent.futures import TimeoutError

    future = pool.submit(agent.complete, query)
    try:
        return future.result(timeout=CALL_TIMEOUT_S), False
    except TimeoutError:
        return [], True


def run_agent(
    tools: list,
    weights: str | None,
    queries: list[str],
    label: str,
    votes: int = 1,
    system: str | None = None,
) -> list:
    """Run complete() over queries, returning [(function_calls, timeout)].

    Resets the agent between independent queries (without reset every
    turn stays in the conversation, slowing and skewing later queries),
    prints flushed progress, and records hung calls as timeouts instead
    of hanging forever. With votes > 1 each query runs that many times
    (resetting between votes so draws stay independent) and the majority
    call list wins; timed-out votes are excluded unless all time out.
    An optional system string primes every conversation identically.
    """
    import cactus
    from concurrent.futures import ThreadPoolExecutor

    kwargs: dict = {"tools": tools}
    if weights is not None:
        kwargs["weights"] = weights
    if system:
        kwargs["system"] = system
    results = []
    with cactus.Cactus(**kwargs) as agent:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for index, query in enumerate(queries):
                completed = []
                for _ in range(max(1, votes)):
                    response, timed_out = complete_with_timeout(
                        pool, agent, query
                    )
                    if not timed_out:
                        completed.append(
                            response.get("function_calls", [])
                        )
                    agent.reset()
                if completed:
                    results.append((majority_vote(completed), False))
                else:
                    results.append(([], True))
                done = index + 1
                if done % PROGRESS_EVERY == 0 or done == len(queries):
                    print(f"[{label}] {done}/{len(queries)}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = _common.add_task_argument(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N eval cases (smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing report instead of refusing.",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=1,
        help="Complete() calls per query; the majority call list wins.",
    )
    parser.add_argument(
        "--ablate",
        default="",
        help="Comma-separated harness layers to disable: prime,fastpath.",
    )
    args = parser.parse_args(argv)
    config, task_dir = _common.load_task_config(args.task)
    out_dir = _common.task_relative_path(task_dir, config["task"]["out_dir"])
    _common.check_no_overwrite(
        [out_dir / config["eval"]["report_file"]], args.overwrite
    )
    data = config["data"]
    tools = _common.read_json(
        _common.task_relative_path(task_dir, config["task"]["tools"])
    )
    # The Needle engine's call validator chokes on a non-empty `required`
    # key and suppresses every call using that tool (round 7b scored 64
    # before this strip, all on `suppressed_calls`). The data contract
    # still requires it (validate_data.py enforces); the runtime simply
    # cannot see it.
    tools = strip_required_keys(tools)
    eval_rows = _common.read_jsonl(out_dir / data["eval_file"])
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]
    with open(out_dir / data["manifest_file"], encoding="utf-8") as handle:
        manifest = json.load(handle)
    weights_path = out_dir / config["build"]["weights_file"]
    if not weights_path.is_file():
        _common.die(f"tuned weights not found: {weights_path}")

    ablated = {
        part.strip() for part in args.ablate.split(",") if part.strip()
    }
    prime = (
        load_prime(config, task_dir) if "prime" not in ablated else None
    )
    rules = None
    fastpath_name = config.get("eval", {}).get("fastpath_file")
    if fastpath_name and "fastpath" not in ablated:
        rules = fastpath.load_rules(
            _common.task_relative_path(task_dir, fastpath_name)
        )
    queries = [row["query"] for row in eval_rows]
    ruled: list = [
        fastpath.apply_rules(rules, query) if rules else None
        for query in queries
    ]
    need = [index for index, hit in enumerate(ruled) if hit is None]
    needed = [queries[index] for index in need]
    base_sub = (
        run_agent(tools, None, needed, "base",
                  votes=args.votes, system=prime)
        if needed else []
    )
    tuned_sub = (
        run_agent(tools, str(weights_path), needed, "tuned",
                  votes=args.votes, system=prime)
        if needed else []
    )
    base_results: list = []
    tuned_results: list = []
    sub = 0
    for index in range(len(queries)):
        if ruled[index] is not None:
            base_results.append((ruled[index], False))
            tuned_results.append((ruled[index], False))
        else:
            base_results.append(base_sub[sub])
            tuned_results.append(tuned_sub[sub])
            sub += 1
    n_ruled = len(queries) - len(need)
    if rules:
        print(f"fastpath ruled {n_ruled}/{len(queries)} queries")

    report = {
        "cases": [], "base": {}, "tuned": {},
        "base_routing": {}, "tuned_routing": {},
        "harness": {
            "votes": args.votes,
            "prime": bool(prime),
            "fastpath": bool(rules),
            "fastpath_ruled": n_ruled,
        },
    }
    for row, (base, base_timeout), (tuned, tuned_timeout) in zip(
        eval_rows, base_results, tuned_results
    ):
        report["cases"].append(
            {
                "id": row["id"],
                "expected": row["answers"],
                "base": base,
                "tuned": tuned,
                "base_timeout": base_timeout,
                "tuned_timeout": tuned_timeout,
                "base_hit": (not base_timeout)
                and calls_match(row["answers"], base),
                "tuned_hit": (not tuned_timeout)
                and calls_match(row["answers"], tuned),
                "base_routing_hit": (not base_timeout)
                and routing_match(row["answers"], base),
                "tuned_routing_hit": (not tuned_timeout)
                and routing_match(row["answers"], tuned),
            }
        )
    report["base"] = summarize(
        [
            {
                "id": case["id"],
                "hit": case["base_hit"],
                "expected": case["expected"],
            }
            for case in report["cases"]
        ],
        manifest["rows"],
    )
    report["tuned"] = summarize(
        [
            {
                "id": case["id"],
                "hit": case["tuned_hit"],
                "expected": case["expected"],
            }
            for case in report["cases"]
        ],
        manifest["rows"],
    )
    report["base_routing"] = summarize(
        [
            {
                "id": case["id"],
                "hit": case["base_routing_hit"],
                "expected": case["expected"],
            }
            for case in report["cases"]
        ],
        manifest["rows"],
    )
    report["tuned_routing"] = summarize(
        [
            {
                "id": case["id"],
                "hit": case["tuned_routing_hit"],
                "expected": case["expected"],
            }
            for case in report["cases"]
        ],
        manifest["rows"],
    )
    report_path = out_dir / config["eval"]["report_file"]
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print_summary("base", report["base"])
    print_summary("tuned", report["tuned"])
    print_summary("base routing", report["base_routing"])
    print_summary("tuned routing", report["tuned_routing"])
    timeouts = sum(
        case["base_timeout"] + case["tuned_timeout"]
        for case in report["cases"]
    )
    print(f"timeouts: {timeouts}")
    print(f"wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
