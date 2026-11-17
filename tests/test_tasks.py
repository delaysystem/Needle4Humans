"""Consistency checks across shipped task directories."""

import json
import tomllib
from pathlib import Path

import kaggle_push

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def shipped_tasks():
    return sorted(
        path for path in TASKS_DIR.glob("*/task.toml")
        if not path.parent.name.startswith("_")
    )


def load_toml(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def test_every_task_has_required_files():
    task_files = shipped_tasks()
    assert {path.parent.name for path in task_files} >= {
        "human_handoff", "moderation",
    }
    for task_file in task_files:
        task = load_toml(task_file)["task"]
        for key in ("tools", "slot_banks", "seed_bank", "eval_bank"):
            assert (task_file.parent / task[key]).is_file(), (
                f"{task_file}: missing {key} file {task[key]}"
            )


def test_kaggle_slugs_distinct_across_tasks():
    seen = {}
    for task_file in shipped_tasks():
        kaggle = load_toml(task_file)["kaggle"]
        for key in (
            "input_dataset", "output_dataset",
            "checkpoints_dataset", "kernel",
        ):
            ref = f"{kaggle['username']}/{kaggle[key]}"
            assert ref not in seen, (
                f"{ref} shared by {seen.get(ref)} and {task_file}"
            )
            seen[ref] = task_file


def test_moderation_renders_gpu_metadata():
    # TPU was tried first (probe v1 starved QUEUED 7+ hours 2026-11-16);
    # training follows the GPU fallback. Synthetic TPU rendering stays
    # covered in test_kaggle_push.py.
    config = load_toml(TASKS_DIR / "moderation" / "task.toml")
    metadata = kaggle_push.render_kernel_metadata(config)
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "enable_tpu" not in metadata


def test_seed_cells_declared_in_task_config():
    for task_file in shipped_tasks():
        config = load_toml(task_file)
        languages = set(config["data"]["languages"])
        categories = set(config["data"]["categories"])
        for bank in ("seed_bank", "eval_bank"):
            text = (task_file.parent / config["task"][bank]).read_text(
                encoding="utf-8"
            )
            for line in text.splitlines():
                if line.strip():
                    seed = json.loads(line)
                    assert seed["lang"] in languages, task_file
                    assert seed["category"] in categories, task_file


def test_fastpath_cases_pass_where_declared():
    import fastpath

    checked = []
    for task_file in shipped_tasks():
        task_dir = task_file.parent
        eval_cfg = load_toml(task_file).get("eval", {})
        spec_name = eval_cfg.get("fastpath_file")
        cases_name = eval_cfg.get("fastpath_cases")
        if not spec_name:
            assert not cases_name, f"{task_file}: cases without a spec"
            continue
        assert cases_name, f"{task_file}: spec without cases"
        rules = fastpath.load_rules(task_dir / spec_name)
        cases = [
            json.loads(line)
            for line in (task_dir / cases_name).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert cases, f"{task_file}: empty cases file"
        failures = fastpath.run_cases(rules, cases)
        assert failures == [], "\n".join(failures)
        checked.append(task_file.parent.name)
    assert checked, "no task declares fastpath_file"
