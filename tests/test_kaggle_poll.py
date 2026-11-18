"""Tests for scripts/kaggle_poll.py status parsing and argv."""

import pytest

import kaggle_poll


def test_parse_status_complete():
    assert kaggle_poll.parse_status("KernelWorker COMPLETE") == "complete"
    assert kaggle_poll.parse_status("status: complete") == "complete"


def test_parse_status_running():
    assert kaggle_poll.parse_status("RUNNING") == "running"
    assert kaggle_poll.parse_status("QUEUED") == "running"
    assert kaggle_poll.parse_status("pending") == "running"


def test_parse_status_error():
    assert kaggle_poll.parse_status("ERROR") == "error"
    assert kaggle_poll.parse_status("run FAILED") == "error"
    assert (
        kaggle_poll.parse_status(
            'o/s has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'
        )
        == "error"
    )


def test_parse_status_unknown():
    assert kaggle_poll.parse_status("") == "unknown"
    assert kaggle_poll.parse_status("???") == "unknown"


def test_plan_download_moves_preserves_tree(tmp_path):
    staging = tmp_path / "staging"
    nested = staging / "task" / "out"
    nested.mkdir(parents=True)
    (staging / "top.json").write_text("t", encoding="utf-8")
    (nested / "deep.json").write_text("d", encoding="utf-8")
    out_dir = tmp_path / "out"
    assert kaggle_poll.plan_download_moves(staging, out_dir) == [
        (nested / "deep.json", out_dir / "task" / "out" / "deep.json"),
        (staging / "top.json", out_dir / "top.json"),
    ]


def test_main_download_refuses_overwrite_without_flag(
    tmp_path, monkeypatch
):
    import subprocess
    from pathlib import Path

    task_toml = write_poll_task(tmp_path)
    out_dir = tmp_path / "t" / "out"
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "STATUS.json"
    sentinel.write_text("old", encoding="utf-8")

    def fresh_script():
        return [
            subprocess.CompletedProcess([], 0, "RUNNING", ""),
            subprocess.CompletedProcess([], 0, "COMPLETE", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]

    script = fresh_script()

    def fake_run(argv, timeout=None):
        if "output" in argv:
            target = argv[argv.index("-p") + 1]
            dest = Path(target) / "STATUS.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("new", encoding="utf-8")
        return script.pop(0)

    monkeypatch.setattr(kaggle_poll, "run_capture", fake_run)
    monkeypatch.setattr(kaggle_poll.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        kaggle_poll._common, "kaggle_bin", lambda: "kaggle"
    )
    with pytest.raises(SystemExit):
        kaggle_poll.main(["--task", str(task_toml)])
    assert sentinel.read_text(encoding="utf-8") == "old"
    script[:] = fresh_script()
    assert kaggle_poll.main(
        ["--task", str(task_toml), "--overwrite"]
    ) == 0
    assert sentinel.read_text(encoding="utf-8") == "new"


def test_build_argv(tmp_path):
    assert kaggle_poll.build_status_argv("kaggle", "o/s") == [
        "kaggle", "kernels", "status", "o/s",
    ]
    assert kaggle_poll.build_output_argv("kaggle", "o/s", tmp_path) == [
        "kaggle", "kernels", "output", "o/s",
        "-p", str(tmp_path), "-o", "-q",
    ]
    assert kaggle_poll.build_logs_argv("kaggle", "o/s") == [
        "kaggle", "kernels", "logs", "o/s",
    ]


def test_is_rate_limited():
    assert kaggle_poll.is_rate_limited("429 Client Error: Too Many Requests")
    assert kaggle_poll.is_rate_limited("You have been rate limited.")
    assert not kaggle_poll.is_rate_limited("connection reset by peer")
    assert not kaggle_poll.is_rate_limited("")


def test_run_capture_timeout():
    import subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        kaggle_poll.run_capture(["sleep", "5"], timeout=0.05)


def write_poll_task(tmp_path):
    task_dir = tmp_path / "t"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[task]\nname = \"mini\"\nout_dir = \"out\"\n"
        "[kaggle]\nusername = \"o\"\nkernel = \"s\"\n"
        "poll_interval = 60\npoll_timeout = 3600\n",
        encoding="utf-8",
    )
    return task_dir / "task.toml"


def test_main_backs_off_429_without_budget(tmp_path, monkeypatch, capsys):
    import subprocess

    calls = []
    sleeps = []
    script = [
        subprocess.CompletedProcess([], 1, "", "429 Too Many Requests"),
        subprocess.CompletedProcess([], 1, "", "429 Too Many Requests"),
        subprocess.CompletedProcess([], 0, "RUNNING", ""),
        subprocess.CompletedProcess([], 0, "COMPLETE", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ]

    def fake_run(argv, timeout=None):
        calls.append(argv)
        return script.pop(0)

    monkeypatch.setattr(kaggle_poll, "run_capture", fake_run)
    monkeypatch.setattr(kaggle_poll.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        kaggle_poll._common, "kaggle_bin", lambda: "kaggle"
    )
    assert kaggle_poll.main(["--task", str(write_poll_task(tmp_path))]) == 0
    # Two 429 backoffs (300s), no failure budget consumed, then download.
    assert sleeps == [300, 300, 60]
    assert calls[-1][1:3] == ["kernels", "output"]
    assert "backing off" in capsys.readouterr().out


def test_main_ignores_terminal_before_live(tmp_path, monkeypatch, capsys):
    import subprocess

    script = [
        subprocess.CompletedProcess([], 0, "ERROR", ""),
        subprocess.CompletedProcess([], 0, "RUNNING", ""),
        subprocess.CompletedProcess([], 0, "ERROR", ""),
        subprocess.CompletedProcess([], 0, "logtext", ""),
    ]

    def fake_run(argv, timeout=None):
        return script.pop(0)

    monkeypatch.setattr(kaggle_poll, "run_capture", fake_run)
    monkeypatch.setattr(kaggle_poll.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        kaggle_poll._common, "kaggle_bin", lambda: "kaggle"
    )
    assert kaggle_poll.main(["--task", str(write_poll_task(tmp_path))]) == 1
    out = capsys.readouterr().out
    assert "ignoring terminal state" in out
    assert "logtext" in out


def test_main_failure_budget_still_applies(tmp_path, monkeypatch):
    import subprocess

    def always_fail(argv, timeout=None):
        return subprocess.CompletedProcess([], 1, "", "boom")

    monkeypatch.setattr(kaggle_poll, "run_capture", always_fail)
    monkeypatch.setattr(kaggle_poll.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        kaggle_poll._common, "kaggle_bin", lambda: "kaggle"
    )
    assert kaggle_poll.main(["--task", str(write_poll_task(tmp_path))]) == 1


@pytest.mark.parametrize(
    "text,want",
    [
        ("Completed", "complete"),
        ("QUEUED", "running"),
        ("CANCELLED", "error"),
        ("success", "unknown"),
    ],
)
def test_parse_status_extra_variants(text, want):
    assert kaggle_poll.parse_status(text) == want
