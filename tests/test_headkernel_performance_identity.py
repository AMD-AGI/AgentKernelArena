"""Duplicate case identity must invalidate the scoped report without fallback."""
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest import mock

import pytest
import yaml


TASKS = Path(__file__).resolve().parents[1] / "tasks/headkernel"
RUNNERS = [path for path in sorted(TASKS.glob("*/scripts/task_runner.py"))
           if not path.parent.parent.name.startswith("qwen")
           and (path.parent.parent / "config.yaml").is_file()
           and not yaml.safe_load((path.parent.parent / "config.yaml").read_text()).get("trusted_evaluation")
           and (path.parent.parent / "ut/meta.json").is_file()
           and (json.loads((path.parent.parent / "ut/meta.json").read_text())
                .get("candidate_bind") or {}).get("file")]


def load_runner(path):
    spec = importlib.util.spec_from_file_location("identity_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_report(ids):
    return {"timer": "cuda_event", "warmup": 10, "iters": 100,
            "cases": [{"sig": name, "mean_ms": 1.0 + i, "median_ms": 1.0 + i,
                       "min_ms": 0.5 + i, "params": {"shape": [i + 1, 128]}}
                      for i, name in enumerate(ids)]}


def test_legacy_protocol_is_only_applied_to_legacy_bound_non_qwen_runners():
    for path in RUNNERS:
        assert not path.parent.parent.name.startswith("qwen")
        assert not yaml.safe_load((path.parent.parent / "config.yaml").read_text()).get("trusted_evaluation")


@pytest.mark.parametrize("path", RUNNERS, ids=lambda p: p.parent.parent.name)
def test_duplicate_ids_reject_all_rows_even_when_shapes_differ(path):
    runner = load_runner(path)
    with pytest.raises(ValueError, match="duplicate case ID"):
        runner._benchmark_cases(raw_report(["first", "truncated|decode", "truncated|decode"]))


@pytest.mark.parametrize("path", RUNNERS, ids=lambda p: p.parent.parent.name)
def test_full_distinct_ids_are_preserved_without_truncation(path):
    runner = load_runner(path)
    prefix = "same-prefix-" * 8
    ids = [prefix + "case-one", prefix + "case-two"]
    cases = runner._benchmark_cases(raw_report(ids))
    assert [case["test_case_id"] for case in cases] == ids
    assert [case["execution_time_ms"] for case in cases] == [1.0, 2.0]


@pytest.mark.parametrize("path", RUNNERS, ids=lambda p: p.parent.parent.name)
def test_duplicate_report_fails_without_automatic_ut_fallback(path, tmp_path):
    runner = load_runner(path)
    runner.TASK_DIR = str(tmp_path)
    runner.UT_DIR = str(tmp_path / "ut")
    runner.BUILD_DIR = str(tmp_path / "build")
    (tmp_path / "ut").mkdir()
    (tmp_path / "build").mkdir()
    final = tmp_path / "build/performance_report.json"
    final.write_text(json.dumps({"status": "ok", "test_cases": [{"execution_time_ms": 0.001}]}))
    duplicate = "block_size=128|disable_index_value=True|init_blocks=0|k_cach|decode"

    def benchmark(*args, **kwargs):
        (tmp_path / "build/_bench_raw.json").write_text(json.dumps(raw_report([duplicate, duplicate])))
        return subprocess.CompletedProcess(args[0], 0, "", "")

    with (mock.patch.object(runner, "candidate_overlay", return_value=None),
          mock.patch.object(runner.subprocess, "run", side_effect=benchmark),
          mock.patch.object(runner, "run_performance_via_ut") as fallback):
        assert runner.run_performance({}, 10) == []
    fallback.assert_not_called()
    result = json.loads(final.read_text())
    assert result["status"] == "fail"
    assert result["test_cases"] == []
    assert result["fallback_used"] is False
    assert "duplicate case ID" in result["failure_reason"]
