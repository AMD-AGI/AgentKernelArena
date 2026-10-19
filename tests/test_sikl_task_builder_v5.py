"""SIKL v5 checkout input: inspection, materialization, runner verdicts and gate states; no GPU needed."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from agents.sikl_task_builder.bundle import ImportProblem
from agents.sikl_task_builder.config import RepoConfig, load_config
from agents.sikl_task_builder.sikl_repo import SiklRepo
from agents.sikl_task_builder.validation import GATE_ACTIONS, gate_outcome
from src.task_spec import load_task_spec

TEMPLATES = Path(__file__).resolve().parents[1] / "agents/sikl_task_builder/templates/sikl_v5"
SCHEMA = "sikl/ops/gemm/impls/flydsl/gemm_n32_k64/0123456789abcdef.json"
KERNEL = '"""Tiny kernel."""\n\n\ndef run(*, a, b, out):\n    pass\n'


def schema(**overrides):
    value = {
        "schema_version": 5, "name": "gemm_n32_k64", "contract_id": "gemm@1.0", "description": "d",
        "op_type": "gemm", "axes": {"m": {"type": "var"}}, "attrs": {}, "inputs": {}, "outputs": {},
        "workloads": [{"axes": {"m": 1}, "accuracy": "passed", "speedup": 1.1}, {"axes": {"m": 8}}],
        "layout_variants": {}, "compare": {},
        "kernel": {"language": "flydsl", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
                   "entry_point": "kernel.py::run", "sources": [{"path": "kernel.py", "content": KERNEL}]},
    }
    value.update(overrides)
    return value


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def commit(root, files):
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    return git(root, "rev-parse", "HEAD").strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "SIKL"
    root.mkdir()
    git(root, "init", "-q")
    commit(root, {SCHEMA: schema(), "sikl/ops/gemm/contract.py": "x = 1\n"})
    return root


def test_inspect_reads_committed_schemas_only(checkout):
    (checkout / SCHEMA).write_text("not json")
    extra = checkout / "sikl/ops/gemm/impls/flydsl/gemm_n32_k64/fedcba9876543210.json"
    extra.write_text(json.dumps(schema()))
    repo = SiklRepo.inspect(checkout, [])
    assert [s.task_id for s in repo.schemas] == ["gemm_n32_k64/0123456789abcdef"]
    assert repo.schemas[0].cases == ["m=1", "m=8"]
    assert repo.schemas[0].supports("gfx950", "MI355X")
    assert not repo.schemas[0].supports("gfx942", "MI300X")


@pytest.mark.parametrize("bad, code", [
    ({"schema_version": 4}, "unsupported_schema_version"),
    ({"name": "other"}, "invalid_schema"),
    ({"workloads": [{"axes": {"m": 1}}, {"axes": {"m": 1}}]}, "invalid_schema"),
    ({"kernel": {**schema()["kernel"], "entry_point": "missing.py::run"}}, "invalid_schema"),
    ({"kernel": {**schema()["kernel"], "sources": [{"path": "b.py", "content": "x"},
                                                   {"path": "a.py", "content": "x"}]}}, "invalid_schema"),
])
def test_inspect_rejects_malformed_schemas(checkout, bad, code):
    commit(checkout, {SCHEMA: schema(**bad)})
    with pytest.raises(ImportProblem) as error:
        SiklRepo.inspect(checkout, [])
    assert error.value.code == code


def test_inspect_rejects_unknown_selectors(checkout):
    assert SiklRepo.inspect(checkout, ["gemm_n32_k64"]).schemas
    with pytest.raises(ValueError, match="Unknown task selectors"):
        SiklRepo.inspect(checkout, ["gemm_n64_k64"])


def test_materialized_task_is_a_valid_arena_task(checkout, tmp_path):
    repo = SiklRepo.inspect(checkout, [])
    task = tmp_path / "task"
    repo.schemas[0].materialize(task, repo.commit, "gfx950", "SIKL-v5/gemm_n32_k64/0123456789abcdef", 600)
    spec = load_task_spec(task / "config.yaml", task_id="SIKL-v5/gemm_n32_k64/0123456789abcdef")
    assert spec.baseline.source_files == ("start.json",)
    assert [e.path for e in spec.candidate.editable] == ["source/kernel.py"]
    assert yaml.safe_load((task / "config.yaml").read_text())["platform_support"]["required_arch"] == "gfx950"
    assert (task / "start.json").read_bytes() == git(checkout, "show", f"HEAD:{SCHEMA}").encode()
    assert (task / "source/kernel.py").read_text() == KERNEL
    provenance = json.loads((task / "scripts/provenance.json").read_text())
    assert (provenance["sikl_commit"], provenance["schema_path"]) == (repo.commit, SCHEMA)


def test_v5_config_rejects_legacy_keys(checkout, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"input_dir": str(checkout), "tasks": []}))
    assert isinstance(load_config(config), RepoConfig)
    config.write_text(yaml.safe_dump({"input_dir": str(checkout), "policy": {}}))
    with pytest.raises(ImportProblem) as error:
        load_config(config)
    assert error.value.code == "config_not_v5"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("sikl_v5_task_runner", TEMPLATES / "task_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(findings=(), records=(), status="passed"):
    return {"version": 2, "stages": [
        {"stage_name": "audit", "status": "passed", "detail": "", "findings": []},
        {"stage_name": "bench", "status": status, "detail": "bench failed", "findings": list(findings),
         "workload_records": list(records)}]}


def finding(code, axes=None, detail="d"):
    return {"code": code, "axes": axes, "detail": detail}


def timing(*ms):
    return {"versus": "baseline", "threshold": "faster", "ratio_lower_bound": 1.0, "candidate_milliseconds": list(ms)}


START = schema()


def test_slow_but_correct_candidate_passes_correctness(runner):
    status, cases, _ = runner.judge("correctness", START, report([finding("not_faster", {"m": 8})], status="failed"), 1)
    assert status == "PASS" and [c["status"] for c in cases] == ["PASS", "PASS"]


def test_wrong_workload_fails_only_that_case(runner):
    status, cases, reason = runner.judge("correctness", START,
                                         report([finding("accuracy_failed", {"m": 8}, "max error 3")], status="failed"), 1)
    assert status == "FAIL" and [c["status"] for c in cases] == ["PASS", "FAIL"]
    assert "m=8" in reason and "max error 3" in reason


def test_finding_without_axes_fails_every_case(runner):
    status, cases, _ = runner.judge("correctness", START, report([finding("source_rejected", None)], status="failed"), 1)
    assert status == "FAIL" and {c["status"] for c in cases} == {"FAIL"}


def test_aborted_validation_fails_correctness_and_names_its_findings(runner):
    crash = [finding("platform_failure", {"m": m}, "RuntimeError: No CUDA GPUs are available") for m in (1, 8)]
    status, cases, reason = runner.judge("correctness", START, report(crash, status="aborted"), 3)
    assert status == "FAIL" and {c["status"] for c in cases} == {"FAIL"} and "aborted" in reason
    assert reason.count("No CUDA GPUs are available") == 1


@pytest.mark.parametrize("code, detail, status", [
    ("build_failed", "x", "FAIL"), ("source_rejected", "SA0 too deep", "FAIL"),
    ("source_rejected", "SA7 mutates input", "PASS"), ("accuracy_failed", "x", "PASS"),
])
def test_compile_fails_only_on_build_problems(runner, code, detail, status):
    assert runner.judge("compile", START, report([finding(code, None, detail)], status="failed"), 1)[0] == status


def test_performance_reports_median_baseline_relative_time(runner):
    records = [{"axes": {"m": 1}, "timings": [timing(3.0, 1.0, 2.0)]},
               {"axes": {"m": 8}, "timings": [timing(5.0, 4.0)]}]
    status, cases, _ = runner.judge("performance", START, report(records=records), 0)
    assert status == "PASS"
    assert [(c["test_case_id"], c["execution_time_ms"]) for c in cases] == [("m=1", 2.0), ("m=8", 4.5)]


def test_performance_fails_when_a_workload_is_untimed(runner):
    records = [{"axes": {"m": 1}, "timings": [timing(1.0, 1.0)]}]
    status, cases, reason = runner.judge("performance", START, report(records=records), 0)
    assert status == "FAIL" and [c["status"] for c in cases] == ["PASS", "FAIL"] and "m=8" in reason


def test_performance_fails_when_correctness_fails(runner):
    records = [{"axes": {"m": w}, "timings": [timing(1.0, 1.0)]} for w in (1, 8)]
    bad = report([finding("accuracy_failed", {"m": 1})], records, status="failed")
    assert runner.judge("performance", START, bad, 1)[0] == "FAIL"


def test_unknown_report_version_is_refused(runner):
    with pytest.raises(runner.Failed, match="report version"):
        runner.judge("correctness", START, {**report(), "version": 3}, 0)


def commands(failed=(), error=None):
    return {key: {"ok": key not in failed, "result": {"status": "FAIL" if key in failed else "PASS"},
                  "execution_error": error if key == GATE_ACTIONS[-1] else None} for key in GATE_ACTIONS}


PROCESS = {"timed_out": False, "exit_code": 0}


@pytest.mark.parametrize("process, cmds, initial, codes, state, code", [
    (PROCESS, commands(), {"accepted": True}, [], "installable", None),
    ({"timed_out": True, "exit_code": None}, commands(), None, [], "failed", "gate_timeout"),
    (PROCESS, commands(error="exceeded its 60s action deadline"), {"accepted": True}, [], "failed", "gate_timeout"),
    (PROCESS, commands(["baseline/correctness"]), {"accepted": False}, ["accuracy_failed"],
     "baseline_rejected", "baseline_rejected"),
    (PROCESS, commands(["baseline/performance"]), {"accepted": False}, ["platform_failure"],
     "failed", "execution_error"),
    (PROCESS, commands(["task/validate-task"]), {"accepted": False}, [], "failed", "validate_task_failed"),
    (PROCESS, commands(["candidate/performance"]), {"accepted": False}, [], "failed", "candidate_rejected"),
    ({"timed_out": False, "exit_code": 1}, commands(), {"accepted": False}, [], "failed", "execution_error"),
])
def test_gate_outcome_maps_results_to_run_states(process, cmds, initial, codes, state, code):
    assert gate_outcome(process, cmds, initial, codes) == {"state": state, "code": code}
