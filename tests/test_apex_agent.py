"""CPU adapter regressions; synthetic timing is not GPU qualification."""
from __future__ import annotations

import difflib
import importlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import yaml

from agents.apex import bridge
from agents.apex.bundle import prepare_delivery, read_json
from agents.apex.contract import build_task, canonical, digest, load_context
from agents.apex.launch_agent import initialize_snapshot, install_candidate, load_options
from agents.apex.process import run_worker
from src.harness_guard import snapshot_workspace_harness
from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler
from src.task_execution import run_action
from src.task_spec import TaskSpec


RUNNER = '''
import ast, json, runpy, sys
from pathlib import Path
role, action = ("task", "validate-task") if len(sys.argv) == 2 else sys.argv[1:]
rows = [{"test_case_id": str(n), "shape": [n], "dtype": "int64", "status": "PASS"} for n in [2, 5]]
metadata = {}
status = "PASS"
if action == "validate-task":
    metadata["candidate_state"] = "implemented"
    for row in rows: row["checks"] = ["correctness", "performance"]
elif action == "compile":
    ast.parse(Path("kernel.py").read_text())
    rows = []
elif action == "correctness":
    fn = runpy.run_path("kernel.py")["compute"]
    for row in rows:
        n = row["shape"][0]
        if fn(n) != n * n:
            status = row["status"] = "FAIL"
else:
    # Synthetic protocol fixture only.
    for row in rows: row.update(execution_time_ms=1.0, benchmark_method="cuda_graph")
result = dict(protocol="arena-eval-v1", role=role, action=action, status=status, cases=rows, metadata=metadata)
if status == "FAIL": result["reason"] = "incorrect square"
print("ARENA_EVAL_RESULT=" + json.dumps(result))
sys.exit(0 if status == "PASS" else 1)
'''


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "kernel.py").write_text("def compute(x):\n    return x * x\n\nPROTECTED = 7\n")
    (root / "runner.py").write_text(RUNNER)
    config = {"schema_version": 2, "candidate": {
        "language": "triton", "initial_state": "implemented",
        "editable": [{"path": "kernel.py", "scope": "symbols", "symbols": ["compute"]}],
        "entrypoints": [{"file": "kernel.py", "kind": "function", "symbol": "compute"}]},
        "baseline": {"kind": "initial_candidate"},
        "evaluation": {"runner": [sys.executable, "runner.py"], "timeout_s": 5}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    spec = TaskSpec.from_mapping(config, task_id="arbitrary/task")
    manifest = run_action(spec, root, role="task", action="validate-task", phase="task_validation").result.to_mapping()
    context = dict(version=1, task_id=spec.task_id, task_config=spec.to_mapping(),
                   workspace=str(root), baseline_workspace=str(tmp_path / "baseline"), manifest=manifest)
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context))
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(path))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_ARCH", "gfx950")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = artifacts / "input"
    shutil.copytree(root, source)
    initialize_snapshot(source)
    job_path = artifacts / "job.json"
    results = artifacts / "results"
    results.mkdir()
    task = build_task(spec, source, results, job_path, load_options({}), "gfx950")
    job = dict(context=context, task=task, source=str(source), result=str(results / "result.json"))
    job_path.write_text(json.dumps(job))
    return root, source, spec, job_path, job


def bundle(job, before: bytes, after: bytes, *, target="kernel.py") -> dict:
    root = Path(job["task"]["results_dir"]) / "bundle"
    root.mkdir()
    patch = "".join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                         fromfile="a/" + target, tofile="b/" + target)).encode()
    (root / "change.patch").write_bytes(patch)
    manifest = dict(schema_version=1, task_id=job["task"]["task_id"],
                    baseline={"file_hashes": {"kernel.py": digest(before)}},
                    changed_files=["kernel.py"], candidate_file_hashes={"kernel.py": digest(after)},
                    patches=[{"path": "change.patch", "sha256": digest(patch)}],
                    delivery={"mode": "bundle", "applied": False})
    (root / "bundle.json").write_bytes(canonical(manifest))
    return dict(schema_version=1, task_id=job["task"]["task_id"], status="candidate_ready",
                reason_code="candidate_deferred_to_external_evaluator", applied=False,
                external_verification_required=True, changed_files=["kernel.py"],
                bundle_path=str(root), bundle_digest=digest(canonical(manifest) + patch))


def deliver(case, result):
    root, source, spec, _, job = case
    return prepare_delivery(result, task=job["task"], original={"kernel.py": (root / "kernel.py").read_bytes()},
                            artifacts=Path(job["task"]["results_dir"]), source=source,
                            harness=snapshot_workspace_harness(root, task_spec=spec))


def test_registration():
    agent = AgentType.from_string("apex")
    assert callable(load_agent_launcher(agent, logging.getLogger()))
    assert load_post_processing_handler(agent, logging.getLogger()).__name__ == "general_post_processing"


def test_contract_and_bridge(case):
    root, source, spec, path, job = case
    assert load_context(root, root / "config.yaml")[0].to_mapping() == spec.to_mapping()
    assert job["task"]["commands"]["compile"]["argv"][-1] == "compile"
    assert job["task"]["recipe"]["provenance"] == "external_evaluator"
    for action in ("compile", "correctness", "performance"):
        bridge.run(path, action, source)
    (source / "kernel.py").write_text("def compute(x):\n    return x\n\nPROTECTED = 7\n")
    with pytest.raises(RuntimeError, match="correctness failed"):
        bridge.run(path, "correctness", source)


@pytest.mark.parametrize("field,value", [("language", "hip"), ("initial_state", "unimplemented"),
                                        ("initial_language", "python")])
def test_unsupported_contract_fails_before_launch(case, monkeypatch, field, value):
    root, _, _, _, job = case
    raw = job["context"]
    raw["task_config"]["candidate"][field] = value
    if value == "unimplemented":
        raw["task_config"]["candidate"].pop("initial_language")
        raw["task_config"]["baseline"] = {"kind": "provided", "source_files": ["runner.py"]}
    Path(os.environ["ARENA_TASK_CONTEXT"]).write_text(json.dumps(raw))
    (root / "config.yaml").write_text(yaml.safe_dump(raw["task_config"]))
    with pytest.raises(ValueError, match="implemented Python/Triton"):
        load_context(root, root / "config.yaml")


def test_bundle_delivery_and_scope(case):
    root, _, _, _, job = case
    before = (root / "kernel.py").read_bytes()
    after = before.replace(b"x * x", b"x ** 2")
    result = bundle(job, before, after)
    candidate = deliver(case, result)
    assert (root / "kernel.py").read_bytes() == before
    install_candidate(root, {"kernel.py": before}, candidate)
    assert (root / "kernel.py").read_bytes() == after
    assert not (root / "task_result.yaml").exists()


@pytest.mark.parametrize("fault", ["harness", "target", "digest", "baseline", "traversal", "symlink", "extra"])
def test_rejected_bundle_never_changes_workspace(case, fault):
    root, _, _, _, job = case
    before = (root / "kernel.py").read_bytes()
    after = before.replace(b"x * x", b"x ** 2")
    if fault == "harness": after = after.replace(b"PROTECTED = 7", b"PROTECTED = 8")
    result = bundle(job, before, after, target="runner.py" if fault == "target" else "kernel.py")
    path = Path(result["bundle_path"])
    if fault == "digest": result["bundle_digest"] = "0" * 64
    if fault == "baseline":
        manifest = json.loads((path / "bundle.json").read_text())
        manifest["baseline"]["file_hashes"]["kernel.py"] = "0" * 64
        (path / "bundle.json").write_text(json.dumps(manifest))
    if fault == "traversal": result["bundle_path"] = str(path / "../bundle")
    if fault == "symlink":
        (path / "alias").symlink_to(path, target_is_directory=True)
        result["bundle_path"] = str(path / "alias")
    if fault == "extra": (path / "extra").write_text("unrequested")
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        deliver(case, result)
    assert (root / "kernel.py").read_bytes() == before


def test_launcher_handoff(case, monkeypatch):
    launcher = importlib.import_module("agents.apex.launch_agent")
    root, _, _, _, _ = case
    monkeypatch.setattr(launcher, "runtime_environment", lambda: os.environ.copy())
    def worker(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text())
        before = (root / "kernel.py").read_bytes()
        result = bundle(job, before, before.replace(b"x * x", b"x ** 2"))
        Path(job["result"]).write_text(json.dumps(result))
        return 0
    monkeypatch.setattr(launcher, "run_worker", worker)
    assert "1 source files" in launcher.launch_agent({}, str(root / "config.yaml"), str(root))
    assert b"x ** 2" in (root / "kernel.py").read_bytes()


@pytest.mark.parametrize("status", ["timeout", "budget_exhausted", "verification_failed"])
def test_failed_result_cannot_deliver(case, status):
    root, _, _, _, job = case
    before = (root / "kernel.py").read_bytes()
    result = bundle(job, before, before.replace(b"x * x", b"x ** 2"))
    result["status"] = status
    with pytest.raises(ValueError, match="did not deliver"):
        deliver(case, result)


def test_process_deadline_and_output_bound(tmp_path):
    for program, timeout, exception in [
        ("import time; time.sleep(20)", 0.2, TimeoutError),
        ("print('x' * (5 * 1024**2))", 5, RuntimeError),
    ]:
        with pytest.raises(exception):
            run_worker([sys.executable, "-c", program], cwd=tmp_path, env=os.environ.copy(),
                       deadline=time.monotonic() + timeout, log=tmp_path / "output.log")
        assert (tmp_path / "output.log").stat().st_size <= 4 * 1024**2


def test_json_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"status": "failed", "status": "candidate_ready"}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)


def test_no_gain_keeps_original_and_rejects_attached_bundle(case):
    root, _, _, _, job = case
    result = dict(schema_version=1, task_id=job["task"]["task_id"], status="no_gain",
                  applied=False, external_verification_required=True, changed_files=[],
                  bundle_path=None, bundle_digest=None)
    before = (root / "kernel.py").read_bytes()
    assert deliver(case, result) == {}
    result["bundle_path"] = "unexpected"
    with pytest.raises(ValueError, match="no_gain"):
        deliver(case, result)
    assert (root / "kernel.py").read_bytes() == before


def test_input_drift_and_concurrent_delivery_are_rejected(case):
    root, source, _, _, job = case
    before = (root / "kernel.py").read_bytes()
    result = bundle(job, before, before.replace(b"x * x", b"x ** 2"))
    (source / "kernel.py").write_text("changed input")
    with pytest.raises(ValueError, match="input source changed"):
        deliver(case, result)
    (root / "kernel.py").write_text("concurrent edit")
    with pytest.raises(ValueError, match="source changed"):
        install_candidate(root, {"kernel.py": before}, {"kernel.py": before})
    assert (root / "kernel.py").read_text() == "concurrent edit"


def test_runtime_rejects_another_branch(tmp_path, monkeypatch):
    from agents.apex import runtime
    monkeypatch.setenv("APEX_ROOT", str(tmp_path))
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: type("Result", (), {"stdout": "wrong-revision"})())
    with pytest.raises(ValueError, match="recovery-integration pin"):
        runtime.runtime_environment()


def test_preflight_uses_ordinary_processes(monkeypatch):
    from agents.apex import runtime
    monkeypatch.setattr(runtime, "runtime_environment", lambda: {})
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda argv, **kw: calls.append(argv))
    runtime.check_runtime(gpu=True)
    assert len(calls) == 1
    assert "probe_assigned_gpus" in calls[0][-1]
    assert "require_pid_namespace=True" not in calls[0][-1]
    assert "OwnershipInspector" not in calls[0][-1]


@pytest.mark.parametrize("gpu", [False, True])
def test_failed_runtime_probe_is_not_accepted(monkeypatch, gpu):
    from agents.apex import runtime
    monkeypatch.setattr(runtime, "runtime_environment", lambda: {})

    def run(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        runtime.check_runtime(gpu=gpu)


@pytest.mark.parametrize("overrides", [{"backend": "unknown"}, {"max_turns": 0},
                                      {"max_iterations": True}, {"unrecognized": 1}])
def test_bad_options_fail_before_runtime(overrides):
    with pytest.raises(ValueError):
        load_options({"agent": {"template": "apex", **overrides}})


def test_pinned_upstream_contract(case):
    if not os.environ.get("APEX_ROOT"):
        pytest.skip("Set APEX_ROOT to exercise the pinned upstream package")
    from agents.apex.runtime import runtime_environment
    _, _, _, job_path, _ = case
    code = '''
import json, sys
from pathlib import Path
from apex.bootstrap import build_application
from apex.intake import TaskSpec
job = json.loads(Path(sys.argv[1]).read_text())
task = TaskSpec.from_mapping(job["task"])
preview = build_application().kernel_optimizer.preview_evaluation_contract(task)
assert preview.draft.repository.resolved
assert preview.draft.task_id == task.task_id
assert preview.draft.recipe_claim["provenance"] == "external_evaluator"
'''
    subprocess.run([sys.executable, "-c", code, str(job_path)], env=runtime_environment(),
                   check=True, timeout=60)
