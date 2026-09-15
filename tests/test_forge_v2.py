"""Real CPU task-command coverage for the provider bridge (no GPU claims)."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.forge import adapter, bridge
from agents.forge.task_context import TaskContext, bounded_spec
from agents.forge.bundles import committed_candidate, copy_workspace, install_candidate
from src.task_protocol import CaseManifest, RESULT_PREFIX, parse_command_result
from src.task_spec import TaskSpec

RUNNER = r'''
import json, sys
from pathlib import Path
role, action = sys.argv[1:]
rows = [{"test_case_id": name, "shape": [2], "dtype": "float32", "status": "PASS"} for name in ("a b", "a_b")]
status = "PASS"
reason = None
if role == "candidate":
    try:
        factor = int(Path("source/kernel.py").read_text().strip())
        helper = int(Path("source/helper.py").read_text().strip())
        if action != "compile" and factor * helper != 6:
            raise ValueError("wrong multiplication")
    except (ValueError, OSError) as exc:
        status, reason = "FAIL", str(exc)
        for row in rows: row["status"] = status
if action == "compile":
    rows = []
if action == "performance" and status == "PASS":
    for row in rows:
        row.update(execution_time_ms=4 if role == "baseline" else 2, benchmark_method="cuda_event_fallback")
if Path("drop_case").exists() and rows:
    rows.pop()
result = dict(protocol="arena-eval-v1", role=role, action=action, status=status, cases=rows)
if reason: result["reason"] = reason
print("ARENA_EVAL_RESULT=" + json.dumps(result))
sys.exit(status != "PASS")
'''


def fixture_task(tmp_path, *, language="flydsl", initial_state="implemented", initial_language=None):
    ws, base = tmp_path / "workspace", tmp_path / "baseline"
    ws.mkdir()
    (ws / "source").mkdir()
    (ws / "source/kernel.py").write_text("2")
    (ws / "source/helper.py").write_text("3")
    (ws / "runner.py").write_text(RUNNER)
    (ws / "README.md").write_text("Multiply input by six using the declared backend.")
    candidate = dict(language=language, initial_state=initial_state,
                     editable=["source/kernel.py", "source/helper.py"],
                     entrypoints=[dict(file="source/kernel.py", kind="function", symbol="not_a_builder")])
    if initial_language:
        candidate["initial_language"] = initial_language
    config = dict(schema_version=2, candidate=candidate,
                  evaluation=dict(runner=[sys.executable, "runner.py"], timeout_s=20))
    (ws / "config.yaml").write_text("schema_version: 2\n")
    copy_workspace(ws, base)
    manifest = dict(protocol="arena-eval-v1", role="task", action="validate-task", status="PASS",
                    cases=[dict(test_case_id=name, shape=[2], dtype="float32", status="PASS",
                                checks=["correctness", "performance"]) for name in ("a b", "a_b")])
    document = dict(version=1, task_id="arbitrary/suite", task_config=config,
                    workspace=str(ws), baseline_workspace=str(base), manifest=manifest)
    path = tmp_path / "context.json"
    path.write_text(json.dumps(document))
    context = TaskContext.load(path, workspace=ws)
    template, engine = tmp_path / "template", tmp_path / "engine"
    copy_workspace(ws, template)
    copy_workspace(ws, engine)
    plan = dict(version=1, context=str(path), template=str(template), engine_root=str(engine),
                workflow="optimize", anchor="source/kernel.py", deadline_unix=time.time()+60,
                result=str(tmp_path / "result.json"), baseline=str(tmp_path / "perf.json"), program=str(engine / "program.md"))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return context, plan, plan_path


def test_context_rejects_missing_wrong_or_workspace_owned_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("ARENA_TASK_CONTEXT", raising=False)
    with pytest.raises(ValueError, match="ARENA_TASK_CONTEXT"):
        TaskContext.load()
    context, _, _ = fixture_task(tmp_path)
    local = context.workspace / "context.json"
    local.write_text(context.path.read_text())
    with pytest.raises(ValueError, match="outside"):
        TaskContext.load(local)
    data = json.loads(context.path.read_text())
    data["baseline_workspace"] = str(context.workspace)
    context.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="independent"):
        TaskContext.load(context.path)


def test_fresh_manifest_required_not_old_workspace_report(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    (context.workspace / "task_result.yaml").write_text("correctness: PASS")
    data = json.loads(context.path.read_text())
    data["manifest"]["cases"][0]["checks"] = ["performance"]
    context.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="correctness coverage"):
        TaskContext.load(context.path)


def test_driver_rebuilds_and_checks_before_timing_and_isolates_roles(tmp_path, monkeypatch, capsys):
    context, plan, path = fixture_task(tmp_path)
    calls = []
    real = bridge.run_action
    def spy(spec, workspace, **kwargs):
        calls.append((kwargs["role"], kwargs["action"], workspace))
        assert workspace != context.workspace and workspace != context.baseline_workspace
        return real(spec, workspace, **kwargs)
    monkeypatch.setattr(bridge, "run_action", spy)
    assert bridge.run(path, plan["engine_root"], ["--bench-mode"]) == 0
    output = capsys.readouterr().out
    assert "case_ms: a%20b 2" in output and "case_ms: a_b 2" in output
    assert [x[1] for x in calls] == ["compile", "correctness", "performance"]
    assert len({x[2] for x in calls}) == 1
    calls.clear()
    assert bridge.run(path, plan["engine_root"], ["--ref-bench-mode"]) == 0
    assert "mean_ms: 4" in capsys.readouterr().out
    assert all(x[0] == "baseline" for x in calls)
    assert (context.baseline_workspace / "source/kernel.py").read_text() == "2"


def test_wrong_candidate_and_incomplete_cases_cannot_emit_timing(tmp_path, capsys):
    _, plan, path = fixture_task(tmp_path)
    (Path(plan["engine_root"]) / "source/helper.py").write_text("100")
    assert bridge.run(path, plan["engine_root"], ["--bench-mode"]) == 1
    output = capsys.readouterr().out
    assert "case_ms:" not in output and "wrong multiplication" in output
    (Path(plan["engine_root"]) / "source/helper.py").write_text("3")
    (Path(plan["template"]) / "drop_case").touch()
    assert bridge.run(path, plan["engine_root"], ["--bench-mode"]) == 1
    assert "manifest mismatch" in capsys.readouterr().out


def test_task_cannot_change_harness_by_editing_engine_copy(tmp_path):
    _, plan, path = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    (engine / "runner.py").write_text('print("allclose: True")')
    (engine / "source/helper.py").write_text("100")
    assert bridge.run(path, engine, []) == 1


def test_generated_driver_uses_own_directory_after_copy(tmp_path):
    context, plan, path = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    (engine / "arena_forge_driver.py").write_text(bridge.render_driver(path, ROOT))
    lane = tmp_path / "lane"
    copy_workspace(engine, lane)
    (engine / "source/helper.py").write_text("100")
    run = subprocess.run([sys.executable, str(lane / "arena_forge_driver.py"), "--bench-mode"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "mean_ms: 2" in run.stdout


def test_profile_reports_public_contract_capability(tmp_path, capsys):
    _, plan, path = fixture_task(tmp_path)
    assert bridge.run(path, plan["engine_root"], ["--profile-run"]) == 2
    assert "unsupported" in capsys.readouterr().out


@pytest.mark.parametrize("selector", ["--case", "--shape", "--profile-case"])
def test_generated_bridge_rejects_case_selection(tmp_path, capsys, selector):
    # Every task uses this generated bridge; engine hints cannot narrow the
    # protected manifest to a convenient case or replace ordinary evaluation.
    _, plan, path = fixture_task(tmp_path)
    with pytest.raises(SystemExit) as error:
        bridge.run(path, plan["engine_root"], [selector, "a_b"])
    assert error.value.code == 2
    assert "case_ms:" not in capsys.readouterr().out


def test_rewrite_binding_keeps_nested_paths_and_all_files(tmp_path, monkeypatch):
    context, plan, path = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    attempt = engine / ".forge_rewrite" / "current"
    copy_workspace(engine, attempt)
    (engine / "source/helper.py").write_text("100")
    plan["workflow"] = "rewrite"
    path.write_text(json.dumps(plan))
    monkeypatch.setenv("KERNELFORGE_REWRITE_CANDIDATE_KERNEL", str(attempt / "source/kernel.py"))
    assert bridge.run(path, engine, ["--bench-mode"]) == 0
    lane = tmp_path / "lane"
    copy_workspace(engine, lane)
    copy_workspace(attempt, lane / ".forge_rewrite/current")
    assert bridge.run(path, lane, []) == 0
    monkeypatch.setenv("KERNELFORGE_REWRITE_CANDIDATE_KERNEL", str(tmp_path / "foreign.py"))
    assert bridge.run(path, engine, []) == 1


def test_bundle_install_preserves_paths_and_rejects_escape(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    (engine / "source/kernel.py").write_text("1")
    (engine / "source/helper.py").write_text("6")
    records = install_candidate(context.spec, engine, context.workspace)
    assert {record["path"] for record in records} == {"source/kernel.py", "source/helper.py"}
    assert not (context.workspace / "kernel.py").exists()
    external = tmp_path / "foreign.py"
    external.write_text("42")
    (engine / "source/helper.py").unlink()
    (engine / "source/helper.py").symlink_to(external)
    before = (context.workspace / "source/kernel.py").read_text()
    with pytest.raises(ValueError):
        install_candidate(context.spec, engine, context.workspace)
    assert (context.workspace / "source/kernel.py").read_text() == before


def test_tree_bundle_propagates_deletions_but_not_harness(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    config = context.spec.to_mapping()
    config["candidate"]["editable"] = [dict(path="source", scope="tree")]
    spec = TaskSpec.from_mapping(config, task_id=context.spec.task_id)
    engine = Path(plan["engine_root"])
    (engine / "source/helper.py").unlink()
    (engine / "source/new.py").write_text("3")
    install_candidate(spec, engine, context.workspace)
    assert not (context.workspace / "source/helper.py").exists()
    assert (context.workspace / "source/new.py").read_text() == "3"


def test_colocated_harness_protected_during_install(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    original = 'def compute():\n    return 1\n\ndef test_compute():\n    assert compute() == 1\n'
    (context.workspace / "source/kernel.py").write_text(original)
    config = context.spec.to_mapping()
    config["candidate"]["editable"][0] = dict(path="source/kernel.py", scope="symbols", symbols=["compute"], allow_new_helpers=True)
    spec = TaskSpec.from_mapping(config, task_id=context.spec.task_id)
    engine = Path(plan["engine_root"])
    (engine / "source/kernel.py").write_text(original.replace("return 1", "return 2").replace("== 1", "== 2"))
    with pytest.raises(ValueError, match="protected statements"):
        install_candidate(spec, engine, context.workspace)
    (engine / "source/kernel.py").write_text(original.replace("return 1", "return 2"))
    install_candidate(spec, engine, context.workspace)


@pytest.mark.parametrize("language", ["hip", "triton", "flydsl"])
def test_auto_routes_verified_existing_targets_to_loop(tmp_path, language):
    context, _, _ = fixture_task(tmp_path, language=language)
    assert adapter.choose_workflow(context, target_verified=True) == "optimize"


@pytest.mark.parametrize("state,initial", [("unimplemented", None), ("implemented", "triton")])
def test_auto_routes_initialization_to_flydsl_rewrite(tmp_path, state, initial):
    context, _, _ = fixture_task(tmp_path, initial_state=state, initial_language=initial)
    assert adapter.choose_workflow(context, target_verified=False) == "rewrite"
    assert adapter.choose_workflow(context, target_verified=True) == "optimize"
    with pytest.raises(ValueError, match="verified"):
        adapter.choose_workflow(context, target_verified=False, requested="optimize")


@pytest.mark.parametrize("language", ["hip", "triton"])
@pytest.mark.parametrize("state,initial", [("unimplemented", None), ("implemented", "pytorch")])
def test_auto_initializes_hip_and_triton_using_forge(tmp_path, language, state, initial):
    context, _, _ = fixture_task(tmp_path, language=language, initial_state=state, initial_language=initial)
    assert adapter.choose_workflow(context, target_verified=False) == "initialize"


def test_action_budget_is_capped_by_remaining_campaign(tmp_path):
    context, _, _ = fixture_task(tmp_path)
    spec = bounded_spec(context.spec, time.time() + 5)
    assert all(0 < action.timeout_s <= 5 for action in spec.actions)
    with pytest.raises(TimeoutError):
        bounded_spec(context.spec, time.time() - 1)


def mock_engine(monkeypatch, *, fail=False, port_ok=True, timeout=False):
    commands = []
    real_run = subprocess.run
    def subprocess_run(command, **kwargs):
        if "--arena-probe" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"backends": ["flydsl", "hip", "triton"], "adapter_api": 1}), stderr="")
        return real_run(command, **kwargs)
    monkeypatch.setattr(adapter.subprocess, "run", subprocess_run)
    monkeypatch.setattr(adapter, "_resolve_gpu_arch", lambda _: "gfx950")
    monkeypatch.setattr(adapter, "_resolve_gpu_type", lambda _: "mi355x")
    def run(command, *, workspace, env, **kwargs):
        commands.append(command)
        plan = json.loads(Path(env["ARENA_FORGE_PLAN"]).read_text())
        engine = Path(workspace)
        result = {"improved": False}
        if "forge-rewrite-by-flydsl" in command:
            root = engine / ".forge_rewrite/current/source"
            root.mkdir(parents=True)
            (root / "kernel.py").write_text("1")
            (root / "helper.py").write_text("6")
            result.update(port_ok=port_ok, temporary_paths=[".forge_rewrite/current"], success=port_ok)
        else:
            (engine / "source/kernel.py").write_text("1")
            (engine / "source/helper.py").write_text("6")
        target = ".forge_rewrite/current" if "forge-rewrite-by-flydsl" in command else "source"
        real_run(["git", "add", "-f", target], cwd=engine, check=True, capture_output=True)
        real_run(["git", "commit", "--quiet", "--allow-empty", "-m", "test selected candidate"], cwd=engine, check=True, capture_output=True)
        commit = real_run(["git", "rev-parse", "HEAD"], cwd=engine, check=True, capture_output=True, text=True).stdout.strip()
        result["best_commit"] = commit
        result["flydsl_best_commit"] = commit
        Path(plan["result"]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=1 if fail else 0), ["engine ran"], [], timeout
    monkeypatch.setattr(adapter, "run_forge_subprocess", run)
    return commands


@pytest.mark.parametrize("state", ["implemented", "unimplemented"])
def test_launcher_delivers_complete_bundle_without_repeating_loop(tmp_path, monkeypatch, state):
    context, _, _ = fixture_task(tmp_path, initial_state=state)
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    commands = mock_engine(monkeypatch)
    output = adapter.launch({"agent": {"model": "chosen", "timeout_seconds": 200}}, "ignored.yaml", str(context.workspace))
    assert len(commands) == 1
    command = commands[0]
    assert ("forge-loop" in command) == (state == "implemented")
    assert ("forge-rewrite-by-flydsl" in command) == (state == "unimplemented")
    assert "--deadline-unix" in command
    assert command[command.index("--model") + 1] == "chosen"
    assert (context.workspace / "source/helper.py").read_text() == "6"
    assert (context.baseline_workspace / "source/helper.py").read_text() == "3"
    assert '"arena_verdict": "pending"' in output
    assert not (context.workspace / "forge_driver.py").exists()


@pytest.mark.parametrize("kwargs", [dict(fail=True), dict(timeout=True), dict(port_ok=False)])
def test_launcher_reports_engine_failure_and_preserves_original(tmp_path, monkeypatch, kwargs):
    context, _, _ = fixture_task(tmp_path, initial_state="unimplemented")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    mock_engine(monkeypatch, **kwargs)
    with pytest.raises(adapter.ForgeRunError):
        adapter.launch({}, "ignored.yaml", str(context.workspace))
    assert (context.workspace / "source/helper.py").read_text() == "3"
    statuses = list(tmp_path.glob("workspace-forge-*/arena_forge_status.json"))
    assert len(statuses) == 1
    assert json.loads(statuses[0].read_text())["status"] == "FAILED"


def test_resume_checks_current_bundle_not_config_initial_stub(tmp_path, monkeypatch):
    context, _, _ = fixture_task(tmp_path, initial_state="unimplemented")
    (context.workspace / "source/kernel.py").write_text("1")
    (context.workspace / "source/helper.py").write_text("6")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    commands = mock_engine(monkeypatch)
    adapter.launch({}, "ignored", str(context.workspace))
    assert "forge-loop" in commands[0]


@pytest.mark.parametrize("initial_state", ["implemented", "unimplemented"])
@pytest.mark.parametrize("outcome", ["normal", "timeout", "exit_error"])
def test_completed_no_keep_search_retains_verified_input_bundle(tmp_path, monkeypatch, initial_state, outcome):
    context, _, _ = fixture_task(tmp_path, initial_state=initial_state)
    # Also cover resuming a generated implementation: the selected starting
    # candidate can differ from the independently frozen production baseline.
    if initial_state == "unimplemented":
        (context.workspace / "source/kernel.py").write_text("3")
        (context.workspace / "source/helper.py").write_text("2")
    expected = {name: (context.workspace / name).read_bytes()
                for name in ("source/kernel.py", "source/helper.py")}
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    mock_engine(monkeypatch, timeout=outcome == "timeout", fail=outcome == "exit_error")
    run = adapter.run_forge_subprocess

    def no_keep(*args, **kwargs):
        output = run(*args, **kwargs)
        plan = json.loads(Path(kwargs["env"]["ARENA_FORGE_PLAN"]).read_text())
        result_path = Path(plan["result"])
        result = json.loads(result_path.read_text())
        result.update(best_commit="", best_iteration=0, iteration_count=1)
        result_path.write_text(json.dumps(result))
        # This discarded working tree must never become the delivery.
        (Path(kwargs["workspace"]) / "source/helper.py").write_text("999")
        return output

    monkeypatch.setattr(adapter, "run_forge_subprocess", no_keep)
    if outcome == "normal":
        output = adapter.launch({}, "unused", str(context.workspace))
        assert '"delivery_selection": "initial_validated_implementation"' in output
    else:
        with pytest.raises(adapter.ForgeRunError):
            adapter.launch({}, "unused", str(context.workspace))
        status_path, = tmp_path.glob("workspace-forge-*/arena_forge_status.json")
        assert json.loads(status_path.read_text())["status"] == "FAILED"
    assert {name: (context.workspace / name).read_bytes() for name in expected} == expected
    assert (context.baseline_workspace / "source/helper.py").read_text() == "3"


@pytest.mark.parametrize("result", [
    {}, {"best_iteration": 1, "iteration_count": 1},
    {"best_iteration": 0}, {"best_iteration": 0, "iteration_count": True},
])
def test_no_commit_fallback_rejects_incomplete_or_contradictory_result(tmp_path, monkeypatch, result):
    context, _, _ = fixture_task(tmp_path)
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    mock_engine(monkeypatch)
    run = adapter.run_forge_subprocess

    def omit_identity(*args, **kwargs):
        output = run(*args, **kwargs)
        plan = json.loads(Path(kwargs["env"]["ARENA_FORGE_PLAN"]).read_text())
        Path(plan["result"]).write_text(json.dumps(result))
        return output

    monkeypatch.setattr(adapter, "run_forge_subprocess", omit_identity)
    with pytest.raises(adapter.ForgeRunError):
        adapter.launch({}, "unused", str(context.workspace))
    assert (context.workspace / "source/helper.py").read_text() == "3"


def test_broken_changed_candidate_not_reclassified_as_empty(tmp_path, monkeypatch):
    context, _, _ = fixture_task(tmp_path, initial_state="unimplemented")
    (context.workspace / "source/helper.py").write_text("100")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    commands = mock_engine(monkeypatch)
    with pytest.raises(adapter.ForgeRunError, match="wrong multiplication"):
        adapter.launch({}, "ignored", str(context.workspace))
    assert not commands


def classify_fixture_failure(context, *, kind="numerical_mismatch", mixed=False):
    reporting = f'''
if status == "FAIL":
    result["failure_kind"] = {kind!r}
    for row in rows: row["failure_kind"] = {kind!r}
    if {mixed!r} and rows: rows[-1]["failure_kind"] = "runtime_error"
'''
    runner = context.workspace / "runner.py"
    runner.write_text(RUNNER.replace('print("ARENA_EVAL_RESULT="', reporting + '\nprint("ARENA_EVAL_RESULT="'))


@pytest.mark.parametrize("language", ["hip", "triton", "flydsl"])
@pytest.mark.parametrize("initial_state", ["implemented", "unimplemented"])
def test_changed_numerical_candidate_reenters_initialization(tmp_path, monkeypatch, language, initial_state):
    context, _, _ = fixture_task(tmp_path, language=language, initial_state=initial_state)
    (context.workspace / "source/helper.py").write_text("100")
    classify_fixture_failure(context)
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    commands = mock_engine(monkeypatch)
    adapter.launch({}, "unused", str(context.workspace))
    assert len(commands) == 1
    assert ("forge-rewrite-by-flydsl" in commands[0]) == (language == "flydsl")
    assert ("--arena-initialize" in commands[0]) == (language != "flydsl")
    status_path, = tmp_path.glob("workspace-forge-*/arena_forge_status.json")
    status = json.loads(status_path.read_text())
    assert status["candidate_requires_repair"] is True
    failed = status["initial_candidate_failure"]
    assert failed["status"] == "FAIL"
    assert failed["failure_kind"] == "numerical_mismatch"
    assert len(failed["cases"]) == 2
    assert "initial_candidate_commit" not in status
    assert (status_path.parent / "template/source/helper.py").read_text() == "100"
    assert "Repair or reimplement" in (status_path.parent / "engine/arena_program.md").read_text()
    assert (context.workspace / "source/helper.py").read_text() == "6"
    assert (context.baseline_workspace / "source/helper.py").read_text() == "3"


@pytest.mark.parametrize("failure", ["runtime", "mixed", "missing_case", "compile", "crash", "unchanged"])
def test_resume_recovery_does_not_hide_environment_or_evidence_errors(tmp_path, monkeypatch, failure):
    context, _, _ = fixture_task(tmp_path, language="hip")
    (context.workspace / "source/helper.py").write_text("100")
    classify_fixture_failure(context, kind="runtime_error" if failure == "runtime" else "numerical_mismatch",
                             mixed=failure == "mixed")
    if failure == "missing_case":
        (context.workspace / "drop_case").touch()
    elif failure == "compile":
        (context.workspace / "source/kernel.py").write_text("not valid source")
    elif failure == "crash":
        (context.workspace / "runner.py").write_text('raise ModuleNotFoundError("fixture runtime unavailable")')
    elif failure == "unchanged":
        (context.baseline_workspace / "source/helper.py").write_text("100")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    commands = mock_engine(monkeypatch)
    with pytest.raises(adapter.ForgeRunError):
        adapter.launch({}, "unused", str(context.workspace))
    assert not commands
    assert (context.workspace / "source/helper.py").read_text() == "100"


@pytest.mark.parametrize("default_branch", ["main", "master"])
def test_scratch_branch_passes_real_upstream_campaign_preflight(tmp_path, monkeypatch, default_branch):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    # Model the actual Docker runtime without modifying any user's Git config.
    git_config = tmp_path / "global.gitconfig"
    git_config.write_text(f"[init]\n\tdefaultBranch = {default_branch}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    root = tmp_path / "engine"
    root.mkdir()
    (root / "kernel.py").write_text("def add(x):\n    return x\n")
    (root / "driver.py").write_text("# task bridge fixture, not executed\n")
    adapter._initialize_git(root)
    actual = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    assert actual == "codex/arena-forge"
    script = r'''
import sys
from agents.forge.upstream import probe
from kernelforge.loop.campaign_config import create_campaign_config
probe()
create_campaign_config(workspace_dir=sys.argv[1], kernel='kernel.py', driver='driver.py',
    source_files=['kernel.py'], program_md_file=None, target_functions=['add'],
    gpu_target='gfx950', gpu_type='mi355x', kernel_backend='triton', task_type='image_kernel')
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    run = subprocess.run([python, "-c", script, str(root)], cwd=root, env=env,
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


def test_installed_upstream_probe_and_real_rewrite_orchestration(tmp_path):
    """Exercise installed engine dispatch with fake PORT/OPTIMIZE, never an LLM/GPU run."""
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to a Hyperloom[forge] interpreter for the compatibility test")
    context, plan, plan_path = fixture_task(tmp_path, initial_state="unimplemented")
    engine = Path(plan["engine_root"])
    plan.update(workflow="rewrite", deadline_unix=time.time()+3600,
                program=str(engine / "arena_program.md"),
                agent_config={"initialization_budget_fraction": .4})
    plan_path.write_text(json.dumps(plan))
    for file in (engine / "source").glob("*.py"):
        file.unlink()
    (engine / "arena_source_hint.py").write_text("# Provided baseline is measured by the task runner\n")
    (engine / "arena_forge_driver.py").write_text(bridge.render_driver(plan_path, ROOT))
    (engine / "arena_program.md").write_text("PORT instructions\n")
    adapter._initialize_git(engine)
    script = r'''
import asyncio, json, os, subprocess
from pathlib import Path
from agents.forge import upstream, bridge
from kernelforge.config import Config
from kernelforge.rewrite_by_flydsl import runner, port_loop
from kernelforge.rewrite_by_flydsl.port_loop import PortResult
from kernelforge.loop import canonical_correctness
plan = json.loads(Path(os.environ["ARENA_FORGE_PLAN"]).read_text())
root = Path(plan["engine_root"])
assert upstream.probe()["rewrite_target"] == "flydsl"
async def port(spec, driver_path, config, *, stop_at_unix=None, **kwargs):
    # Real seed + preflight have already run. Inspect the adapter's program and
    # materialize two simple files for the CPU-only task evaluator.
    prompt = upstream.program_text(plan, port=True)
    assert stop_at_unix == plan['phase_deadline_unix']
    assert stop_at_unix < plan['deadline_unix'] - 1800
    assert "not_a_builder" in prompt
    assert "build_arbitrary" not in prompt
    candidate = Path(spec.flydsl_kernel)
    candidate.write_text("2")
    (candidate.parent / "helper.py").write_text("3")
    return PortResult(ok=True, attempts=1)
def optimize(spec, driver_path, config, **kwargs):
    root = Path(spec.flydsl_kernel).parent
    (root / "kernel.py").write_text("1")
    (root / "helper.py").write_text("6")
    check = asyncio.run(canonical_correctness.accept_candidate(str(Path(spec.workspace)), timeout_cap_sec=30, candidate_label="cpu-test"))
    assert check.passed, check.output
    subprocess.run(['git', 'add', '--', str(root/'kernel.py'), str(root/'helper.py')], cwd=spec.workspace, check=True)
    subprocess.run(['git', 'commit', '-m', 'CPU fixture candidate'], cwd=spec.workspace, check=True, capture_output=True)
    args = upstream.configure_nested_loop(plan, ["forge-loop", "--source-files", spec.flydsl_kernel])
    assert str(root / "helper.py") in args[args.index("--source-files")+1]
    assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=spec.workspace, text=True) == ''
    assert subprocess.check_output(['git', 'show', '--pretty=format:', '--name-only', 'HEAD'], cwd=spec.workspace, text=True).strip() == 'arena_program.md'
    from kernelforge.loop.campaign_config import create_campaign_config
    create_campaign_config(workspace_dir=spec.workspace, kernel=str(root/'kernel.py'), driver=str(driver_path),
        source_files=[str(root/'kernel.py'), str(root/'helper.py')], program_md_file=plan['program'],
        target_functions=['not_a_builder'], gpu_target='gfx950', gpu_type='mi355x', kernel_backend='flydsl', task_type='image_kernel')
    return {"best_ms":2, "mean_case_speedup":2, "llm_usage_complete":True}
port_loop.run_port_loop = port
upstream.install_hooks(plan)
assert runner.run_port_loop is port_loop.run_port_loop
assert runner.run_port_loop.__wrapped__ is port
runner.run_optimize = optimize
result = runner.run_rewrite(op_name="operator-different-from-function", source_kernel=str(root / "arena_source_hint.py"),
        driver=str(root / "arena_forge_driver.py"), workspace=str(root), experiments_dir=str(root / "forge_experiments"),
        target_functions=[], config=Config(workspace=str(root), gpu_target="gfx950", agent_backend="claude"),
        flydsl_kernel_name="source/kernel.py", prepare_driver=False, rewrite_kb_enabled=False,
        result_json=plan["result"], deadline_unix=plan["deadline_unix"])
assert result["port_ok"], result
assert result["success"], result
assert result["applyback_required"] is False
assert result["builder_symbol"] == "not_a_builder"
assert 'phase_deadline_unix' not in plan
budget = json.loads(Path(plan['result']).with_name('port_budget.json').read_text())
assert budget['status'] == 'PASS' and budget['attempts'] == 1
print("UPSTREAM_ADAPTER_CPU_TEST_PASS")
'''
    env = os.environ.copy()
    env.update(ARENA_FORGE_PLAN=str(plan_path), PYTHONPATH=str(ROOT))
    run = subprocess.run([python, "-c", script], cwd=engine, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout[-6000:] + run.stderr[-6000:]
    assert "UPSTREAM_ADAPTER_CPU_TEST_PASS" in run.stdout


@pytest.mark.parametrize("nested", [{}, {"llm_usage_complete": False},
                                   {"llm_usage_complete": True, "terminated_for_deadline": True},
                                   {"llm_usage_complete": True, "experiment_id": "complete"}])
def test_nested_loop_failure_cannot_be_hidden_by_successful_port(tmp_path, nested):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    _, plan, plan_path = fixture_task(tmp_path, initial_state="unimplemented")
    plan["workflow"] = "rewrite"
    plan_path.write_text(json.dumps(plan))
    script = r'''
import json, os
from pathlib import Path
from agents.forge import upstream
from kernelforge.rewrite_by_flydsl import runner
plan = json.loads(Path(os.environ['ARENA_FORGE_PLAN']).read_text())
value = json.loads(os.environ['NESTED_RESULT'])
runner.run_optimize = lambda *a, **kw: value
upstream.install_hooks(plan)
complete = value.get('llm_usage_complete') is True and not value.get('terminated_for_deadline')
try:
    result = runner.run_optimize()
except RuntimeError as error:
    assert not complete
    assert 'PORT evidence retained' in str(error)
else:
    assert complete and result == value
evidence = json.loads(Path(plan['result']).with_name('nested_loop_status.json').read_text())
assert evidence['status'] == ('COMPLETED' if complete else 'FAILED')
assert evidence['result'] == value
'''
    result = subprocess.run([python, "-c", script], cwd=ROOT,
                            env=dict(os.environ, PYTHONPATH=str(ROOT), ARENA_FORGE_PLAN=str(plan_path),
                                     NESTED_RESULT=json.dumps(nested)),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_absolute_symlinks_rebind_to_snapshot(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    source.mkdir()
    (source / "data.txt").write_text("original")
    (source / "alias.txt").symlink_to(source / "data.txt")
    copy_workspace(source, destination)
    (source / "data.txt").write_text("changed")
    assert (destination / "alias.txt").read_text() == "original"
    assert (destination / "alias.txt").resolve().is_relative_to(destination)


def test_test_named_target_still_checks_colocated_harness_before_execution(tmp_path):
    context, plan, _ = fixture_task(tmp_path, language="triton")
    data = json.loads(context.path.read_text())
    data["task_config"]["candidate"] = {
        "language": "triton",
        "editable": [{"path": "test_impl.py", "scope": "symbols",
                      "symbols": ["compute"], "allow_new_helpers": True}],
    }
    context.path.write_text(json.dumps(data))
    original = "def compute(x):\n    return x\n\ndef test_oracle():\n    assert compute(3) == 6\n"
    template, engine = Path(plan["template"]), Path(plan["engine_root"])
    for root in (template, engine):
        (root / "test_impl.py").write_text(original)
    # Real public task actions pass with a legal symbol edit; filename alone
    # does not make the declared implementation a protected file.
    (engine / "test_impl.py").write_text(original.replace("return x", "return x * 2"))
    assert bridge.execute(plan, engine, role="candidate", action="correctness").passed
    # The same native target exemption must never authorize its test body.
    (engine / "test_impl.py").write_text(original.replace("assert compute(3) == 6", "assert True"))
    with pytest.raises(ValueError, match="protected statements"):
        bridge.execute(plan, engine, role="candidate", action="performance")


def test_nested_session_children_are_reaped_after_timeout(tmp_path):
    """A grandchild calls setsid(), so killing only the parent group is insufficient."""
    script = r'''
import os, subprocess, sys, time
from pathlib import Path
from agents.forge.process_tree import managed_children
with managed_children():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    Path(sys.argv[1]).write_text(str(child.pid))
    time.sleep(120)
'''
    marker = tmp_path / "child.pid"
    process = subprocess.Popen([sys.executable, "-c", script, str(marker)], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic()+5
    try:
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists()
        child_pid = int(marker.read_text())
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert not Path(f"/proc/{child_pid}").exists(), stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_delivery_reads_selected_commit_not_uncommitted_tail(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    adapter._initialize_git(engine)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
    (engine / "source/kernel.py").write_text("100")
    bundle = committed_candidate(context.spec, engine, commit, tmp_path / "selected")
    assert (bundle / "source/kernel.py").read_text() == "2"
    assert (bundle / "source/helper.py").read_text() == "3"
    with pytest.raises(ValueError, match="full selected commit"):
        committed_candidate(context.spec, engine, "--all", tmp_path / "bad")


def test_backend_override_does_not_forward_a_different_providers_default():
    config = adapter._config({"agent": {"agent_backend": "codex"}})
    assert config["model"] is None
    explicit = adapter._config({"agent": {"agent_backend": "codex", "model": "my-model"}})
    assert explicit["model"] == "my-model"


def test_new_nested_tree_helpers_are_visible_to_git_in_rewrite_scratch(tmp_path):
    from agents.forge.bundles import allow_candidate_paths
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    config = context.spec.to_mapping()
    config["candidate"]["editable"] = [{"path": "source", "scope": "tree"}]
    spec = TaskSpec.from_mapping(config, task_id=context.spec.task_id)
    adapter._initialize_git(engine)
    allow_candidate_paths(engine, spec, prefix=".forge_rewrite/current")
    target = engine / ".forge_rewrite/current/source/nested/helper.py"
    target.parent.mkdir(parents=True)
    target.write_text("# new helper")
    result = subprocess.run(["git", "check-ignore", str(target)], cwd=engine, capture_output=True)
    assert result.returncode == 1
    subprocess.run(["git", "add", "--", str(target)], cwd=engine, check=True, capture_output=True)
    listed = subprocess.check_output(["git", "ls-files", "--cached"], cwd=engine, text=True)
    assert ".forge_rewrite/current/source/nested/helper.py" in listed


def test_internal_candidate_symlink_cannot_change_declared_delivery_path(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    (engine / "source/real.py").write_text("2")
    (engine / "source/kernel.py").unlink()
    (engine / "source/kernel.py").symlink_to("real.py")
    with pytest.raises(ValueError, match="Candidate symlink"):
        install_candidate(context.spec, engine, context.workspace)
