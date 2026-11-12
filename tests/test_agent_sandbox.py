"""Agent sandbox: policy construction, launcher integration and Landlock enforcement.

Enforcement tests run only where the kernel offers Landlock ABI 3 or later; the
rest are pure CPU checks.
"""
import ast
import json
import logging
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import yaml

from src import agent_sandbox as sandbox

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def default_agent_sandbox_mode():
    """Overrides tests/conftest.py: this module tests the real default mode."""


def test_mode_defaults_to_auto_and_rejects_unknown_values():
    assert sandbox.sandbox_mode({}) == "auto"
    assert sandbox.sandbox_mode({"agent_sandbox": None}) == "auto"
    for mode in ("none", "auto", "landlock"):
        assert sandbox.sandbox_mode({"agent_sandbox": mode}) == mode
    for value in ("Landlock", "bubblewrap", True, 1):
        with pytest.raises(ValueError, match="agent_sandbox"):
            sandbox.sandbox_mode({"agent_sandbox": value})


@pytest.fixture
def landlock_host(tmp_path, monkeypatch):
    """A Landlock ABI 6 host whose HOME is outside the checkout."""
    monkeypatch.setattr(sandbox, "landlock_abi", lambda: 6)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def test_auto_confines_supported_kernel_launches_and_skips_the_rest(landlock_host, monkeypatch):
    for agent in sorted(sandbox.SUPPORTED_AGENTS):
        assert sandbox.resolve("auto", agent) == (True, None)
        assert sandbox.resolve("landlock", agent) == (True, None)
    for mode in ("none", "auto", "landlock"):
        assert sandbox.resolve(mode, "qa_auditor") == (False, None)
    assert sandbox.resolve("none", "geak", measurement="serving") == (False, None)
    unsupported = [
        (dict(agent_name="geak"), "not supported"),
        (dict(agent_name="forge"), "not supported"),
        (dict(agent_name="codex", measurement="serving"), "single-GPU kernel tasks only"),
        (dict(agent_name="codex", world_size=2), "uses 2 GPUs"),
        (dict(agent_name="codex", environ={"HOME": "/"}), "contains the repository"),
    ]
    for kwargs, reason in unsupported:
        confine, why = sandbox.resolve("auto", **kwargs)
        assert confine is False and reason in why
        with pytest.raises(ValueError, match=reason):
            sandbox.resolve("landlock", **kwargs)
    monkeypatch.setattr(sandbox, "landlock_abi", lambda: 2)
    confine, why = sandbox.resolve("auto", "codex")
    assert confine is False and "ABI 2" in why
    with pytest.raises(ValueError, match="ABI 2"):
        sandbox.resolve("landlock", "codex")


def _start_run(agent, **settings):
    """Run-start validation of main.py; ``started`` once it passes every check."""
    import argparse
    from contextlib import redirect_stdout
    import io
    from unittest.mock import patch

    from main import _build_context
    config = {"tasks": [], "agent": {"template": agent}, "workspace_directory_prefix": "workspace",
              "target_gpu_model": "MI355X", **settings}
    output = io.StringIO()
    with patch("main._load_config", return_value=config), \
            patch("main._resolve_run", side_effect=AssertionError("started")), redirect_stdout(output):
        try:
            context = _build_context(argparse.Namespace(run_suffix=None, config_name="unused.yaml"),
                                     need_agent_launcher=True)
        except AssertionError:
            context = "started"
    return context, output.getvalue()


def test_run_start_notes_unconfined_auto_runs_and_refuses_strict_ones(landlock_host):
    assert _start_run("codex") == ("started", "")
    context, output = _start_run("geak")
    assert context == "started" and "agents run unconfined" in output and "not supported" in output
    context, output = _start_run("geak", agent_sandbox="landlock")
    assert context is None and "not supported" in output
    assert _start_run("geak", agent_sandbox="none") == ("started", "")


def test_complement_excludes_subtree_and_symlinks_into_it(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "file").write_text("x")
    protected = tmp_path / "outer" / "protected"
    (protected / "inner").mkdir(parents=True)
    (tmp_path / "outer" / "sibling").mkdir()
    (tmp_path / "escape").symlink_to(protected / "inner")
    (tmp_path / "outer" / "loop").symlink_to(tmp_path)
    paths = sandbox._complement(str(tmp_path), [os.path.realpath(protected)])
    real = os.path.realpath(tmp_path)
    assert sorted(paths) == sorted([f"{real}/file", f"{real}/keep", f"{real}/outer/sibling"])
    assert sandbox._complement(str(protected / "inner"), [os.path.realpath(protected)]) == []
    assert sandbox._complement(str(tmp_path / "keep"), [os.path.realpath(protected)]) == [f"{real}/keep"]


def test_rules_grant_only_the_workspace_and_declared_files_inside_the_repository(tmp_path):
    repo = tmp_path / "repo"
    workspace = repo / "workspace_GPU_agent" / "run" / "task_ws"
    state = repo / "workspace_GPU_agent" / "run" / ".task-sessions" / "task_ws"
    baseline = state / "baseline"
    for path in (repo / "src", workspace, baseline, repo / ".aka-pyuserbase"):
        path.mkdir(parents=True)
    (state / "agent_context.json").write_text("{}")
    rules = sandbox.build_rules(workspace=workspace, repo_root=repo, home=str(tmp_path / "home"),
                                read_only=[baseline, state / "agent_context.json", repo / ".aka-pyuserbase",
                                           state / "validation_context.json"])
    real_repo = os.path.realpath(repo)
    inside = {rule["path"]: rule["access"] for rule in rules if sandbox._within(rule["path"], real_repo)}
    assert inside == {os.path.realpath(workspace): "rw", os.path.realpath(baseline): "ro",
                      os.path.realpath(state / "agent_context.json"): "ro",
                      os.path.realpath(repo / ".aka-pyuserbase"): "ro"}
    assert not any(sandbox._within("/dev", rule["path"]) and rule["access"] != "dev" for rule in rules)
    assert {rule["path"] for rule in rules if rule["access"] == "dev"} <= set(sandbox.DEVICE_PATHS)
    assert any(rule["path"] == os.path.realpath("/tmp") and rule["access"] == "rw" for rule in rules) \
        or sandbox._within(real_repo, os.path.realpath("/tmp"))
    with pytest.raises(sandbox.SandboxError, match="HOME"):
        sandbox.build_rules(workspace=workspace, repo_root=repo, home=str(tmp_path))


def test_wrap_command_is_identity_without_a_policy():
    assert sandbox.wrap_command(["agent", "--x"], {}) == ["agent", "--x"]
    wrapped = sandbox.wrap_command(["agent", "--x"], {sandbox.POLICY_ENV: "/state/p.json"})
    assert wrapped == [sys.executable, "-I", str(Path(sandbox.__file__).resolve()), "--policy",
                       "/state/p.json", "--", "agent", "--x"]


def _prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "require_landlock", lambda minimum=sandbox.MIN_ABI: 6)
    repo = tmp_path / "repo"
    workspace, state = repo / "ws", repo / ".task-sessions" / "ws"
    for path in (workspace, state / "baseline"):
        path.mkdir(parents=True)
    secret = repo / "src" / "framework.py"
    secret.parent.mkdir()
    secret.write_text("protected")
    return sandbox.prepare("landlock", workspace=workspace, state_directory=state,
                           baseline_workspace=state / "baseline", environ={"HOME": str(tmp_path / "home")},
                           repo_root=repo, self_test_path=secret)


def test_prepare_is_off_by_default_and_requires_landlock(tmp_path, monkeypatch):
    assert sandbox.prepare("none", workspace=tmp_path, state_directory=tmp_path, baseline_workspace=tmp_path) is None
    monkeypatch.setattr(sandbox, "landlock_abi", lambda: 2)
    with pytest.raises(sandbox.SandboxError, match="ABI 2"):
        sandbox.prepare("landlock", workspace=tmp_path, state_directory=tmp_path, baseline_workspace=tmp_path)


def test_record_is_enforced_only_when_every_launch_attested_the_policy(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch)
    policy = json.loads(prepared.policy_path.read_text())
    assert policy["policy_sha256"] == prepared.policy["policy_sha256"]
    assert prepared.environment == {sandbox.POLICY_ENV: str(prepared.policy_path)}
    assert prepared.record()["status"] == "unattested"
    directory = Path(policy["attestation_directory"])
    good = {"nonce": policy["nonce"], "policy_sha256": policy["policy_sha256"], "self_test": "denied", "abi": 6}
    (directory / f"agent-sandbox-{policy['nonce']}-1.json").write_text(json.dumps(good))
    record = prepared.record()
    assert record["status"] == "enforced" and record["launches"] == 1 and record["abi"] == [6]
    (directory / f"agent-sandbox-{policy['nonce']}-2.json").write_text(json.dumps({**good, "policy_sha256": "x"}))
    assert prepared.record()["status"] == "invalid"


def test_wrapper_attests_then_executes_without_the_policy_variable(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch)
    secret = Path(prepared.policy["self_test_path"])
    calls = {}
    monkeypatch.setattr(sandbox, "_restrict", lambda rules, abi: calls.update(rules=rules) or {
        "abi": abi, "rules_applied": len(rules), "rules_missing": [], "handled_access_fs": 1,
        "signal_scoped": True})
    real_open = os.open
    def guarded_open(path, flags, *args):
        if str(path) == str(secret):
            raise PermissionError(13, "denied by test")
        return real_open(path, flags, *args)
    monkeypatch.setattr(sandbox.os, "open", guarded_open)
    monkeypatch.setattr(sandbox.os, "execve", lambda exe, argv, env: calls.update(exe=exe, argv=argv, env=env))
    monkeypatch.setenv(sandbox.POLICY_ENV, str(prepared.policy_path))
    sandbox._exec(str(prepared.policy_path), [sys.executable, "-c", "pass"])
    assert calls["exe"] == sys.executable and sandbox.POLICY_ENV not in calls["env"]
    assert calls["rules"] == prepared.policy["rules"]
    assert prepared.record()["status"] == "enforced"


def test_wrapper_refuses_a_tampered_policy(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch)
    policy = json.loads(prepared.policy_path.read_text())
    policy["rules"].append({"path": "/", "access": "rw"})
    prepared.policy_path.write_text(json.dumps(policy))
    with pytest.raises(sandbox.SandboxError, match="digest"):
        sandbox._exec(str(prepared.policy_path), [sys.executable])
    assert sandbox.main(["--policy", str(prepared.policy_path), "--", sys.executable]) == 126


@pytest.mark.parametrize("agent", sorted(sandbox.SUPPORTED_AGENTS))
def test_every_supported_launcher_wraps_each_agent_process(agent):
    tree = ast.parse((ROOT / "agents" / agent / "launch_agent.py").read_text())
    popens = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
    assert popens, agent
    for call in popens:
        first = call.args[0]
        assert isinstance(first, ast.Call) and getattr(first.func, "id", None) == "wrap_command", (
            f"{agent}: subprocess.Popen at line {call.lineno} does not use wrap_command")


def _fixture_task(tmp_path):
    from tests.test_task_run_v2 import package
    return package(tmp_path)


def _run(tmp_path, path, launcher, agent="codex", mode="landlock"):
    from src.task_run import run_task_v2
    return run_task_v2(eval_config={"agent": {"template": agent}, "agent_sandbox": mode},
                       agent=SimpleNamespace(value=agent), agent_launcher=launcher,
                       task_name="suite/protocol_fixture", task_config_dir=str(path),
                       run_directory=tmp_path / "run", timestamp="fixture",
                       logger=logging.getLogger("test_agent_sandbox"))


@pytest.fixture
def cpu_runtime(monkeypatch):
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {
        "gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU", "torch": "fixture"})
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)


def test_unsupported_agent_fails_before_any_workspace_is_created(tmp_path, cpu_runtime):
    path = _fixture_task(tmp_path)
    with pytest.raises(ValueError, match="not supported"):
        _run(tmp_path, path, lambda **_: pytest.fail("must not launch"), agent="geak")
    assert not (tmp_path / "run").exists() or not any((tmp_path / "run").glob("*fixture*"))


def test_serving_tasks_are_refused_before_any_workspace_is_created(tmp_path, monkeypatch, cpu_runtime,
                                                                   landlock_host):
    monkeypatch.setattr("src.measurement.measurement_kind", lambda config: "serving")
    path = _fixture_task(tmp_path)
    with pytest.raises(ValueError, match="kernel tasks only"):
        _run(tmp_path, path, lambda **_: pytest.fail("must not launch"))
    assert not (tmp_path / "run").exists() or not any((tmp_path / "run").glob("*fixture*"))


@pytest.mark.parametrize("mode", ["auto", "landlock"])
def test_unattested_launch_is_not_accepted(tmp_path, cpu_runtime, landlock_host, mode):
    path = _fixture_task(tmp_path)
    seen = {}

    def launcher(**kwargs):
        seen["policy"] = os.environ.get(sandbox.POLICY_ENV)
        (Path(kwargs["workspace"]) / "kernel.py").write_text("def compute(): return 1 + 2\n")

    _, workspace = _run(tmp_path, path, launcher, mode=mode)
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert seen["policy"] and Path(seen["policy"]).is_file()
    assert not sandbox._within(os.path.realpath(seen["policy"]), os.path.realpath(workspace))
    assert report["agent_sandbox"]["status"] == "unattested"
    assert report["agent_execution"]["sandbox"]["status"] == "unattested"
    assert report["agent_sandbox"]["mode"] == mode
    assert report["candidate_accepted"] is False


def test_attested_launch_keeps_the_normal_outcome(tmp_path, cpu_runtime, landlock_host):
    path = _fixture_task(tmp_path)

    def launcher(**kwargs):
        policy = json.loads(Path(os.environ[sandbox.POLICY_ENV]).read_text())
        Path(policy["attestation_directory"], f"agent-sandbox-{policy['nonce']}-1.json").write_text(json.dumps({
            "nonce": policy["nonce"], "policy_sha256": policy["policy_sha256"], "self_test": "denied", "abi": 6}))
        (Path(kwargs["workspace"]) / "kernel.py").write_text("def compute(): return 1 + 2\n")

    _, workspace = _run(tmp_path, path, launcher, mode="auto")
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert report["agent_sandbox"]["status"] == "enforced"
    assert report["candidate_accepted"] is True


@pytest.mark.parametrize("agent,world_size,reason", [("geak", 1, "not supported"), ("codex", 4, "uses 4 GPUs")])
def test_auto_runs_unsupported_launches_unconfined_and_records_why(tmp_path, monkeypatch, cpu_runtime,
                                                                   landlock_host, agent, world_size, reason):
    monkeypatch.setattr("src.task_run._world_size", lambda spec: world_size)
    path = _fixture_task(tmp_path)
    seen = {}

    def launcher(**kwargs):
        seen["policy"] = os.environ.get(sandbox.POLICY_ENV)
        (Path(kwargs["workspace"]) / "kernel.py").write_text("def compute(): return 1 + 2\n")

    _, workspace = _run(tmp_path, path, launcher, agent=agent, mode="auto")
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert seen["policy"] is None
    assert report["agent_sandbox"]["status"] == "not_applied"
    assert reason in report["agent_sandbox"]["reason"]
    assert report["agent_execution"]["sandbox"] == report["agent_sandbox"]
    assert report["candidate_accepted"] is True


def test_sandbox_off_leaves_results_unchanged(tmp_path, cpu_runtime):
    path = _fixture_task(tmp_path)
    _, workspace = _run(tmp_path, path, lambda **_: None, mode="none")
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert "agent_sandbox" not in report and "sandbox" not in report["agent_execution"]
    assert report["candidate_accepted"] is True


# -- Enforcement (Linux with Landlock ABI >= 3) ------------------------------------------------

landlock = pytest.mark.skipif(sandbox.landlock_abi() < sandbox.MIN_ABI,
                              reason="needs Linux with Landlock ABI 3 or later")

PROBE = textwrap.dedent("""
    import json, os, subprocess, sys
    workspace, secret, sibling, baseline, scratch = sys.argv[1:6]
    results = {}
    def attempt(name, action):
        try:
            action()
            results[name] = "ok"
        except PermissionError:
            results[name] = "denied"
    attempt("read_workspace", lambda: open(os.path.join(workspace, "kernel.py")).read())
    attempt("write_workspace", lambda: open(os.path.join(workspace, "new.py"), "w").write("x"))
    attempt("read_baseline", lambda: open(os.path.join(baseline, "kernel.py")).read())
    attempt("write_baseline", lambda: open(os.path.join(baseline, "kernel.py"), "a").write("x"))
    attempt("read_framework", lambda: open(secret).read())
    attempt("truncate_framework", lambda: os.truncate(secret, 0))
    attempt("list_repository", lambda: os.listdir(os.path.dirname(os.path.dirname(secret))))
    attempt("read_other_workspace", lambda: open(os.path.join(sibling, "kernel.py")).read())
    attempt("write_tmp", lambda: open(os.path.join(scratch, "probe-%d" % os.getpid()), "w").write("x"))
    attempt("read_system", lambda: open(os.path.realpath(sys.executable), "rb").read(16))
    child = subprocess.run([sys.executable, "-c", "open(%r).read()" % secret], capture_output=True, text=True)
    results["child_read_framework"] = ("ok" if child.returncode == 0 else
                                       "denied" if "PermissionError" in child.stderr else child.stderr[-200:])
    attempt("dev_null", lambda: open("/dev/null", "w").write("x"))
    print(json.dumps(results))
""")


@landlock
def test_landlock_confines_the_agent_process_tree(tmp_path):
    repo = tmp_path / "repo"
    workspace = repo / "workspace_GPU_agent" / "run" / "task_ws"
    sibling = repo / "workspace_GPU_agent" / "run" / "other_ws"
    state = repo / "workspace_GPU_agent" / "run" / ".task-sessions" / "task_ws"
    for path in (workspace, sibling, state / "baseline", repo / "src"):
        path.mkdir(parents=True)
    for path in (workspace, sibling, state / "baseline"):
        (path / "kernel.py").write_text("def compute(): return 3\n")
    secret = repo / "src" / "evaluator.py"
    secret.write_text("protected")
    # Entries of a directory that contains the repository are granted one by one,
    # so the probe must exist before the policy is built.
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)
    scratch = tmp_path / "scratch"  # stands in for /tmp, which contains this test's repository
    scratch.mkdir()
    prepared = sandbox.prepare("landlock", workspace=workspace, state_directory=state,
                               baseline_workspace=state / "baseline", repo_root=repo, self_test_path=secret,
                               environ={"HOME": str(tmp_path / "home"), "PATH": os.environ.get("PATH", "")})
    env = {**os.environ, **prepared.environment}
    command = sandbox.wrap_command([sys.executable, "-I", str(probe), str(workspace), str(secret), str(sibling),
                                    str(state / "baseline"), str(scratch)], env)
    result = subprocess.run(command, env=env, cwd=workspace, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {
        "read_workspace": "ok", "write_workspace": "ok", "read_baseline": "ok", "write_baseline": "denied",
        "read_framework": "denied", "truncate_framework": "denied", "list_repository": "denied",
        "read_other_workspace": "denied", "write_tmp": "ok", "read_system": "ok",
        "child_read_framework": "denied", "dev_null": "ok"}
    assert secret.read_text() == "protected"
    assert prepared.record()["status"] == "enforced"
    attestations = list(state.glob(f"agent-sandbox-{prepared.policy['nonce']}-*.json"))
    assert len(attestations) == 1 and stat.S_IMODE(attestations[0].stat().st_mode) == 0o600
