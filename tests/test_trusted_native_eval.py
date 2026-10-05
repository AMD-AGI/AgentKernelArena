"""Host-side trust-boundary tests; Docker/GPU execution is mocked explicitly."""

import copy
import getpass
import json
import math
import os
import pwd
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools import trusted_native_eval as trusted

TASK_PATH = "experimental/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8"
REAL_TASK = Path(__file__).resolve().parents[1] / TASK_PATH


@pytest.fixture
def committed_task(tmp_path):
    if shutil.which("rclone") is None:
        pytest.skip("trusted payload staging requires rclone")
    repo = tmp_path / "trusted-repo"
    task = repo / TASK_PATH
    task.mkdir(parents=True)
    subprocess.run([
        "rclone", "copy", str(REAL_TASK), str(task), "--transfers", "64000", "--progress",
        "--config", os.devnull, "--exclude", "build/**", "--exclude", "**/__pycache__/**",
    ], check=True, timeout=300)
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "trusted"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    agent = tmp_path / "agent"
    agent.mkdir()
    candidate = agent / "candidate.cu"
    candidate.write_bytes((task / trusted.SOURCE).read_bytes())
    return repo, task, commit, agent, candidate


def report(mode, identity, cases, timing=2.0):
    run_id = uuid.uuid4().hex + uuid.uuid4().hex[:16]
    proofs = {leg: {"leg": leg, "production_namespace_rebound": False,
                    "extension_sha256": "a" * 64, "fresh_compilation": True,
                    "source_tree_sha256": identity["source_tree_sha256"]} for leg in trusted.LEGS}
    payload = {"status": "ok", "run_id": run_id, **identity}
    if mode != "performance":
        payload["workers"] = {
            leg: {**payload, "mode": mode, "leg": leg, "native_build": proofs[leg],
                  "results": [] if mode == "compile" else [
                      {"case_id": c["case_id"], "shape": c["shape"], "trace_call_count": c["trace_call_count"],
                       "correct": True, "input_immutable": True, "negative_controls": True, "seeds": [0, 1]}
                      for c in cases]}
            for leg in trusted.LEGS}
        return payload
    payload.update(benchmark_method="cuda_graph", warmup_iterations=10, benchmark_iterations=100,
                   isolated_native_processes=True, fresh_input_and_poisoned_output_each_replay=True,
                   native_builds=proofs, paired_cases=[], test_cases=[])
    for index, case in enumerate(cases):
        value = timing * (index + 1)
        times = {}
        for leg, sample in (("candidate_native", value), ("production_native", 9999.0)):
            times[leg] = {"samples_ms": [sample] * 100, "mean_ms": sample, "min_ms": sample, "max_ms": sample}
        payload["paired_cases"].append({"case_id": case["case_id"], "shape": case["shape"],
                                        "trace_call_count": case["trace_call_count"],
                                        "graph_correctness": True, "timings": times})
        payload["test_cases"].append({"test_case_id": case["case_id"], "shape": case["shape"],
                                     "execution_time_ms": value, "params": {"trace_call_count": case["trace_call_count"]},
                                     "metadata": {"benchmark_method": "cuda_graph"}})
    return payload


def test_retest_ignores_agent_reports_caches_and_dirty_trusted_checkout(committed_task, tmp_path, monkeypatch):
    repo, task, commit, agent, candidate = committed_task
    (agent / "performance_report.json").write_text('{"arithmetic_mean_speedup": 999999}')
    (agent / "sitecustomize.py").write_text("raise RuntimeError('must never load')")
    (agent / "__pycache__").mkdir()
    (agent / "__pycache__/oracle.pyc").write_bytes(b"poison")
    (task / "ut/oracle.py").write_text("raise RuntimeError('dirty checkout must never load')")
    output = tmp_path / "accepted"
    scratch = tmp_path / "local-scratch"
    original_check_output = subprocess.check_output
    image = json.loads((task / "cases.json").read_text())["runtime_image"]

    def check_output(command, *args, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"RepoDigests": [image], "Id": "sha256:" + "b" * 64}])
        return original_check_output(command, *args, **kwargs)

    calls = []

    def run_mode(image, stage, build, render_device, mode, log_path, timeout, jit_source=None):
        calls.append((stage.name, mode, build))
        assert stage.parent.parent == scratch
        assert not stage.is_relative_to(output)
        assert jit_source == stage.parent / "image-jit/jit"
        assert not build.exists()
        assert not stage.is_relative_to(agent)
        assert not (stage / "sitecustomize.py").exists()
        assert "dirty checkout" not in (stage / "ut/oracle.py").read_text()
        cases = json.loads((stage / "cases.json").read_text())["cases"]
        return report(mode, trusted.identities(stage), cases, 4.0 if stage.name == "reference" else 2.0)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(trusted, "seed_image_cache", lambda *a, **k: {"complete_parity": True})
    monkeypatch.setattr(trusted, "run_mode", run_mode)
    result = trusted.trusted_retest(repo=repo, commit=commit, task_path=TASK_PATH, candidate=candidate,
                                   agent_workspace=agent, output=output, render_device="/dev/dri/renderD128",
                                   scratch_dir=scratch)
    assert [(leg, mode) for leg, mode, _ in calls] == [
        (leg, mode) for leg in ("reference", "candidate") for mode in trusted.MODES]
    assert len({build for _, _, build in calls}) == 6
    assert result["arithmetic_mean_speedup"] == 2.0
    assert [row["speedup"] for row in result["cases"]] == [2.0] * 3
    assert result["candidate_source_sha256"] == trusted.sha256(candidate.read_bytes())
    assert result["full_case_coverage"] is True
    assert (output / "trusted_measurement.json").is_file()
    assert not (output / "task_result.yaml").exists()
    assert not (output / "validation_report.yaml").exists()


@pytest.mark.parametrize("parent", [False, True])
def test_candidate_symlink_is_rejected_before_extract_or_docker(tmp_path, monkeypatch, parent):
    agent = tmp_path / "agent"
    agent.mkdir()
    source = agent / "source"
    source.mkdir()
    candidate = source / "candidate.cu"
    candidate.write_text("kernel")
    if parent:
        source.rename(agent / "redirected")
        source.symlink_to("redirected", target_is_directory=True)
    else:
        candidate.rename(source / "regular.cu")
        candidate.symlink_to("regular.cu")
    monkeypatch.setattr(trusted, "extract_task", lambda *a: pytest.fail("must reject before extraction"))
    with pytest.raises(OSError):
        trusted.trusted_retest(repo=tmp_path / "repo", commit="a" * 40, task_path=TASK_PATH,
                               candidate=candidate, agent_workspace=agent, output=tmp_path / "out",
                               render_device="/dev/dri/renderD128")


def test_host_hook_rejected_by_committed_guard_before_docker(committed_task, tmp_path, monkeypatch):
    repo, task, commit, agent, candidate = committed_task
    candidate.write_text(candidate.read_text() + '\n__attribute__((constructor)) void forge() {}\n')
    (task / "ut/source_guard.py").write_text("def validate_source(*args): pass\n")
    original = subprocess.check_output

    def forbid_docker(command, *args, **kwargs):
        assert command[0] != "docker"
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", forbid_docker)
    with pytest.raises(subprocess.CalledProcessError):
        trusted.trusted_retest(repo=repo, commit=commit, task_path=TASK_PATH, candidate=candidate,
                               agent_workspace=agent, output=tmp_path / "out", render_device="/dev/dri/renderD128")
    assert not (tmp_path / "out/trusted_measurement.json").exists()


def test_mutable_commit_name_and_unsafe_output_are_rejected(committed_task, tmp_path):
    repo, _task, commit, agent, candidate = committed_task
    with pytest.raises(ValueError, match="explicit full Git commit"):
        trusted.extract_task(repo, "HEAD", TASK_PATH, tmp_path / "extracted")
    with pytest.raises(ValueError, match="outside the agent workspace"):
        trusted.trusted_retest(repo=repo, commit=commit, task_path=TASK_PATH, candidate=candidate,
                               agent_workspace=agent, output=agent / "out", render_device="/dev/dri/renderD128")
    with pytest.raises(ValueError, match="candidate must be in declared"):
        trusted.trusted_retest(repo=repo, commit=commit, task_path=TASK_PATH,
                               candidate=agent / ".." / "external.cu", agent_workspace=agent,
                               output=tmp_path / "out", render_device="/dev/dri/renderD128")
    with pytest.raises(ValueError, match="scratch directory must be outside"):
        trusted.trusted_retest(repo=repo, commit=commit, task_path=TASK_PATH, candidate=candidate,
                               agent_workspace=agent, output=tmp_path / "out", scratch_dir=agent / "scratch",
                               render_device="/dev/dri/renderD128")
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "trusted_measurement.json").write_text("user-owned previous run")
    with pytest.raises(FileExistsError):
        trusted.trusted_retest(repo=repo, commit=commit, task_path=TASK_PATH, candidate=candidate,
                               agent_workspace=agent, output=destination, render_device="/dev/dri/renderD128")
    assert (destination / "trusted_measurement.json").read_text() == "user-owned previous run"


def test_docker_command_has_only_fresh_task_and_build_mounts(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda path: SimpleNamespace(st_gid=109))
    command = trusted.docker_command("registry/image@sha256:" + "a" * 64, tmp_path / "task",
                                     tmp_path / "build", "/dev/dri/renderD128", "unique")
    assert all(flag in command for flag in ("--network=none", "--read-only", "--cap-drop=ALL",
                                            "--security-opt=no-new-privileges", "--pull=never"))
    assert "--privileged" not in command and "--ipc=host" not in command
    assert command.count("--mount") == 2
    assert f"type=bind,src={tmp_path / 'task'},dst=/task,readonly" in command
    assert not any("docker.sock" in x or ".codex" in x or ".cache" in x for x in command)
    assert not any(value.startswith("AITER_JIT_DIR=") for value in command)
    assert "HOME=/tmp" in command
    assert "USER=aka-evaluator" in command and "LOGNAME=aka-evaluator" in command
    assert "TORCHINDUCTOR_CACHE_DIR=/cache/inductor" in command
    assert "/tmp:rw,exec,nosuid,mode=1777,size=8g" in command
    assert command[-4:] == ["registry/image@sha256:" + "a" * 64, "-I", "-B", "/task/scripts/task_runner.py"]


def test_runtime_username_does_not_require_image_passwd_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda path: SimpleNamespace(st_gid=109))
    command = trusted.docker_command("image", tmp_path / "task", tmp_path / "build",
                                     "/dev/dri/renderD128", "unique")
    for index, argument in enumerate(command[:-1]):
        if argument == "--env":
            key, value = command[index + 1].split("=", 1)
            if key in ("USER", "LOGNAME"):
                monkeypatch.setenv(key, value)

    def missing_passwd(uid):
        raise KeyError(f"getpwuid(): uid not found: {uid}")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd)
    assert getpass.getuser() == "aka-evaluator"


def test_verified_cache_override_does_not_make_runtime_root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda path: SimpleNamespace(st_gid=109))
    command = trusted.docker_command("image", tmp_path / "task", tmp_path / "build",
                                     "/dev/dri/renderD128", "unique", tmp_path / "complete-cache")
    assert "AITER_JIT_DIR=/aiter-jit" in command
    assert f"type=bind,src={tmp_path / 'complete-cache'},dst=/aiter-jit" in command
    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "--cap-add=CHOWN" not in command


def test_payload_copy_uses_required_rclone_flags_and_fails_on_partial_copy(tmp_path, monkeypatch):
    source, destination = tmp_path / "reference", tmp_path / "candidate"
    source.mkdir()
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(trusted, "identities", lambda path: {"source": "complete" if path == source else "partial"})
    with pytest.raises(ValueError, match="differs from its trusted reference"):
        trusted.copy_payload(source, destination)
    assert calls == [([
        "rclone", "copy", str(source), str(destination), "--transfers", "64000", "--progress",
        "--config", os.devnull,
    ], {"check": True, "timeout": 300})]


def test_payload_transfer_failure_stops_staging(tmp_path, monkeypatch):
    source = tmp_path / "reference"
    source.mkdir()

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(trusted, "identities", lambda path: pytest.fail("must stop on rclone failure"))
    with pytest.raises(subprocess.CalledProcessError):
        trusted.copy_payload(source, tmp_path / "candidate")


@pytest.mark.parametrize("attack", ["stale", "missing", "duplicate", "shape", "summary", "nan", "zero", "short", "proof", "score"])
def test_host_rejects_forged_or_incomplete_report(attack):
    cases = [{"case_id": str(i), "shape": [8192, i + 128], "trace_call_count": 1} for i in range(3)]
    identity = {"package_sha256": "a" * 64, "source_tree_sha256": "b" * 64}
    payload = report("performance", identity, cases)
    row = payload["paired_cases"][0]
    timing = row["timings"]["candidate_native"]
    if attack == "stale":
        payload["package_sha256"] = "c" * 64
    elif attack == "missing":
        payload["paired_cases"].pop()
    elif attack == "duplicate":
        payload["paired_cases"][1] = copy.deepcopy(row)
    elif attack == "shape":
        row["shape"] = [1, 1]
    elif attack == "summary":
        timing["mean_ms"] = 0.000001
    elif attack == "nan":
        timing["samples_ms"][0] = math.nan
    elif attack == "zero":
        timing["samples_ms"][0] = 0
    elif attack == "short":
        timing["samples_ms"].pop()
    elif attack == "proof":
        payload["native_builds"]["candidate_native"]["fresh_compilation"] = False
    else:
        payload["test_cases"][0]["execution_time_ms"] = 0.000001
    with pytest.raises(ValueError):
        trusted.validate_report(payload, "performance", identity, cases)


def test_container_failure_cannot_publish_success(tmp_path, monkeypatch):
    task, build = tmp_path / "task", tmp_path / "build"
    task.mkdir()
    calls = []
    monkeypatch.setattr(trusted, "docker_command", lambda *a: ["docker", "run"])

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        trusted.run_mode("image", task, build, "/dev/dri/renderD128", "compile", tmp_path / "log", 30)
    assert calls[-1][:3] == ["docker", "rm", "-f"]
    assert not (tmp_path / "trusted_measurement.json").exists()


@pytest.mark.parametrize("fails", [False, True])
def test_worker_logs_survive_build_cleanup(tmp_path, monkeypatch, fails):
    task, build = tmp_path / "task", tmp_path / "build"
    task.mkdir()
    log_path = tmp_path / "reference_compile.log"
    original_run = subprocess.run
    monkeypatch.setattr(trusted, "docker_command", lambda *a: ["docker", "run"])

    def run(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            (build / "compile_candidate_native.log").write_text("compiler error details\n")
            (build / "compile_report.json").write_text('{"status": "ok"}')
            if fails:
                raise subprocess.CalledProcessError(7, command)
            return None
        if command[:2] == ["docker", "rm"]:
            return None
        assert command[0:2] == ["rclone", "copyto"]
        assert command[4:7] == ["--transfers", "64000", "--progress"]
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    if fails:
        with pytest.raises(subprocess.CalledProcessError) as error:
            trusted.run_mode("image", task, build, "/dev/dri/renderD128", "compile", log_path, 30)
        assert error.value.returncode == 7
    else:
        assert trusted.run_mode("image", task, build, "/dev/dri/renderD128", "compile", log_path, 30) == {"status": "ok"}
    shutil.rmtree(build)
    saved = tmp_path / "reference_compile.diagnostics/compile_candidate_native.log"
    assert saved.read_text() == "compiler error details\n"
    manifest = json.loads((saved.parent / "manifest.json").read_text())
    assert manifest["logs"][saved.name]["sha256"] == trusted.sha256(saved.read_bytes())
    raw_report = saved.parent / "compile_report.json"
    assert json.loads(raw_report.read_text()) == {"status": "ok"}
    assert manifest["reports"][raw_report.name]["sha256"] == trusted.sha256(raw_report.read_bytes())


def test_diagnostic_copy_failure_preserves_original_container_error(tmp_path, monkeypatch):
    task, build = tmp_path / "task", tmp_path / "build"
    task.mkdir()
    monkeypatch.setattr(trusted, "docker_command", lambda *a: ["docker", "run"])

    def run(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            (build / "compile_candidate_native.log").write_text("compiler failed")
            raise subprocess.CalledProcessError(7, command)
        if command[0] == "rclone":
            raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        trusted.run_mode("image", task, build, "/dev/dri/renderD128", "compile", tmp_path / "phase.log", 30)
    assert error.value.returncode == 7
    assert "Additional cleanup/diagnostic failure" in error.value.__notes__[0]
    assert json.loads((tmp_path / "phase.diagnostics/manifest.json").read_text())["errors"]
