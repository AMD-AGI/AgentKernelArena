"""A failed phase retains resource evidence without changing its verdict."""
import json
import subprocess
from pathlib import Path

import pytest

from src.tools import trusted_task_eval as trusted


def prepare_phase(tmp_path, monkeypatch):
    staging, output = tmp_path / "staging", tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    image = "image@sha256:" + "a" * 64
    monkeypatch.setattr(trusted, "docker_command", lambda image, task, build, render, name, cache:
                        ["docker", "run", "--rm", "--name", name, image, "/task/scripts/task_runner.py"])
    monkeypatch.setattr(trusted, "command_with_binding", lambda command, *args: command)
    monkeypatch.setattr(trusted, "process_memory_snapshot", lambda:
                        {"status": "observed", "counters": {"memory.max": "274877906944"}})
    monkeypatch.setattr(trusted, "preserve_diagnostics", lambda *args: None)
    arguments = (image, tmp_path / "task", staging, output, "reference",
                 {"phase": "performance", "gpu": {}}, "/dev/dri/renderD184", 7200)
    return arguments, output


def test_oom_evidence_precedes_cleanup_and_original_failure_survives(tmp_path, monkeypatch):
    arguments, output = prepare_phase(tmp_path, monkeypatch)
    original = subprocess.CalledProcessError(137, ["docker", "run"])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            assert "--rm" not in command
            assert command[command.index("--cpus") + 1] == "24"
            assert command[command.index("--memory") + 1] == "192g"
            raise original
        if command[:2] == ["docker", "inspect"]:
            payload = [{"Id": "b" * 64, "Name": "/" + command[-1], "Image": "sha256:abc",
                        "State": {"Status": "exited", "OOMKilled": True, "ExitCode": 137},
                        "HostConfig": {"Memory": 192 << 30, "NanoCpus": 24_000_000_000},
                        "Config": {"Env": ["SECRET=test-credential-must-not-be-archived"]}}]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        assert command[:3] == ["docker", "rm", "-f"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        trusted.run_phase(*arguments, container_resources={"cpus": 24, "memory_gib": 192})
    assert caught.value is original
    assert [command[1] for command in calls] == ["run", "inspect", "rm"]
    saved = output / "reference_performance.container.json"
    record = json.loads(saved.read_text())
    assert record["status"] == "observed"
    assert record["container"]["state"]["OOMKilled"] is True
    assert record["container"]["limits"]["Memory"] == 192 << 30
    assert "SECRET" not in saved.read_text() and "Config" not in saved.read_text()
    assert not (output / "trusted_measurement.json").exists()


def test_unavailable_inspection_does_not_claim_oom_or_skip_cleanup(tmp_path, monkeypatch):
    arguments, output = prepare_phase(tmp_path, monkeypatch)
    original = subprocess.CalledProcessError(137, ["docker", "run"])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise original
        if command[:2] == ["docker", "inspect"]:
            raise subprocess.TimeoutExpired(command, 10)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        trusted.run_phase(*arguments)
    assert caught.value is original
    record = json.loads((output / "reference_performance.container.json").read_text())
    assert record["status"] == "unavailable" and record["container"] is None
    assert calls[-1][:3] == ["docker", "rm", "-f"]


def test_resource_record_write_failure_still_cleans_up_and_preserves_error(tmp_path, monkeypatch):
    arguments, output = prepare_phase(tmp_path, monkeypatch)
    original = subprocess.CalledProcessError(137, ["docker", "run"])
    (output / "reference_performance.container.json").write_text("old evidence")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise original
        return subprocess.CompletedProcess(command, 1, "", "not available")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        trusted.run_phase(*arguments)
    assert caught.value is original and caught.value.__notes__
    assert (output / "reference_performance.container.json").read_text() == "old evidence"
    assert calls[-1][:3] == ["docker", "rm", "-f"]


def test_snapshot_reads_only_current_namespaced_cgroup(tmp_path):
    proc, mount = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    mount.mkdir()
    (proc / "self/cgroup").write_text("0::/\n")
    (proc / "self/mountinfo").write_text(f"1 2 0:3 /jobs/current {mount} rw - cgroup2 cgroup rw\n")
    (mount / "memory.current").write_text("100\n")
    (mount / "memory.max").write_text("200\n")
    (mount / "memory.peak").write_text("200\n")
    (mount / "memory.events").write_text("oom 1\noom_kill 1\n")
    record = trusted.process_memory_snapshot(proc)
    assert record["status"] == "observed" and record["cgroup"] == str(mount)
    assert record["counters"]["memory.peak"] == "200"
    assert record["counters"]["memory.events"] == "oom 1\noom_kill 1"


@pytest.mark.parametrize("resources", [{"cpus": 0}, {"cpus": float("inf")},
                                       {"memory_gib": -1}, {"memory_gib": True},
                                       {"other": 10}])
def test_invalid_resource_limits_fail_before_launch(resources):
    with pytest.raises(ValueError):
        trusted.container_resource_arguments(resources)
