"""Portable phases compile FlyDSL afresh without changing AITER cache lookup."""

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from src.tools import trusted_task_eval as trusted


@pytest.fixture
def phase_runtime(tmp_path, monkeypatch):
    task, staging, output = (tmp_path / name for name in ("task", "staging", "output"))
    for path in (task, staging, output):
        path.mkdir()
    image = "registry/image@sha256:" + "a" * 64
    gpu = {"rocr_uuid": "GPU-000000000000abcd"}
    cache_source = tmp_path / "verified-cache"
    (cache_source / "flydsl_cache").mkdir(parents=True)
    (cache_source / "module.so").write_bytes(b"precompiled AITER module")
    (cache_source / "flydsl_cache/compiled.pkl").write_bytes(b"image FlyDSL artifact")
    original_cache = trusted.tree_manifest(cache_source)
    commands = []
    lifecycle = []
    inspections = {}
    memory = {"status": "observed", "counters": {"memory.current": "4096"}}
    original_stat = Path.stat

    def cpu_stat(path, *args, **kwargs):
        if str(path) in ("/dev/kfd", "/dev/dri/renderD128"):
            return SimpleNamespace(st_gid=1000)
        return original_stat(path, *args, **kwargs)

    def copy_cache(source, destination, *, timeout):
        assert source == cache_source and timeout == 30
        shutil.copytree(source, destination, symlinks=True)

    def invoke(command, **_kwargs):
        if command[:2] == ["docker", "run"]:
            commands.append(command)
            name = command[command.index("--name") + 1]
            lifecycle.append(("run", name))
            inspections[name] = {
                "Id": f"{len(commands):064x}", "Name": "/" + name,
                "Image": "sha256:" + "b" * 64,
                "State": {"Status": "exited", "OOMKilled": False, "ExitCode": 0, "Error": "",
                          "StartedAt": "2026-01-01T00:00:00Z", "FinishedAt": "2026-01-01T00:00:01Z"},
                "HostConfig": {"Memory": 1 << 30, "MemorySwap": 2 << 30, "NanoCpus": 2_000_000_000,
                               "CpuQuota": 0, "CpuPeriod": 0, "CpusetCpus": "", "PidsLimit": 128},
                "Config": {"Env": ["UNIT_TEST_ENV_MUST_NOT_BE_RETAINED=1"]},
            }
            build = next(value.split("src=", 1)[1].split(",", 1)[0] for value in command
                         if value.endswith(",dst=/task/build"))
            request_path = next(value.split("src=", 1)[1].split(",", 1)[0] for value in command
                                if value.endswith(",dst=/evaluation-request.json,readonly"))
            request = json.loads(Path(request_path).read_text())
            (Path(build) / "gpu_preflight.json").write_text("{}")
            (Path(build) / (request["phase"] + "_report.json")).write_text(json.dumps({"unit_test": True}))
        elif command[:2] == ["docker", "inspect"]:
            name = command[2]
            assert command == ["docker", "inspect", name]
            assert lifecycle[-1] == ("run", name)
            assert _kwargs == {"capture_output": True, "text": True, "check": False, "timeout": 10}
            lifecycle.append(("inspect", name))
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps([inspections[name]]), stderr="")
        else:
            assert command[:3] == ["docker", "rm", "-f"]
            name = command[3]
            assert lifecycle[-1] == ("inspect", name)
            lifecycle.append(("rm", name))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(Path, "stat", cpu_stat)
    monkeypatch.setattr(trusted, "copy_cache", copy_cache)
    monkeypatch.setattr(trusted, "validate_preflight", lambda *_args: None)
    monkeypatch.setattr(trusted, "preserve_diagnostics", lambda *_args: None)
    monkeypatch.setattr(trusted, "process_memory_snapshot", lambda: memory)
    monkeypatch.setattr(trusted.subprocess, "run", invoke)

    def run(leg="candidate", phase="compile", request_id="a" * 48, seeded=True):
        request = {"phase": phase, "gpu": gpu, "request_id": request_id}
        return trusted.run_phase(image, task, staging, output, leg, request,
                                 "/dev/dri/renderD128", 30, cache_source if seeded else None)

    return SimpleNamespace(run=run, image=image, staging=staging, output=output, cache_source=cache_source,
                           original_cache=original_cache, commands=commands, lifecycle=lifecycle,
                           inspections=inspections, memory=memory)


def assert_preserved_exit(runtime, leg, phase):
    command = runtime.commands[-1]
    name = command[command.index("--name") + 1]
    inspected = runtime.inspections[name]
    path = runtime.output / (leg + "_" + phase + ".container.json")
    record = json.loads(path.read_text())
    assert record["schema"] == "trusted-container-exit-v1" and record["status"] == "observed"
    assert record["name"] == name
    assert record["process_memory_before"] == record["process_memory_after"] == runtime.memory
    assert record["container"] == {
        "id": inspected["Id"], "name": inspected["Name"], "image_id": inspected["Image"],
        "state": inspected["State"], "limits": inspected["HostConfig"],
    }
    assert "UNIT_TEST_ENV_MUST_NOT_BE_RETAINED" not in path.read_text()
    assert runtime.lifecycle[-3:] == [("run", name), ("inspect", name), ("rm", name)]


def test_all_seeded_phases_use_absent_flydsl_cache_and_preserve_aiter_lookup(phase_runtime):
    runtime = phase_runtime
    settings = []
    for index, (leg, phase) in enumerate((leg, phase) for leg in ("reference", "candidate")
                                         for phase in trusted.PHASES):
        request_id = f"{index:048x}"
        assert runtime.run(leg, phase, request_id) == {"unit_test": True}
        command = runtime.commands[-1]
        setting = "FLYDSL_RUNTIME_CACHE_DIR=/aiter-jit/fresh_flydsl_" + request_id
        settings.append(setting)
        assert command.count(setting) == 1
        assert command[command.index(setting) - 1] == "--env"
        assert command.index(setting) < command.index(runtime.image)
        assert "AITER_JIT_DIR=/aiter-jit" in command
        cache = runtime.staging / (leg + "_" + phase + "_jit")
        assert f"type=bind,src={cache},dst=/aiter-jit" in command
        assert not (cache / ("fresh_flydsl_" + request_id)).exists()
        assert trusted.tree_manifest(cache) == runtime.original_cache
        assert_preserved_exit(runtime, leg, phase)
    assert len(settings) == len(set(settings)) == 6
    assert trusted.tree_manifest(runtime.cache_source) == runtime.original_cache


@pytest.mark.parametrize("entry", ["directory", "file", "dangling_symlink"])
def test_existing_flydsl_phase_path_rejected_before_container_launch(phase_runtime, entry):
    runtime = phase_runtime
    path = runtime.cache_source / ("fresh_flydsl_" + "a" * 48)
    if entry == "directory":
        path.mkdir()
        (path / "compiled.pkl").write_bytes(b"old phase artifact")
    elif entry == "file":
        path.write_bytes(b"occupied cache path")
    else:
        path.symlink_to("missing-target")
    with pytest.raises(ValueError, match="FlyDSL phase cache must start absent"):
        runtime.run()
    assert runtime.commands == []


def test_unseeded_phase_keeps_default_cache_lookup(phase_runtime):
    runtime = phase_runtime
    assert runtime.run(seeded=False) == {"unit_test": True}
    command, = runtime.commands
    assert not any(value.startswith(("FLYDSL_RUNTIME_CACHE_DIR=", "AITER_JIT_DIR=")) for value in command)
    assert not any("dst=/aiter-jit" in value for value in command)
    assert_preserved_exit(runtime, "candidate", "compile")
