"""The Qwen phase bridge preserves the maintained container boundary."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from src.tools import trusted_task_eval as trusted
from test_trusted_qwen_eval import ROOT, clone


def test_phase_uses_readonly_inputs_no_network_auth_and_exact_gpu(tmp_path, monkeypatch):
    task = clone(ROOT / "tasks/headkernel/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode",
                 tmp_path / "task")
    contract = trusted.package_contract(task)
    staging, output = tmp_path / "stage", tmp_path / "output"
    staging.mkdir(); output.mkdir()
    gpu = {"render_device": "/dev/dri/renderD129", "rocr_uuid": "GPU-0000000000000001",
           "pci_bus_id": "0000:01:00.0"}
    request = {"phase": "correctness", "request_id": "a" * 48, "gpu": gpu}
    stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda path, *a, **k:
        SimpleNamespace(st_gid=109) if str(path) in ("/dev/kfd", gpu["render_device"]) else stat(path, *a, **k))
    commands, diagnostics = [], []
    def run(command, **kwargs):
        commands.append(command)
        assert command[:2] in (["docker", "run"], ["docker", "rm"])
        if command[:2] == ["docker", "run"]:
            build = task / "build"
            (build / "correctness_report.json").write_text('{"status":"ok"}')
            (build / "gpu_preflight.json").write_text(json.dumps({
                "hip_device_count": 1, "hsa_gpu_uuids": [gpu["rocr_uuid"]],
                "hip_pci_bus_ids": [gpu["pci_bus_id"]], "rocr_visible_devices": gpu["rocr_uuid"]}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(trusted.subprocess, "run", run)
    monkeypatch.setattr(trusted, "preserve_container_exit", lambda *a: None)
    monkeypatch.setattr(trusted, "preserve_diagnostics", lambda *a, **k: diagnostics.append(k))
    result = trusted.run_phase(contract["manifest"]["runtime_image"], task, staging, output,
                               "reference", request, gpu["render_device"], 123,
                               qwen_contract=contract)
    assert result == {"status": "ok"}
    command = commands[0]
    assert "--network=none" in command and "--read-only" in command and "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command and "--privileged" not in command
    assert f"type=bind,src={task},dst=/task,readonly" in command
    assert f"type=bind,src={task / 'ut/meta.json'},dst=/task/ut/meta.json,readonly" in command
    assert not any(".codex" in part or ".claude" in part or "docker.sock" in part for part in command)
    assert "ROCR_VISIBLE_DEVICES=" + gpu["rocr_uuid"] in command
    assert "--device" in command and "/dev/dri" not in command and gpu["render_device"] in command
    assert "/trusted_qwen_phase.py" in command
    assert command[-6:] == ["--task", "/task", "--request", "/evaluation-request.json", "--timeout", "123"]
    assert commands[-1][:3] == ["docker", "rm", "-f"]
    assert diagnostics[0]["recursive"] is True
    assert ".runtime-ut/meta.json" in diagnostics[0]["excluded"]


def test_nested_raw_and_logs_survive_but_readonly_mount_placeholders_do_not(tmp_path):
    build = tmp_path / "build"
    (build / "reports/request").mkdir(parents=True)
    (build / ".runtime-ut").mkdir()
    (build / "reports/request/raw.json").write_text('{"samples":[1,2,3]}')
    (build / "reports/request/stderr.txt").write_text("retained worker context")
    (build / ".runtime-ut/meta.json").touch()
    destination = tmp_path / "diagnostics"
    trusted.preserve_diagnostics(build, destination, recursive=True, excluded={".runtime-ut/meta.json"})
    hashes = json.loads((destination / "hashes.json").read_text())
    assert set(hashes) == {"reports/request/raw.json", "reports/request/stderr.txt"}
    for name, digest in hashes.items():
        assert (destination / name).read_bytes() == (build / name).read_bytes()
        assert trusted.sha256((destination / name).read_bytes()) == digest
    assert not (destination / ".runtime-ut/meta.json").exists()
