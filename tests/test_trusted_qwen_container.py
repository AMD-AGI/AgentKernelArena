"""The Qwen phase bridge preserves the maintained container boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.tools import trusted_task_eval as trusted
from src.tools import trusted_qwen_eval as adapter
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


@pytest.mark.parametrize("suffix", ["fused_recurrent_gated_delta_rule_decode", "paged_attention_decode"])
def test_no_protected_UT_input_has_a_renameable_unmounted_parent(tmp_path, suffix):
    task = ROOT / "tasks/headkernel" / ("qwen3.8-2.4t__" + suffix)
    contract = trusted.package_contract(task)
    build, runtime = tmp_path / "build", tmp_path / "private-ut"
    build.mkdir()
    mounts, _ = adapter.runtime_mounts(task, build, contract, runtime_root=runtime)
    records = [dict(part.split("=", 1) if "=" in part else (part, True)
                    for part in mounts[index].split(",")) for index in range(1, len(mounts), 2)]
    readonly = {Path(record["dst"]) for record in records if record.get("readonly")}
    outputs = [Path(name) for name in contract["config"]["harness_protection"]["runtime_outputs"]]
    for path in (task / "ut").rglob("*"):
        relative = path.relative_to(task)
        if path.is_dir() or any(relative == out or relative.is_relative_to(out) for out in outputs):
            continue
        # The first child of /task/ut must itself be a read-only mount point.
        # A read-only nested file alone permits rename(parent)+replacement.
        anchor = Path("/task/ut") / relative.parts[1]
        assert anchor in readonly, (relative, "writable parent can shadow this input")
    assert Path("/task/ut/baseline_overlay") in readonly
    assert Path("/task/ut/kernel_src") in readonly
    assert not runtime.is_relative_to(build) and not runtime.is_relative_to(task)
    with pytest.raises(ValueError, match="must not be exposed"):
        adapter.runtime_mounts(task, build, contract, runtime_root=build / ".runtime-ut")


def test_protected_parent_rename_is_rejected_in_private_mount_namespace(tmp_path):
    if not shutil.which("unshare") or not shutil.which("mount"):
        pytest.skip("private user/mount namespace tools are unavailable")
    probe = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--", "true"],
                           capture_output=True, text=True)
    if probe.returncode:
        if "Operation not permitted" in probe.stderr or "uid_map" in probe.stderr:
            pytest.skip("host denies user-namespace uid_map; deterministic mount-layout regression still runs")
        pytest.fail(probe.stderr)
    task = tmp_path / "inputs"
    (task / "ut/baseline_overlay").mkdir(parents=True)
    (task / "ut/baseline_overlay/protected.py").write_text("PROTECTED = True\n")
    build = task / "build"
    build.mkdir()
    runtime = tmp_path / "private-runtime"
    contract = {"config": {"harness_protection": {"runtime_outputs": ["ut/result.json", "ut/reports"]}}}
    mounts, _ = adapter.runtime_mounts(task, build, contract, runtime_root=runtime)
    records = [dict(part.split("=", 1) if "=" in part else (part, True)
                    for part in mounts[index].split(",")) for index in range(1, len(mounts), 2)]
    view = tmp_path / "container"
    (view / "task").mkdir(parents=True)
    specification = tmp_path / "mounts.json"
    specification.write_text(json.dumps({"view": str(view), "task": str(task), "build": str(build), "mounts": records}))
    script = '''import errno,json,os,subprocess,sys
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text());view=Path(spec['view'])
def bind(source,destination,readonly=False):
    subprocess.run(['mount','--bind',source,str(destination)],check=True)
    if readonly:subprocess.run(['mount','-o','remount,bind,ro',str(destination)],check=True)
bind(spec['task'],view/'task',True);bind(spec['build'],view/'task/build')
for item in spec['mounts']:bind(item['src'],view/item['dst'].lstrip('/'),bool(item.get('readonly')))
os.chroot(view);os.chdir('/')
parent=Path('/task/ut/baseline_overlay')
try:parent.rename(parent.with_name('moved'))
except OSError as error:assert error.errno in (errno.EBUSY,errno.EROFS,errno.EPERM)
else:raise AssertionError('protected parent could be renamed')
assert (parent/'protected.py').read_text()=='PROTECTED = True'+chr(10)
assert not Path('/task/build/.runtime-ut').exists()
Path('/task/ut/result.json.tmp').write_text('{}');Path('/task/ut/result.json.tmp').replace('/task/ut/result.json')
'''
    result = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--", sys.executable,
                             "-B", "-c", script, str(specification)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
