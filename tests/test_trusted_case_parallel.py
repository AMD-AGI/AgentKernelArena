"""Restricted whole-case container routing with mocked device execution only."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest
import yaml

from src.tools import case_parallel as cp
from src.tools import trusted_task_eval as trusted
from src import task_contract
from test_case_parallel import manifest_for, plan_for, row_for


ROOT = Path(__file__).resolve().parents[1]
CPU_FINALIZER = '''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import finalize_report
def finalize_case_union(manifest,request,union):
    return finalize_report({'schema_version':1,'status':'ok','request':request,'compiled':True,
        'cases':union['rows'],'compiled_specializations':union['compiled_specializations'],
        'case_parallelism':union['case_parallelism']},manifest,request)
'''


@pytest.fixture
def group(tmp_path, monkeypatch):
    monkeypatch.delenv(cp.PLAN_ENV, raising=False)
    monkeypatch.delenv(cp.HASH_ENV, raising=False)
    task = tmp_path / "task"
    for name in ("source", "ut", "scripts", "build"):
        (task / name).mkdir(parents=True)
    (task / "source/kernel.py").write_text("raise RuntimeError('candidate source must never be imported on the host')\n")
    (task / "scripts/task_runner.py").write_text(CPU_FINALIZER)
    shutil.copyfile(ROOT / "src/task_contract.py", task / "ut/evaluation_contract.py")
    manifest = manifest_for(16)
    (task / "cases.json").write_text(cp.canonical(manifest))
    request = {"schema_version": 1, "request_id": "fresh-parent", "phase": "performance",
               "manifest_sha256": cp.fingerprint(manifest), "package_sha256": "e" * 64,
               "source_sha256": {"source/kernel.py": hashlib.sha256((task / "source/kernel.py").read_bytes()).hexdigest()},
               "challenge_seed": 8}
    plan = plan_for(manifest, 8)
    path = tmp_path / "plan.json"
    path.write_text(cp.canonical(plan))
    path.chmod(0o444)
    plan = cp.load_plan(task, manifest, request, path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    plan["_manifest"] = manifest
    request = cp.bind_request(request, plan)
    staging, output = tmp_path / "staging", tmp_path / "output"
    staging.mkdir(); output.mkdir()
    return task, staging, output, manifest, request, plan


def fake_gpu_docker(monkeypatch, manifest):
    commands = []
    original_run = subprocess.run
    original_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda path, **kw: SimpleNamespace(st_gid=os.getgid())
                        if str(path) == "/dev/kfd" or str(path).startswith("/dev/dri/renderD") else original_stat(path, **kw))
    def fake(command, *args, **kwargs):
        if command[:2] == ["docker", "run"]:
            commands.append(command)
            mounts = {}
            for index, item in enumerate(command):
                if item == "--mount":
                    fields = dict(part.split("=", 1) for part in command[index+1].split(",") if "=" in part)
                    mounts[fields["dst"]] = Path(fields["src"])
            request = json.loads(mounts["/evaluation-request.json"].read_text())
            assignment = json.loads(mounts["/case-worker-assignment.json"].read_text())
            task, build = mounts["/task"], mounts["/task/build"]
            gpu = assignment["expected_gpu"]
            proof = {"hsa_gpu_uuids": [gpu["rocr_uuid"]], "hip_device_count": 1,
                     "hip_pci_bus_ids": [gpu["pci_bus_id"]], "rocr_visible_devices": gpu["rocr_uuid"]}
            (build / "gpu_preflight.json").write_text(cp.canonical(proof))
            cases = tuple(c for c in manifest["cases"] if c["case_id"] in assignment["case_ids"])
            context = cp.WorkerContext(task, manifest, request, assignment, cases,
                                      build / Path(assignment["output_directory"]).relative_to("build"), proof)
            rows = [] if request["phase"] == "compile" else [row_for(c, request["phase"]) for c in cases]
            compiled = [{"case_id": c["case_id"], "source_sha256": request["source_sha256"],
                         "candidate_invoked": True, "independent_reference_invoked": True} for c in cases]
            cp.finish_worker(context, rows=rows, compiled_specializations=compiled,
                             evidence={"completed_case_ids": assignment["case_ids"]})
            return SimpleNamespace(returncode=0)
        if command[:2] in (["docker", "rm"], ["docker", "stop"]):
            return SimpleNamespace(returncode=0)
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(trusted.subprocess, "run", fake)
    original_popen = subprocess.Popen
    def popen(command, *args, **kwargs):
        if command[:2] == ["docker", "run"]:
            fake(command)
            return SimpleNamespace(returncode=0, poll=lambda: 0)
        return original_popen(command, *args, **kwargs)
    monkeypatch.setattr(trusted.subprocess, "Popen", popen)
    monkeypatch.setattr(trusted, "preserve_container_exit", lambda *a: None)
    monkeypatch.setattr(trusted, "preserve_diagnostics", lambda *a, **k: None)
    return commands


def test_six_full_phases_use_48_fresh_restricted_workers_and_same_gpu_mapping(group, monkeypatch):
    task, staging, output, manifest, request, plan = group
    commands = fake_gpu_docker(monkeypatch, manifest)
    seen_mapping = None
    for leg in ("reference", "candidate"):
        for phase in trusted.PHASES:
            request = {**request, "request_id": leg + "-" + phase, "phase": phase}
            report = trusted.run_group_phase(plan["runtime_image"], task, staging, output, leg, request, plan, 30)
            task_contract.validate_report(report, manifest, request)
            cp.validate_group_report(task, manifest, report, required=True)
            assert len(report["compiled_specializations"]) == 16
            mapping = report["case_parallelism"]["case_gpu_mapping"]
            assert seen_mapping is None or mapping == seen_mapping
            seen_mapping = mapping
    assert len(commands) == 48
    assert len({command[command.index("--name")+1] for command in commands}) == 48
    assert len({command[command.index("--mount", command.index("--mount")+1)+1] for command in commands}) == 48
    for command in commands:
        assert "--read-only" in command and "--network=none" in command and "--cap-drop=ALL" in command
        assert "--privileged" not in command and "/dev/dri" not in command
        assert "--case-worker" in command and "/case-worker-assignment.json" in command
        assert "--cpus" in command and "--memory" in command
        assert any(arg.endswith("dst=/run/aka-case-gpu-plan.json,readonly") for arg in command)
    assert len(list(output.rglob("receipt.json"))) == 48


def test_group_failure_cancels_containers_and_never_finalizes(group, monkeypatch):
    task, staging, output, manifest, request, plan = group
    finished = []
    def worker(*args, worker_assignment, group_control, **kwargs):
        name = "cpu-test-" + str(worker_assignment["worker_id"])
        group_control.register(name)
        if worker_assignment["worker_id"] == 0:
            raise subprocess.CalledProcessError(9, "docker")
        assert group_control.cancelled.wait(2)
        group_control.finished(name)
        finished.append(name)
        raise RuntimeError("cancelled")
    stops = []
    monkeypatch.setattr(trusted, "run_phase", worker)
    monkeypatch.setattr(trusted.subprocess, "run", lambda cmd, **kw: stops.append(cmd) or SimpleNamespace(returncode=0))
    with pytest.raises(subprocess.CalledProcessError):
        trusted.run_group_phase(plan["runtime_image"], task, staging, output, "candidate", request, plan, 30)
    assert not list(staging.glob("*_full_report.json"))
    assert stops and all(cmd[:2] == ["docker", "stop"] for cmd in stops)


def test_trusted_retest_assembles_all_six_group_reports(group, monkeypatch):
    from src.tools import case_gpu_plan as admission
    task, staging, _, manifest, _, plan = group
    config = {"source_file_path": ["source/kernel.py"], "trusted_evaluation": {
        "schema_version": 1, "case_parallelism": {"schema_version": 1}},
        **{phase + "_timeout": value for phase, value in plan["phase_timeouts"].items()}}
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    for name in ("case_parallel.py", "gpu_binding.py"):
        shutil.copyfile(ROOT / "src/tools" / name, task / "ut" / name)
    contract = {"config": config, "manifest": manifest, "sources": ["source/kernel.py"],
                "distribution_correctness_mode": None, "distributions": None, "needs_cache": False,
                "scoring_policy": None, "guard": "ut/source_guard.py"}
    def prepare(*, reference, **kwargs):
        shutil.copytree(task, reference, ignore=shutil.ignore_patterns("build"))
        return contract, None
    monkeypatch.setattr(trusted, "prepare_reference", prepare)
    monkeypatch.setattr(trusted, "guard_sources", lambda *a: None)
    monkeypatch.setattr(trusted, "copy_cache", lambda source, destination: shutil.copytree(source, destination))
    monkeypatch.setattr(admission, "scheduler_state", lambda _: {
        "JobId": "12345", "JobState": "RUNNING", "Restarts": "0", "NodeList": "fresh-node",
        "AllocTRES": "cpu=192,mem=1536G,gres/gpu=8",
        "EndTime": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time()+3600))})
    monkeypatch.setattr(admission.socket, "gethostname", lambda: "fresh-node")
    monkeypatch.setattr(trusted, "select_gpu", lambda render: {k: v for k, v in next(
        b for b in plan["bindings"] if b["render_device"] == render).items() if k != "worker_id"})
    commands = fake_gpu_docker(monkeypatch, manifest)
    original_check_output = subprocess.check_output
    def inspect(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"RepoDigests": [plan["runtime_image"]], "Id": "cpu-test-image"}])
        return original_check_output(command, **kwargs)
    monkeypatch.setattr(subprocess, "check_output", inspect)
    output = staging.parent / "trusted-output"
    result = trusted.trusted_retest(repo=ROOT, commit=plan["baseline_commit"], task_path=plan["task_path"],
        candidate_workspace=task, output=output, scratch_dir=staging, timeout=30,
        case_gpu_plan=plan["_plan_path"], case_gpu_plan_sha256=plan["_plan_sha256"])
    assert len(commands) == 48 and result["case_parallelism"]["fresh_worker_containers"] == 48
    assert "gpu" not in result and result["gpu_group"] == plan["bindings"]
    assert all(set(result["reports"][leg]) == set(trusted.PHASES) for leg in ("reference", "candidate"))
    assert result["full_case_coverage"] is True and len(result["cases"]) == 16
    assert result["framework_task_validator_status"] == "not_asserted"


def test_populated_aiter_cache_uses_rclone_with_exact_parity(tmp_path, monkeypatch):
    if shutil.which("rclone") is None:
        pytest.skip("rclone unavailable")
    source, destination = tmp_path / "template:jit", tmp_path / "worker-cache"
    (source / "empty").mkdir(parents=True)
    (source / "jit.py").write_text("VALUE=1\n")
    (source / "alias.py").symlink_to("jit.py")
    monkeypatch.chdir(tmp_path)
    with (tmp_path / "copy.log").open("wb") as log:
        cp._copy_cache(source.relative_to(tmp_path), destination, deadline=time.time()+20, log=log)
    assert cp._cache_manifest(source) == cp._cache_manifest(destination)
    assert (destination / "alias.py").is_symlink()


@pytest.mark.parametrize("name", ["kimi-k3__attn_residual_aggregate_hip", "minimax-m3__gqa_share_sparse_fwd_kernel",
                                 "minimax-m3__gqa_share_sparse_decode_kernel", "minimax-m3__decode_score_kernel"])
@pytest.mark.parametrize("phase", trusted.PHASES)
def test_real_protected_finalizers_run_in_fresh_cpu_process_for_full_domains(tmp_path, monkeypatch, name, phase):
    original = ROOT / "tasks/headkernel" / name
    task = tmp_path / "task"
    for directory in ("scripts", "ut", "provenance"):
        (task / directory).mkdir(parents=True)
    shutil.copyfile(original / "scripts/task_runner.py", task / "scripts/task_runner.py")
    for path in (original / "ut").glob("*.py"):
        shutil.copyfile(path, task / "ut" / path.name)
    for relative in ("cases.json", "config.yaml", "task_definition.json", "provenance/PAIRED-REFERENCE.json"):
        if (original / relative).is_file():
            shutil.copyfile(original / relative, task / relative)
    config = yaml.safe_load((task / "config.yaml").read_text())
    sources = {}
    for relative in config["source_file_path"]:
        path = task / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Any accidental host import fails; the finalizer reads receipt data.
        path.write_text("raise RuntimeError('CPU finalizer imported candidate source')\n")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((task / "cases.json").read_text())
    request = {"schema_version": 1, "request_id": "cpu-protected-hook-" + phase, "phase": phase,
               "manifest_sha256": cp.fingerprint(manifest), "package_sha256": "a" * 64,
               "source_sha256": sources, "challenge_seed": 7}
    compiled = [{"case_id": case["case_id"], "source_sha256": sources, "candidate_invoked": True,
                 "independent_reference_invoked": True, "invoked_and_synchronized": True,
                 "candidate_launch_evidence": {"source_sha256": next(iter(sources.values())),
                    "engaged_kernels": ["CPU_TEST_ONLY"], "last_launches": [{"CPU_TEST_ONLY": True}]}}
                for case in manifest["cases"]]
    union = {"rows": [] if phase == "compile" else [row_for(case, phase) for case in manifest["cases"]],
             "compiled_specializations": compiled, "workers": [], "case_parallelism": {"CPU_TEST_ONLY": True}}
    if name in ("minimax-m3__gqa_share_sparse_decode_kernel", "minimax-m3__decode_score_kernel"):
        monkeypatch.syspath_prepend(str(task / "ut"))
        spec = importlib.util.spec_from_file_location("_cpu_finalizer_controls", task / "ut/workload_controls.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(os.sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        for row in union["rows"]:
            controls = module.RecordedControls(row["case"])
            policy = manifest["measurement"]
            row["workload_control_coverage"] = controls.coverage(policy, controls.targeted_variants(), controls.original_plan(policy))
            draws = [controls.choose(request["challenge_seed"] + i) for i in range(110)]
            row["workload_control_sampling"] = controls.sampling(request["challenge_seed"], draws, policy, row["samples_ms"] if phase == "performance" else [1.0] * 100)
    if phase == "performance":
        for row in union["rows"]:
            row["paired_reference"]["case_id"] = row["case"]["case_id"]
    request_path, union_path, output = tmp_path / "request.json", tmp_path / "union.json", tmp_path / "full.json"
    request_path.write_text(cp.canonical(request)); union_path.write_text(cp.canonical(union))
    subprocess.run([os.sys.executable, "-I", "-B", str(ROOT / "src/tools/case_finalize.py"),
        "--task", str(task), "--request", str(request_path), "--union", str(union_path), "--output", str(output)], check=True)
    report = json.loads(output.read_text())
    task_contract.validate_report(report, manifest, request)
    assert len(report["compiled_specializations"]) == len(manifest["cases"])
    if phase == "performance":
        assert len(report["test_cases"]) == len(manifest["cases"])
        if "PAIRED-REFERENCE.json" in [p.name for p in (task / "provenance").iterdir()]:
            assert len(report["paired_reference_comparison"]["cases"]) == len(manifest["cases"])
