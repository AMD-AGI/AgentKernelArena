"""The coordinator only returns complete original cases; no GPU is simulated as evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pytest

from src import task_contract
from src.tools import case_parallel as cp


ROOT = Path(__file__).resolve().parents[1]


def manifest_for(count=4):
    tensor = {"role": "input", "shape": [2], "strides": [1], "dtype": "float32",
              "storage_offset": 0, "device_type": "cuda"}
    return {"schema_version": 1, "runtime_image": "registry/image@sha256:" + "a" * 64,
        "cases": [{"case_id": f"case-{i}", "occurrences": 1, "calls_per_sample": 1,
                   "tensors": {"x": tensor, "y": {**tensor, "role": "output"}}, "scalars": {}} for i in range(count)],
        "measurement": {"method": "cuda_graph", "warmup_iterations": 10, "benchmark_iterations": 100,
                        "correctness_seeds": [0, 1, 2], "negative_controls": ["no_op", "wrong_output"],
                        "refresh_inputs": "each_replay", "initialize_outputs": "each_replay", "validate_outputs": "each_replay"}}


def plan_for(manifest, count=2):
    assignments = [{"worker_id": i, "case_ids": [c["case_id"] for c in manifest["cases"][i::count]]} for i in range(count)]
    return {"schema": "case-gpu-plan-v1", "authorized": True, "plan_id": "fresh-plan", "job_id": "12345",
            "node": "fresh-node", "restart_generation": 0, "allocation_receipt_sha256": "b" * 64,
            "baseline_commit": "c" * 40, "work_stop_unix": time.time() + 120,
            "task_path": "tasks/test/example", "manifest_sha256": cp.fingerprint(manifest),
            "runtime_image": manifest["runtime_image"], "assignments": assignments,
            "assignment_sha256": cp.fingerprint(assignments),
            "bindings": [{"worker_id": i, "render_device": f"/dev/dri/renderD{128+i}", "drm_minor": 128+i,
                          "rocr_uuid": f"GPU-{i+1:016x}", "pci_bus_id": f"0000:{i+1:02x}:00.0", "kfd_node": str(i+1)} for i in range(count)],
            "resources": {"cpus_per_worker": 1, "memory_gib_per_worker": 1, "coordinator_cpus": 1, "coordinator_memory_gib": 1},
            "allocation_capacity": {"cpus": count+1, "memory_gib": count+1, "gpus": count}}


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.delenv(cp.PLAN_ENV, raising=False)
    monkeypatch.delenv(cp.HASH_ENV, raising=False)
    root = tmp_path / "task"
    root.mkdir()
    (root / "source").mkdir()
    (root / "source/kernel.py").write_text("# inert CPU test source\n")
    manifest = manifest_for()
    (root / "cases.json").write_text(cp.canonical(manifest))
    request = {"schema_version": 1, "request_id": "fresh-request", "phase": "performance",
               "manifest_sha256": cp.fingerprint(manifest), "package_sha256": "d" * 64,
               "source_sha256": {"source/kernel.py": hashlib.sha256((root / "source/kernel.py").read_bytes()).hexdigest()},
               "challenge_seed": 7, "gpu": {"old": "serial"}}
    plan = plan_for(manifest)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    path.chmod(0o444)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = cp.load_plan(root, manifest, request, path=path, sha256=digest)
    return root, manifest, cp.bind_request(request, loaded), loaded


def row_for(case, phase="performance"):
    if phase == "correctness":
        return {"case": case, "correct": True, "seeds": [0, 1, 2],
                "negative_controls": {"no_op": True, "wrong_output": True}}
    return {"case": case, "correct": True, "samples_ms": [1.0] * 100,
            "warmup_iterations": 10, "fresh_input_resets": 100, "output_initializations": 100,
            "oracle_checks": 100, "benchmark_method": "cuda_graph",
            "paired_reference": {"raw_nested_structure": [1, 2, 3]}}


def make_receipts(package):
    root, manifest, request, plan = package
    assignments = cp.prepare_assignments(root, manifest, request, plan, leg="framework")
    paths = []
    for assignment in assignments:
        cases = tuple(c for c in manifest["cases"] if c["case_id"] in assignment["case_ids"])
        gpu = assignment["expected_gpu"]
        context = cp.WorkerContext(root, manifest, request, assignment, cases, root / assignment["output_directory"],
             {"hsa_gpu_uuids": [gpu["rocr_uuid"]], "hip_device_count": 1,
              "hip_pci_bus_ids": [gpu["pci_bus_id"]], "rocr_visible_devices": gpu["rocr_uuid"]})
        compiled = [{"case_id": c["case_id"], "source_sha256": request["source_sha256"],
                     "candidate_invoked": True, "independent_reference_invoked": True} for c in cases]
        cp.finish_worker(context, rows=[] if request["phase"] == "compile" else [row_for(c, request["phase"]) for c in cases],
                         compiled_specializations=compiled, evidence={"completed_case_ids": assignment["case_ids"]})
        paths.append(context.output_directory / "receipt.json")
    return assignments, paths


def test_no_plan_preserves_serial_request(tmp_path, monkeypatch):
    monkeypatch.delenv(cp.PLAN_ENV, raising=False)
    monkeypatch.delenv(cp.HASH_ENV, raising=False)
    request = {"gpu": {"serial": True}}
    assert cp.load_plan(tmp_path, {}, request) is None
    assert request == {"gpu": {"serial": True}}


def test_binding_preserves_original_parent_fields(package):
    _, manifest, request, plan = package
    assert "gpu" not in request
    assert request["package_sha256"] == "d" * 64
    assert request["manifest_sha256"] == cp.fingerprint(manifest)
    assert cp.bind_request(request, plan) == request


@pytest.mark.parametrize("count", [69, 40, 24, 16])
def test_original_full_domain_partition(count):
    manifest = manifest_for(count)
    plan = plan_for(manifest, count=8)
    cp.validate_plan(plan, manifest)
    assert sum(len(a["case_ids"]) for a in plan["assignments"]) == count


@pytest.mark.parametrize("attack", ["unauthorized", "expired", "duplicate_gpu", "wrong_uuid", "missing", "duplicate",
                                    "order", "foreign_manifest", "capacity", "headroom", "boolean_worker"])
def test_plan_rejects_incomplete_or_foreign_grants(attack):
    manifest = manifest_for()
    plan = plan_for(manifest)
    if attack == "unauthorized": plan["authorized"] = False
    elif attack == "expired": plan["work_stop_unix"] = time.time() - 1
    elif attack == "duplicate_gpu": plan["bindings"][1]["rocr_uuid"] = plan["bindings"][0]["rocr_uuid"]
    elif attack == "wrong_uuid": plan["bindings"][0]["rocr_uuid"] = "0"
    elif attack == "missing": plan["assignments"][0]["case_ids"].pop()
    elif attack == "duplicate": plan["assignments"][1]["case_ids"][0] = plan["assignments"][0]["case_ids"][0]
    elif attack == "order": plan["assignments"][0]["case_ids"].reverse()
    elif attack == "foreign_manifest": plan["manifest_sha256"] = "f" * 64
    elif attack == "capacity": plan["allocation_capacity"]["cpus"] = 2
    elif attack == "headroom": del plan["resources"]["coordinator_memory_gib"]
    else: plan["bindings"][0]["worker_id"] = False
    plan["assignment_sha256"] = cp.fingerprint(plan["assignments"])
    with pytest.raises(ValueError):
        cp.validate_plan(plan, manifest)


@pytest.mark.parametrize("phase", ["compile", "correctness", "performance"])
def test_full_union_retains_raw_rows_and_real_finalizer(package, phase):
    root, manifest, request, plan = package
    request["phase"] = phase
    assignments, paths = make_receipts(package)
    result = cp.collect_workers(manifest, request, plan, assignments, [0, 0], paths)
    report = {"schema_version": 1, "status": "ok", "compiled": True, "request": request,
              "cases": result["rows"], "compiled_specializations": result["compiled_specializations"],
              "case_parallelism": result["case_parallelism"]}
    cp.validate_group_report(root, manifest, report)
    full = task_contract.finalize_report(report, manifest, request)
    assert [row["case_id"] for row in full["compiled_specializations"]] == [c["case_id"] for c in manifest["cases"]]
    if phase == "performance":
        assert len(full["test_cases"]) == 4
        assert full["cases"][0]["paired_reference"]["raw_nested_structure"] == [1, 2, 3]


@pytest.mark.parametrize("attack", ["nonce", "source", "package", "tree", "request", "gpu", "omitted",
                                    "duplicate", "compile", "scoreable", "gain", "exit", "missing", "assignment"])
def test_collector_rejects_incomplete_stale_or_scoreable_workers(package, attack):
    _, manifest, request, plan = package
    assignments, paths = make_receipts(package)
    receipt = json.loads(paths[0].read_text())
    exits = [0, 0]
    if attack == "nonce": receipt["worker_nonce"] = "0" * 48
    elif attack == "source": receipt["source_sha256"] = {"source/kernel.py": "0" * 64}
    elif attack == "package": receipt["package_after_sha256"] = "0" * 64
    elif attack == "tree": receipt["input_tree_after_sha256"] = "0" * 64
    elif attack == "request": receipt["parent_request"]["challenge_seed"] += 1
    elif attack == "gpu": receipt["gpu_preflight"]["hip_device_count"] = 2
    elif attack == "omitted": receipt["rows"].pop()
    elif attack == "duplicate": receipt["rows"][1] = receipt["rows"][0]
    elif attack == "compile": receipt["compiled_specializations"] = []
    elif attack == "scoreable": receipt["scoreable"] = True
    elif attack == "gain": receipt["accepted_gain"] = 10
    elif attack == "exit": exits[1] = 137
    elif attack == "assignment": assignments[0]["case_ids"] = assignments[0]["case_ids"][:1]
    if attack == "missing": paths[0].unlink()
    else: paths[0].write_text(cp.canonical(receipt))
    with pytest.raises(ValueError):
        cp.collect_workers(manifest, request, plan, assignments, exits, paths)


@pytest.mark.parametrize("attack", ["99_samples", "101_samples", "seeds", "control"])
def test_full_numerical_validator_still_rejects_weakened_case_work(package, attack):
    _, manifest, request, plan = package
    request["phase"] = "correctness" if attack in ("seeds", "control") else "performance"
    assignments, paths = make_receipts(package)
    receipt = json.loads(paths[0].read_text())
    if attack == "99_samples": receipt["rows"][0]["samples_ms"].pop()
    elif attack == "101_samples": receipt["rows"][0]["samples_ms"].append(1.0)
    elif attack == "seeds": receipt["rows"][0]["seeds"].pop()
    else: receipt["rows"][0]["negative_controls"]["no_op"] = False
    paths[0].write_text(cp.canonical(receipt))
    result = cp.collect_workers(manifest, request, plan, assignments, [0, 0], paths)
    with pytest.raises(ValueError):
        task_contract.finalize_report({"schema_version": 1, "status": "ok", "request": request,
                                       "cases": result["rows"]}, manifest, request)


def test_worker_tree_mutation_and_resume_rejected(package):
    root, manifest, request, plan = package
    make_receipts(package)
    with pytest.raises(ValueError, match="resume"):
        cp.prepare_assignments(root, manifest, request, plan, leg="framework")
    before = cp.input_tree_identity(root)
    (root / "source/kernel.py").write_text("# changed\n")
    assert cp.input_tree_identity(root) != before


CPU_WORKER = '''
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'ut'))
import case_parallel as cp
p=argparse.ArgumentParser(); p.add_argument('phase'); p.add_argument('--request'); p.add_argument('--case-worker'); a=p.parse_args()
root=Path(__file__).resolve().parent
manifest=json.loads((root/'cases.json').read_text()); request=json.loads(Path(a.request).read_text())
assignment=json.loads(Path(a.case_worker).read_text()); gpu=assignment['expected_gpu']
# Only this CPU fixture replaces hardware identity. Production helpers have no override.
cp.gpu_binding.runtime_identity=lambda: {'hsa_gpu_uuids':[gpu['rocr_uuid']], 'hip_device_count':1,'hip_pci_bus_ids':[gpu['pci_bus_id']]}
context=cp.enter_worker(root,manifest,request,a.case_worker)
rows=[{'case': c,'correct':True,'samples_ms':[1.0]*100,'warmup_iterations':10,'fresh_input_resets':100,'output_initializations':100,'oracle_checks':100,'benchmark_method':'cuda_graph'} for c in context.cases]
compiled=[{'case_id': c['case_id'],'source_sha256':request['source_sha256'],'candidate_invoked':True,'independent_reference_invoked':True} for c in context.cases]
cp.finish_worker(context,rows=rows,compiled_specializations=compiled,evidence={'completed_case_ids':assignment['case_ids'],'pid':os.getpid(),'tmpdir':os.environ['TMPDIR']})
'''


def test_actual_fresh_process_dispatch_and_full_finalizer(package):
    root, manifest, request, plan = package
    (root / "ut").mkdir()
    for name in ("case_parallel.py", "gpu_binding.py"):
        shutil.copyfile(ROOT / "src/tools" / name, root / "ut" / name)
    entrypoint = root / "worker.py"
    entrypoint.write_text(CPU_WORKER)
    result = cp.run_workers(root, manifest, request, plan, entrypoint, leg="framework")
    report = {"schema_version": 1, "status": "ok", "request": request, "cases": result["rows"],
              "compiled_specializations": result["compiled_specializations"], "case_parallelism": result["case_parallelism"]}
    cp.validate_group_report(root, manifest, report)
    full = task_contract.finalize_report(report, manifest, request)
    assert len(full["test_cases"]) == 4
    receipts = [json.loads((root / w["receipt_path"]).read_text()) for w in result["workers"]]
    assert len({r["evidence"]["pid"] for r in receipts}) == 2
    assert all(r["evidence"]["pid"] != os.getpid() for r in receipts)
    assert len({r["evidence"]["tmpdir"] for r in receipts}) == 2


def test_failure_cancels_owned_sibling_process_group(package):
    root, manifest, request, plan = package
    entrypoint = root / "fails.py"
    entrypoint.write_text("import argparse,json,time,sys\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('--request');p.add_argument('--case-worker');a=p.parse_args()\n"
        "w=json.loads(Path(a.case_worker).read_text())['worker_id']\n"
        "if w==0: time.sleep(.3);sys.exit(9)\n"
        "time.sleep(10)\nPath('build/should-not-exist').write_text('leaked')\n")
    start = time.monotonic()
    with pytest.raises(ValueError, match="cancelling"):
        cp.run_workers(root, manifest, request, plan, entrypoint, leg="framework")
    assert time.monotonic() - start < 5
    assert not (root / "build/should-not-exist").exists()

