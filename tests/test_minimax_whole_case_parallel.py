"""Exercise protected MiniMax CLI control flow with CPU-only device workers."""
from copy import deepcopy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["minimax-m3__gqa_share_sparse_fwd_kernel",
         "minimax-m3__gqa_share_sparse_decode_kernel", "minimax-m3__decode_score_kernel"]


@pytest.fixture(params=TASKS)
def runner(request, monkeypatch, tmp_path):
    task = ROOT / "tasks/headkernel" / request.param
    monkeypatch.syspath_prepend(str(task / "ut"))
    for name in ("evaluation_contract", "served_contract", "source_guard", "minimax_native",
                 "paired_reference", "case_parallel", "workload_controls", "minimax_work"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("_minimax_case_runner", task / "scripts/task_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((task / "cases.json").read_text())
    definition = json.loads((task / "task_definition.json").read_text())
    working = tmp_path / "task"
    (working / "build").mkdir(parents=True)
    (working / "provenance").mkdir()
    (working / "config.yaml").write_bytes((task / "config.yaml").read_bytes())
    (working / "cases.json").write_bytes((task / "cases.json").read_bytes())
    (working / "provenance/PAIRED-REFERENCE.json").write_bytes((task / "provenance/PAIRED-REFERENCE.json").read_bytes())
    monkeypatch.setattr(module, "ROOT", working)
    real_tree_identity = module.tree_identity
    monkeypatch.setattr(module, "load_cases", lambda root: (manifest, definition))
    monkeypatch.setattr(module, "validate_sources", lambda *args: True)
    monkeypatch.setattr(module, "tree_identity", lambda: "a" * 64)
    monkeypatch.setattr(module, "sha", lambda path: definition["source_sha256"])
    for name in ("AKA_CASE_GPU_PLAN", "AKA_CASE_GPU_PLAN_SHA256"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(module=module, manifest=manifest, definition=definition, root=working,
                           task=task, monkeypatch=monkeypatch, real_tree_identity=real_tree_identity)


def phase_request(runner, phase):
    return {"schema_version": 1, "request_id": "CPU_TEST_REQUEST", "phase": phase,
            "manifest_sha256": runner.module.fingerprint(runner.manifest), "package_sha256": runner.module.tree_identity(),
            "source_sha256": {runner.definition["source_file"]: runner.definition["source_sha256"]},
            "challenge_seed": 317}


def payload(runner, request, cases=None):
    cases = runner.manifest["cases"] if cases is None else cases
    proof = {"source_sha256": runner.definition["source_sha256"],
             "engaged_kernels": runner.definition["kernels"], "last_launches": [{"CPU_TEST_ONLY": True}],
             "production_namespace_rebound": False, "CPU_TEST_ONLY": True}
    rows, compiled = [], []
    for case in cases:
        row = {"case": deepcopy(case), "correct": True,
               "seeds": runner.manifest["measurement"]["correctness_seeds"],
               "negative_controls": {"no_op": True, "wrong_output": True},
               "samples_ms": [2.0] * 100, "warmup_iterations": 10, "fresh_input_resets": 100,
               "output_initializations": 100, "oracle_checks": 100, "benchmark_method": "cuda_graph"}
        row["paired_reference"] = {"case_id": case["case_id"], "CPU_TEST_ONLY": True}
        if runner.definition["kind"] in ("decode_score", "sparse_decode"):
            from workload_controls import RecordedControls
            controls = RecordedControls(case)
            policy = runner.manifest["measurement"]
            row["workload_control_coverage"] = controls.coverage(
                policy, controls.targeted_variants(), controls.original_plan(policy))
            draws = [controls.choose(request["challenge_seed"] + i) for i in range(110)]
            row["workload_control_sampling"] = controls.sampling(
                request["challenge_seed"], draws, policy, row["samples_ms"])
        if request["phase"] != "compile":
            rows.append(row)
        compiled.append({"case_id": case["case_id"], "kernel_engagement": proof,
                         "candidate_launch_evidence": proof, "invoked_and_synchronized": True,
                         "source_sha256": request["source_sha256"]})
    return {"rows": rows, "compiled_specializations": compiled, "kernel_engagement": proof,
            "completed_case_ids": [c["case_id"] for c in cases]}


def argv(runner, request, *, worker=False):
    path = runner.root / "build/request.json"
    path.write_text(json.dumps(request))
    values = [str(runner.task / "scripts/task_runner.py"), request["phase"], "--request", str(path)]
    if worker:
        values.extend(["--case-worker", str(runner.root / "build/assignment.json")])
    runner.monkeypatch.setattr(sys, "argv", values)


def parallel_helper(runner, request, result=None):
    helper = ModuleType("case_parallel")
    calls = []
    def load_plan(root, manifest, supplied):
        assert manifest is runner.manifest and supplied == request
        calls.append("load_plan")
        return {"CPU_TEST_PLAN": True}
    def bind_request(supplied, plan):
        calls.append("bind_request")
        return {**supplied, "case_parallelism": {"schema_version": 1, "plan_sha256": "CPU_TEST_PLAN"},
                "gpu_group": [{"CPU_TEST_ONLY": True}]}
    def run_workers(root, manifest, supplied, plan, entrypoint, *, leg):
        assert manifest is runner.manifest and supplied["challenge_seed"] == request["challenge_seed"]
        assert supplied["manifest_sha256"] == request["manifest_sha256"] and "gpu" not in supplied
        assert leg == "framework"
        calls.append("run_workers")
        if isinstance(result, Exception):
            raise result
        union = deepcopy(result) if result is not None else payload(runner, supplied)
        return {**union, "workers": [], "case_parallelism": supplied["case_parallelism"]}
    helper.load_plan, helper.bind_request, helper.run_workers = load_plan, bind_request, run_workers
    helper.enter_worker = helper.finish_worker = lambda *args, **kwargs: pytest.fail("coordinator entered worker")
    runner.monkeypatch.setitem(sys.modules, "case_parallel", helper)
    runner.monkeypatch.setenv("AKA_CASE_GPU_PLAN", "/CPU_TEST_PLAN")
    runner.monkeypatch.setenv("AKA_CASE_GPU_PLAN_SHA256", "CPU_TEST_DIGEST")
    return helper, calls


@pytest.mark.parametrize("phase", ["compile", "correctness", "performance"])
def test_full_task_command_finalizes_complete_union_without_parent_runtime(runner, phase):
    request = phase_request(runner, phase)
    argv(runner, request)
    _, calls = parallel_helper(runner, request)
    runner.monkeypatch.setattr(runner.module, "configure_cpu_threads", lambda: pytest.fail("parent imported Torch"))
    runner.monkeypatch.setattr(runner.module, "evaluate_cases", lambda *a, **k: pytest.fail("parent ran GPU cases"))
    original = runner.module.finalize_report
    finalized = []
    def finalize(report, manifest, supplied):
        assert manifest is runner.manifest
        finalized.append(report)
        return original(report, manifest, supplied)
    runner.monkeypatch.setattr(runner.module, "finalize_report", finalize)
    assert runner.module.entrypoint() == 0
    report = json.loads((runner.root / "build" / (phase + "_report.json")).read_text())
    assert calls == ["load_plan", "bind_request", "run_workers"] and len(finalized) == 1
    assert len(report["compiled_specializations"]) == len(runner.manifest["cases"])
    if phase != "compile":
        assert [r["case"] for r in report["cases"]] == runner.manifest["cases"]
    if phase == "performance":
        assert len(report["test_cases"]) == len(runner.manifest["cases"])
        assert len(report["paired_reference_comparison"]["cases"]) == len(runner.manifest["cases"])
        assert all(r["samples_ms"] == [2.0] * 100 for r in report["cases"])


@pytest.mark.parametrize("corruption", ["missing_case", "duplicate_case", "changed_case", "99_samples", "101_samples", "missing_compile", "failed_worker"])
def test_full_task_command_rejects_partial_or_changed_results(runner, corruption):
    request = phase_request(runner, "performance")
    result = payload(runner, request)
    if corruption == "missing_case":result["rows"].pop()
    elif corruption == "duplicate_case":result["rows"][-1] = deepcopy(result["rows"][0])
    elif corruption == "changed_case":result["rows"][0]["case"]["occurrences"] += 1
    elif corruption == "99_samples":result["rows"][0]["samples_ms"].pop()
    elif corruption == "101_samples":result["rows"][0]["samples_ms"].append(2.0)
    elif corruption == "missing_compile":result["compiled_specializations"].pop()
    else:result = RuntimeError("CPU_TEST worker exited nonzero")
    argv(runner, request)
    parallel_helper(runner, request, result)
    with pytest.raises((ValueError, RuntimeError)):
        runner.module.entrypoint()
    report = json.loads((runner.root / "build/performance_report.json").read_text())
    assert report["status"] == "fail" and "test_cases" not in report


@pytest.mark.parametrize("phase", ["compile", "correctness", "performance"])
def test_serial_mode_keeps_original_public_report_and_never_loads_dispatcher(runner, phase):
    request = phase_request(runner, phase)
    argv(runner, request)
    runner.monkeypatch.setitem(sys.modules, "case_parallel", None)
    runner.monkeypatch.setattr(runner.module, "configure_cpu_threads", lambda: {"intraop_threads": 8})
    def evaluate(manifest, definition, supplied, *, cases, worker):
        assert manifest is runner.manifest and cases is None and worker is False
        return payload(runner, supplied)
    runner.monkeypatch.setattr(runner.module, "evaluate_cases", evaluate)
    assert runner.module.entrypoint() == 0
    report = json.loads((runner.root / "build" / (phase + "_report.json")).read_text())
    assert report["request"] == request and "case_parallelism" not in report
    assert ("compiled_specializations" in report) == (phase == "performance")


@pytest.mark.parametrize("phase", ["compile", "correctness", "performance"])
@pytest.mark.parametrize("failure", [False, True])
def test_worker_preflight_precedes_runtime_and_never_touches_parent_report(runner, phase, failure):
    request = phase_request(runner, phase)
    argv(runner, request, worker=True)
    public = runner.root / "build" / (phase + "_report.json")
    public.write_text("PARENT_OWNED_SENTINEL")
    private = runner.root / "build/worker-0"
    private.mkdir()
    selected = runner.manifest["cases"][::8]
    events, saved = [], []
    helper = ModuleType("case_parallel")
    def enter(root, manifest, supplied, assignment):
        assert manifest is runner.manifest and supplied == request and events == []
        events.append("preflight")
        if failure:raise ValueError("CPU_TEST wrong device")
        return SimpleNamespace(cases=selected, output_directory=private)
    helper.enter_worker = enter
    helper.bind_request = helper.load_plan = helper.run_workers = lambda *a, **k: pytest.fail("worker coordinated")
    helper.finish_worker = lambda context, **values: saved.append(values)
    runner.monkeypatch.setitem(sys.modules, "case_parallel", helper)
    def cpu():
        assert events == ["preflight"]
        events.append("cpu_runtime")
        return {"intraop_threads": 8}
    runner.monkeypatch.setattr(runner.module, "configure_cpu_threads", cpu)
    def evaluate(manifest, definition, supplied, *, cases, worker):
        assert manifest is runner.manifest and supplied == request and cases is selected and worker
        return payload(runner, supplied, cases)
    runner.monkeypatch.setattr(runner.module, "evaluate_cases", evaluate)
    runner.monkeypatch.setattr(runner.module, "complete_report", lambda *a, **k: pytest.fail("worker finalized"))
    if failure:
        with pytest.raises(ValueError, match="wrong device"):runner.module.entrypoint()
        assert not saved
    else:
        assert runner.module.entrypoint() == 0
        assert len(saved) == 1 and saved[0]["evidence"]["completed_case_ids"] == [c["case_id"] for c in selected]
        assert "test_cases" not in saved[0]
        assert (private / ("cpu_runtime_" + phase + ".json")).is_file()
    assert public.read_text() == "PARENT_OWNED_SENTINEL"


@pytest.mark.parametrize("corruption", ["empty", "duplicate", "reordered", "changed_abi"])
def test_case_selection_rejects_mutation_before_native_import(runner, corruption):
    cases = deepcopy(runner.manifest["cases"][::8])
    if corruption == "empty":cases = []
    elif corruption == "duplicate":cases.append(cases[0])
    elif corruption == "reordered":cases.reverse()
    else:cases[0]["occurrences"] += 1
    runner.monkeypatch.setitem(sys.modules, "minimax_native", None)
    with pytest.raises(ValueError):
        runner.module.evaluate_cases(runner.manifest, runner.definition, phase_request(runner, "performance"), cases=cases, worker=True)


@pytest.mark.parametrize("phase", ["compile", "correctness", "performance"])
def test_real_worker_loop_keeps_full_manifest_policy_and_original_case_objects(runner, phase):
    request = phase_request(runner, phase)
    selected = runner.manifest["cases"][::8]
    calls = []
    proof = payload(runner, request)["kernel_engagement"]
    class Operator:
        def __init__(self, root, definition):
            assert root == runner.root and definition is runner.definition
        def proof(self):return proof
    native = ModuleType("minimax_native")
    native.Operator = Operator
    runner.monkeypatch.setitem(sys.modules, "minimax_native", native)
    class Evaluation:
        def __init__(self, case, definition, operator, *, paired, paired_seed):
            assert any(case is original for original in selected)
            assert definition is runner.definition and paired == (phase == "performance")
            assert paired_seed == request["challenge_seed"] + 110
            self.case = case
            calls.append(("prepare", case["case_id"]))
            self.paired_specialization = {"case_id": case["case_id"], "candidate_launch_evidence": proof}
        def correctness(self, policy):
            assert policy is runner.manifest["measurement"]
            assert policy["correctness_seeds"] == [0, 1, 2]
            calls.append(("correctness", self.case["case_id"]))
            return payload(runner, request, [self.case])["rows"][0]
        def performance(self, policy, seed, *, manifest, request):
            assert manifest is runner.manifest and len(manifest["cases"]) > len(selected)
            assert seed == request["challenge_seed"] == 317
            assert policy is manifest["measurement"] and policy["warmup_iterations"] == 10
            assert policy["benchmark_iterations"] == 100 and self.case["calls_per_sample"] == 1
            calls.append(("performance", self.case["case_id"]))
            return payload(runner, request, [self.case])["rows"][0]
    runner.monkeypatch.setattr(runner.module, "CaseEvaluation", Evaluation)
    result = runner.module.evaluate_cases(runner.manifest, runner.definition, request, cases=selected, worker=True)
    ids = [c["case_id"] for c in selected]
    assert result["completed_case_ids"] == ids
    assert [row["case_id"] for row in result["compiled_specializations"]] == ids
    assert [case_id for event, case_id in calls if event == "prepare"] == ids
    if phase != "compile":
        assert [case_id for event, case_id in calls if event == phase] == ids
        assert len(result["rows"]) == len(selected)


@pytest.fixture
def canonical_parallel(runner):
    path = Path(os.environ.get("MINIMAX_CASE_PARALLEL_HELPER", ROOT / "src/tools/case_parallel.py"))
    if not path.exists():
        pytest.skip("Canonical whole-case helper must be integrated or explicitly supplied")
    runner.monkeypatch.syspath_prepend(str(path.parent))
    runner.monkeypatch.delitem(sys.modules, "gpu_binding", raising=False)
    spec = importlib.util.spec_from_file_location("_canonical_parallel_for_minimax", path)
    module = importlib.util.module_from_spec(spec)
    runner.monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    runner.monkeypatch.setitem(sys.modules, "case_parallel", module)
    source = runner.root / runner.definition["source_file"]
    source.parent.mkdir(parents=True)
    source.write_bytes((runner.task / runner.definition["source_file"]).read_bytes())
    runner.monkeypatch.setattr(runner.module, "tree_identity", runner.real_tree_identity)
    bindings = [{"worker_id": i, "render_device": f"/dev/dri/renderD{128 + 8*i}", "drm_minor": 128 + 8*i,
                 "kfd_node": str(i + 2), "rocr_uuid": f"GPU-{i:016x}", "pci_bus_id": f"0002:00:{i+1:02x}.0"}
                for i in range(8)]
    assignments = [{"worker_id": i, "case_ids": [c["case_id"] for c in runner.manifest["cases"][i::8]]} for i in range(8)]
    config = yaml.safe_load((runner.task / "config.yaml").read_text())
    plan = {"schema": "case-gpu-plan-v1", "authorized": True, "plan_id": "CPU_TEST_ONLY_NOT_A_REAL_GRANT",
            "job_id": "999999", "node": "CPU_TEST_ONLY", "restart_generation": 0,
            "allocation_receipt_sha256": "b" * 64, "baseline_commit": "c" * 40,
            "work_stop_unix": time.time() + 600, "task_path": "tasks/headkernel/" + runner.task.name,
            "manifest_sha256": module.fingerprint(runner.manifest), "runtime_image": runner.manifest["runtime_image"],
            "bindings": bindings, "assignments": assignments, "assignment_sha256": module.fingerprint(assignments),
            "phase_timeouts": {phase: config[phase + "_timeout"] for phase in ("compile", "correctness", "performance")},
            "resources": {"cpus_per_worker": 1, "memory_gib_per_worker": 1, "coordinator_cpus": 1,
                          "coordinator_memory_gib": 1, "total_cpus": 9, "total_memory_gib": 9},
            "allocation_capacity": {"cpus": 9, "memory_gib": 9, "gpus": 8}}
    plan_path = runner.root.parent / "CPU_TEST_PLAN.json"
    plan_path.write_text(json.dumps(plan))
    plan_path.chmod(0o444)
    runner.monkeypatch.setenv("AKA_CASE_GPU_PLAN", str(plan_path))
    runner.monkeypatch.setenv("AKA_CASE_GPU_PLAN_SHA256", hashlib.sha256(plan_path.read_bytes()).hexdigest())
    return module


@pytest.mark.parametrize("phase", ["compile", "correctness", "performance"])
def test_actual_shared_receipt_collector_and_full_parent_finalizer(runner, canonical_parallel, phase):
    """Only device execution is mocked; use real assignments, receipts and reducer."""
    helper = canonical_parallel
    request = phase_request(runner, phase)
    argv(runner, request)
    def execute_cpu_workers(root, manifest, parent, plan, entrypoint, *, leg):
        assignments = helper.prepare_assignments(root, manifest, parent, plan, leg=leg)
        receipts = []
        for assignment in assignments:
            directory = root / assignment["output_directory"]
            assignment_path = directory / "assignment.json"
            assignment_path.write_text(json.dumps(assignment))
            expected = assignment["expected_gpu"]
            with runner.monkeypatch.context() as patch:
                patch.setenv("ROCR_VISIBLE_DEVICES", expected["rocr_uuid"])
                for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
                    patch.setenv(name, "0")
                patch.setattr(helper.gpu_binding, "runtime_identity", lambda: {
                    "hsa_gpu_uuids": [expected["rocr_uuid"]], "hip_device_count": 1,
                    "hip_pci_bus_ids": [expected["pci_bus_id"]]})
                context = helper.enter_worker(root, manifest, parent, assignment_path)
                result = payload(runner, parent, context.cases)
                helper.finish_worker(context, rows=result["rows"], compiled_specializations=result["compiled_specializations"],
                                     evidence={"completed_case_ids": result["completed_case_ids"]})
            receipts.append(directory / "receipt.json")
        return helper.collect_workers(manifest, parent, plan, assignments, [0] * 8, receipts)
    runner.monkeypatch.setattr(helper, "run_workers", execute_cpu_workers)
    runner.monkeypatch.setattr(runner.module, "configure_cpu_threads", lambda: pytest.fail("coordinator constructed runtime"))
    assert runner.module.entrypoint() == 0
    report = json.loads((runner.root / "build" / (phase + "_report.json")).read_text())
    helper.validate_group_report(runner.root, runner.manifest, report, required=True)
    assert len(report["case_parallelism"]["workers"]) == 8
    assert len(report["case_parallelism"]["case_gpu_mapping"]) == len(runner.manifest["cases"])
    assert len(report["compiled_specializations"]) == len(runner.manifest["cases"])
