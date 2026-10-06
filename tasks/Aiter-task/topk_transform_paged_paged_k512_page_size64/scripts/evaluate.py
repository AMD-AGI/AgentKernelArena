#!/usr/bin/env python3
"""The seven task-owned arena-eval-v1 actions. See the task README."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import task_contract

# The framework materializes its canonical benchmark helper at workspace root.
sys.path.insert(0, str(task_contract.ROOT))

# The paged top-k v2 kernel source, relative to the installed sglang package.
KERNEL_SOURCE = "kernels/jit/csrc/deepseek_v4/topk_v2.cuh"


def load_candidate(config):
    entry = task_contract.candidate_entry(config)
    path = task_contract.task_path(entry["file"])
    source = path.read_text()
    compile(source, str(path), "exec")  # Real Python syntax check, no text heuristic.
    task_contract.assert_source_independent(source)
    if not any(isinstance(node, ast.FunctionDef) and node.name == entry["symbol"]
               for node in ast.parse(source).body):
        raise RuntimeError(f"Candidate is unimplemented: {entry['symbol']} is missing")
    spec = importlib.util.spec_from_file_location("_sikl_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import candidate {entry['file']}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    builder = getattr(module, entry["symbol"], None)
    if not callable(builder):
        raise RuntimeError("Declared candidate builder is not callable")
    return builder


def check_initial_state(config):
    entry = task_contract.candidate_entry(config)
    path = task_contract.task_path(entry["file"], must_exist=False)
    if not path.exists():
        return "unimplemented"  # A missing declared generation target.
    tree = ast.parse(path.read_text())
    if any(isinstance(node, ast.FunctionDef) and node.name == entry["symbol"] for node in tree.body):
        if os.environ.get("ARENA_EVAL_PHASE", "candidate_evaluation") == "task_validation":
            raise RuntimeError("Initial candidate defines its builder but is declared unimplemented")
        return "implemented"
    # Shipped stubs have only a docstring and __future__ imports. Do not mistake
    # arbitrary broken candidate code for an initial generation target.
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        raise RuntimeError("Declared unimplemented initial state contains executable candidate code")
    return "unimplemented"


def require_runtime(workload):
    import torch
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("This task requires a ROCm GPU runtime")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"This task requires gfx950; found {arch}")
    importlib.import_module("flydsl")
    import task_baseline
    try:
        task_baseline.resolve_transform()
    except ImportError as error:
        raise RuntimeError("The runtime's sglang lacks the paged top-k v2 entry point "
                           f"({task_baseline.TRANSFORM_MODULE}): {error}") from error


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_evidence(workload):
    import torch
    import sglang
    import task_baseline
    versions = {"torch": torch.__version__, "rocm": torch.version.hip}
    for name in ("sglang", "flydsl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = str(getattr(importlib.import_module(name), "__version__", "unreported"))
    module = importlib.import_module(task_baseline.TRANSFORM_MODULE)
    source = Path(module.__file__).resolve()
    package = Path(sglang.__file__).resolve().parent
    if source.is_relative_to((task_contract.ROOT / "sglang_source").resolve()):
        raise RuntimeError("Materialized source shadows the installed sglang baseline")
    kernel = package / KERNEL_SOURCE
    name, _ = task_baseline.resolve_transform()
    return {"runtime_versions": versions, "sglang_package": str(package),
            "baseline_module": task_baseline.TRANSFORM_MODULE, "baseline_source": str(source),
            "baseline_source_sha256": _sha256(source), "baseline_entry_point": name,
            "baseline_kernel_source": str(kernel),
            "baseline_kernel_source_sha256": _sha256(kernel) if kernel.is_file() else None,
            "device": torch.cuda.get_device_name(0),
            "arch": torch.cuda.get_device_properties(0).gcnArchName,
            "nonfinite_metric_encoding": "null represents a nonfinite diagnostic, never a tolerance override"}


def dispatch_evidence(workload, case):
    import task_baseline
    name, _ = task_baseline.resolve_transform()
    return {"dispatch": f"{task_baseline.TRANSFORM_MODULE}.{name}"}


def compile_case(case, *, role, launch, measure, reuse):
    # Execute one real specialization per case. A builder may lazily compile on
    # launch; import-only checks cannot prove either baseline or candidate runs.
    inputs, out = measure.case_buffers(case, reuse)
    result = measure.invoke_checked(inputs, out, measure.case_call(inputs, out, role=role, launch=launch))
    if result.get("failure_kind") == "output_contract":
        raise AssertionError(result["reason"])
    return {"status": "PASS", "metadata": {"syntax_and_runtime_specialization": True}}


def validate_case(case, measure):
    import torch
    task_inputs = measure.task_inputs
    inputs = task_inputs.build_case_inputs(case)
    evidence = task_inputs.check_case_inputs(inputs, case, thorough=True)
    expected = measure.task_reference.run(**task_inputs.call_kwargs(inputs))
    # Real workload cases still execute their reference and validate its output
    # contract. Independent known answers and comparator controls run separately.
    spec = task_inputs.OUTPUTS[task_inputs.OUTPUT_NAME]
    shape = task_inputs.declared_shape(spec, task_inputs.dimensions(case))
    if (not isinstance(expected, torch.Tensor) or expected.layout != torch.strided
            or tuple(expected.shape) != shape or expected.dtype != task_inputs.DTYPES[spec["dtype"]]
            or expected.device != inputs["scores"].device or expected.device.type != "cuda"):
        raise RuntimeError(f"Reference output violates declared shape/dtype/device: {shape}, int32, CUDA")
    if bool((expected < -1).any()):
        raise RuntimeError("Reference output contains values other than -1 padding and slots")
    lengths = inputs["seq_lens"].to(torch.int64)
    if not torch.equal((expected >= 0).sum(dim=-1), lengths.clamp(max=task_inputs.K)):
        raise RuntimeError("Reference selects a different number of slots than min(length, k)")
    return {"status": "PASS", "metadata": {"reference_output_contract_checked": True,
                                           "input_validation": evidence}}


def report_for(role, action, cases, metadata):
    failed = [row for row in cases if row["status"] == "FAIL"]
    report = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "FAIL" if failed else "PASS", "cases": cases, "metadata": metadata}
    if failed:
        kinds = {row.get("failure_kind", "execution_error") for row in failed}
        report.update(reason=f"{len(failed)}/{len(cases)} cases failed; see per-case evidence",
                      failure_kind=next(iter(kinds)) if len(kinds) == 1 else "multiple_failures")
    return report


def run(role, action):
    rows, metadata = [], {}
    try:
        config = task_contract.load_config()
        workload = task_contract.load_workload(config)
        rows = task_contract.case_manifest(workload)
        if action != "validate-task":
            rows = [{k: v for k, v in row.items() if k != "checks"} for row in rows]
        builder = load_candidate(config) if role == "candidate" else None
        if action == "validate-task":
            metadata["candidate_state"] = check_initial_state(config)
            for source in config["baseline"]["source_files"]:
                task_contract.task_path(source)
        require_runtime(workload)
        metadata.update(runtime_evidence(workload))
        import task_measure as measure
        if action == "validate-task":
            import task_validation
            metadata["validation_controls"] = []
            task_validation.run_controls(measure, records=metadata["validation_controls"])
        # One launch serves every case of a shape, and correctness reuses one
        # set of buffers per shape, in manifest order of growing lengths.
        launches, reuse = {}, {}
        for row, case in zip(rows, workload["cases"]):
            try:
                launch = None
                if role == "candidate":
                    key = tuple(sorted(measure.builder_axes(case).items()))
                    if key not in launches:
                        launches.clear()
                        launches[key] = measure.build_launch(builder, case)
                    launch = launches[key]
                if action == "validate-task":
                    result = validate_case(case, measure)
                elif action == "compile":
                    if role == "baseline":
                        path = Path(measure.task_baseline.__file__)
                        compile(path.read_text(), str(path), "exec")
                    result = compile_case(case, role=role, launch=launch, measure=measure, reuse=reuse)
                elif action == "correctness":
                    result = measure.check_case(case, role=role, launch=launch, reuse=reuse)
                else:
                    result = measure.time_case(case, role=role, launch=launch,
                                              baseline_diagnostic=config["baseline"].get("correctness_policy") == "diagnostic")
                row.update(result)
                if role == "baseline":
                    row.setdefault("metadata", {}).update(dispatch_evidence(workload, case))
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                row.update(status="FAIL", failure_kind="execution_error", reason=f"{type(error).__name__}: {error}")
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        reason = f"{type(error).__name__}: {error}"
        for row in rows:
            row.update(status="FAIL", failure_kind="execution_error", reason=reason)
        if not rows:
            return {"protocol": "arena-eval-v1", "role": role, "action": action,
                    "status": "FAIL", "cases": [], "reason": reason, "failure_kind": "execution_error"}
    return report_for(role, action, rows, metadata)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["validate-task"]:
        role, action = "task", "validate-task"
    elif len(argv) == 2 and argv[0] in ("baseline", "candidate") and argv[1] in ("compile", "correctness", "performance"):
        role, action = argv
    else:
        print("usage: evaluate.py validate-task | {baseline,candidate} {compile,correctness,performance}", file=sys.stderr)
        return 2
    import json
    report = task_contract.json_safe(run(role, action))
    print("ARENA_EVAL_RESULT=" + json.dumps(report, allow_nan=False, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
