#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task-local v2 actions. The caller selects a role; a missing HIP candidate never falls back."""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError(f"Task path escapes workspace: {value}")
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe_inputs(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride())}
    if isinstance(value, (list, tuple)):
        return [describe_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: describe_inputs(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported input metadata type: {type(value).__name__}")


def output_contract(expected, actual):
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise ValueError("Candidate output must be a tensor")
        if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.device != actual.device:
            raise ValueError("Candidate output shape/dtype/device differs from reference")
        if not torch.isfinite(actual).all():
            raise ValueError("Candidate output contains nonfinite values")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError("Candidate output dictionary contract differs")
        for key in expected:
            output_contract(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise ValueError("Candidate output sequence contract differs")
        for left, right in zip(expected, actual):
            output_contract(left, right)
    elif type(expected) is not type(actual):
        raise ValueError("Candidate scalar output type differs")
    elif isinstance(actual, float) and not math.isfinite(actual):
        raise ValueError("Candidate scalar output is nonfinite")


def case_rows(args):
    manifest = json.loads(local_path(args.workloads).read_text())
    rows = manifest["cases"]
    if not rows or len({row["test_case_id"] for row in rows}) != len(rows):
        raise ValueError("Task workload manifest is empty or has duplicate cases")
    return copy.deepcopy(rows)


def has_implementation(path):
    if not path.is_file():
        return False
    import re
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
    return bool(text.strip())


def compile_hip(path, slot="candidate"):
    if not has_implementation(path):
        raise ValueError("HIP candidate is missing or unimplemented; no baseline fallback")
    import torch
    if not torch.version.hip:
        raise RuntimeError("Native HIP compilation requires the ROCm PyTorch runtime")
    from torch.utils.cpp_extension import load
    identity = hashlib.sha256(path.read_bytes()).hexdigest()[:16] + "_" + slot
    build = ROOT / "build" / "v2_extensions" / identity
    build.mkdir(parents=True, exist_ok=True)
    # Compile the declared source in place: quoted includes retain their directory.
    extension = load(name="arena_hip_" + identity, sources=[str(path)],
                     build_directory=str(build), with_cuda=True, verbose=True)
    entry = getattr(extension, "forward", None)
    if not callable(entry):
        raise ValueError("Compiled HIP extension must export callable forward")
    return entry


def prepare_models(args, device="cuda"):
    import torch
    import correctness_check as checks
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    module = checks.load_modu_obj(str(local_path(args.module)), args.model_class, "get_init_inputs").to(device)
    functional = checks.load_func_obj(str(local_path(args.functional)), args.model_class, "get_init_inputs").to(device)
    ok, alignment = checks._align_state_dict(module, functional)
    if not ok:
        raise ValueError(f"Functional/reference state alignment failed: {alignment}")
    module.eval()
    functional.eval()
    return module, functional


def check_case_identity(row, inputs):
    if describe_inputs(inputs) != row["params"]["inputs"]:
        raise ValueError(f"Input generator no longer matches manifest: {row['test_case_id']}")


def validate_task(args, rows):
    import torch
    module = load_module(local_path(args.module), "arena_reference")
    functional = load_module(local_path(args.functional), "arena_functional")
    forward = getattr(functional, args.model_class).forward
    default = inspect.signature(forward).parameters["fn"].default
    if not callable(default):
        raise ValueError("Protected functional reference must provide a callable forward default")
    # Metadata enumeration does not allocate the largest GPU workloads or run a candidate.
    with torch.device("meta"):
        import correctness_check as checks
        cases = checks._normalize_get_inputs_result(module.get_inputs())
        count = 0
        for index, inputs in enumerate(cases):
            if index >= len(rows):
                raise ValueError("Input generator has undeclared cases")
            check_case_identity(rows[index], inputs)
            count += 1
    if count != len(rows):
        raise ValueError("Input generator omitted declared cases")
    actual_state = "implemented" if has_implementation(local_path(args.candidate)) else "unimplemented"
    if actual_state != args.initial_state:
        raise ValueError(f"Initial candidate state is {actual_state}, expected {args.initial_state}")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Task needs a ROCm PyTorch runtime and a visible GPU")
    return {"candidate_state": actual_state, "case_count": len(rows)}


def correctness(args, role, rows):
    import torch
    import correctness_check as checks
    source = local_path(args.baseline_hip) if role == "baseline" and args.baseline_hip else local_path(args.candidate)
    hip_fn = None if role == "baseline" and not args.baseline_hip else compile_hip(source)
    input_func = checks.load_function_from_path(str(local_path(args.module)), "get_inputs")
    inputs_gen = checks._normalize_get_inputs_result(input_func())
    module, functional = prepare_models(args)
    tolerance = inspect.signature(checks.correctness_check).parameters
    rtol, atol = tolerance["rtol"].default, tolerance["atol"].default
    result = []
    for index, inputs in enumerate(inputs_gen):
        if index >= len(rows):
            raise ValueError("Correctness generated undeclared cases")
        check_case_identity(rows[index], inputs)
        inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
        reference_inputs = [value.to("cuda") if isinstance(value, torch.Tensor) else value for value in inputs]
        torch.manual_seed(1337 + index)
        torch.cuda.manual_seed_all(1337 + index)
        expected = module(*copy.deepcopy(reference_inputs))
        torch.manual_seed(1337 + index)
        torch.cuda.manual_seed_all(1337 + index)
        # A provided PyTorch baseline is checked against the independently written
        # functional reference. HIP always uses the explicitly loaded extension.
        actual = (functional(*copy.deepcopy(reference_inputs)) if hip_fn is None else
                  functional(*copy.deepcopy(reference_inputs), fn=hip_fn))
        torch.cuda.synchronize()
        output_contract(expected, actual)
        passed = checks._compare_results(expected, actual, rtol=rtol, atol=atol)
        row = {**rows[index], "status": "PASS" if passed else "FAIL", "metrics": {"rtol": rtol, "atol": atol}}
        if not passed:
            row["failure_kind"] = "numerical_mismatch"
        result.append(row)
    if len(result) != len(rows):
        raise ValueError("Correctness omitted declared cases")
    return result


def performance(args, role, rows):
    import cal_kernel_perf as perf
    # Keep the original case generation, seed schedule, module state alignment,
    # warmups, sampling, reset callbacks, and protected timing helper calls.
    reports = []
    original_compare = perf._compare_results
    def checked_compare(expected, actual, **kwargs):
        output_contract(expected, actual)
        return original_compare(expected, actual, **kwargs)
    perf._compare_results = checked_compare
    from replay_validation import install
    install(perf, output_contract)
    perf._write_perf_report = lambda report: reports.append(copy.deepcopy(report))
    # Legacy helper internals derive class names from the HIP filename. Bind
    # the explicit task declaration, including when timing the _ref HIP file.
    # Compile original declared files in place. The old helper's scratch
    # basename copies must not lose quoted includes or nested source paths.
    def load_declared_kernel(_name, directory, _filename):
        reference = bool(args.baseline_hip and Path(directory).name == "hip_ref")
        source = args.baseline_hip if reference or (role == "baseline" and args.baseline_hip) else args.candidate
        return compile_hip(local_path(source), slot="perf_reference" if reference else "perf_selected")
    perf.load_hip_kernel = load_declared_kernel
    original_module_loader = perf.load_modu_obj
    original_functional_loader = perf.load_func_obj
    perf.load_modu_obj = lambda path, _class, init: original_module_loader(path, args.model_class, init)
    perf.load_func_obj = lambda path, _class, init: original_functional_loader(path, args.model_class, init)
    original_loader = perf.load_function_from_path
    seen = []
    def tracked_loader(path, symbol):
        value = original_loader(path, symbol)
        if symbol != "get_inputs":
            return value
        def generate():
            inputs = value()
            if isinstance(inputs, (list, tuple)):
                inputs = iter([inputs])
            for index, case in enumerate(inputs):
                if index >= len(rows):
                    raise ValueError("Performance generated undeclared cases")
                check_case_identity(rows[index], case)
                seen.append(index)
                yield case
        return generate
    perf.load_function_from_path = tracked_loader
    # The unimplemented target's baseline originally uses graph timing. Final
    # candidates cannot choose a weaker paired policy from their source text.
    if not args.baseline_hip:
        perf.hip_source_graph_capture_policy = lambda *paths: (True, None)
    with tempfile.TemporaryDirectory(prefix="v2-perf-", dir=ROOT / "build") as build:
        candidate = args.baseline_hip if role == "baseline" and args.baseline_hip else args.candidate
        kwargs = dict(build_dir=build, auto_cleanup=False)
        if args.baseline_hip:
            perf.cal_kernel_perf(str(local_path(args.module)), str(local_path(args.functional)),
                                 str(local_path(candidate)), str(local_path(args.baseline_hip)), **kwargs)
        else:
            perf.cal_kernel_perf(str(local_path(args.module)), str(local_path(args.functional)),
                                 str(local_path(candidate)), baseline_only=(role == "baseline"), **kwargs)
    if len(reports) != 1 or reports[0].get("status") not in ("ok", "baseline_ok"):
        raise RuntimeError("Original benchmark rejected the execution: " + str(reports[-1].get("message") if reports else "no result"))
    measured = reports[0]["test_cases"]
    if len(seen) != len(rows) or len(measured) != len(rows):
        raise RuntimeError("Benchmark omitted cases; partial baseline timing is not accepted")
    result = []
    for index, case in enumerate(measured):
        if case.get("case_idx") != index or not case.get("correct"):
            raise RuntimeError("Benchmark case identity/correctness is invalid")
        time_key = ("ref_time" if args.baseline_hip else "ori_time") if role == "baseline" else "opt_time"
        elapsed = case.get(time_key)
        method = case.get("reference_benchmark_method") if role == "baseline" and args.baseline_hip else case.get("benchmark_method")
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Benchmark returned invalid device timing")
        if method not in ("cuda_graph", "cuda_event_fallback"):
            raise RuntimeError("Benchmark did not establish device timing method")
        if (case.get("benchmark_method_consistent") is False
                or (role == "candidate" and case.get("reference_benchmark_method") != method)):
            raise RuntimeError("Baseline/candidate benchmark methods are incomparable")
        result.append({**rows[index], "status": "PASS", "execution_time_ms": elapsed,
                       "benchmark_method": method, "metadata": {"original_benchmark": case}})
    return result


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--functional", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--baseline-hip")
    parser.add_argument("--initial-state", choices=("implemented", "unimplemented"), required=True)
    parser.add_argument("--workloads", default="workload.json")
    parser.add_argument("operation", nargs="+")
    args = parser.parse_args(argv)
    role, action = ("task", "validate-task") if args.operation == ["validate-task"] else tuple(args.operation)
    rows = []
    report = {"protocol": "arena-eval-v1", "role": role, "action": action, "status": "FAIL", "cases": []}
    try:
        if (role, action) not in {( "task", "validate-task"), *( (r,a) for r in ("baseline","candidate") for a in ("compile","correctness","performance"))}:
            raise ValueError("Unsupported role/action")
        os.chdir(ROOT)
        (ROOT / "build").mkdir(exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "build" / "torch_extensions")
        rows = case_rows(args)
        if role == "candidate" and not has_implementation(local_path(args.candidate)):
            raise ValueError("HIP candidate is unimplemented; final actions cannot use a baseline")
        if action == "validate-task":
            report["metadata"] = validate_task(args, rows)
            report["cases"] = [{**row, "checks": ["correctness", "performance"], "status": "PASS"} for row in rows]
        elif action == "compile":
            if role == "baseline" and not args.baseline_hip:
                for path in (args.module, args.functional):
                    compile(local_path(path).read_text(), path, "exec")
                    load_module(local_path(path), "arena_compile_" + Path(path).stem)
            else:
                compile_hip(local_path(args.baseline_hip if role == "baseline" else args.candidate))
        else:
            report["cases"] = correctness(args, role, rows) if action == "correctness" else performance(args, role, rows)
        if any(row["status"] != "PASS" for row in report["cases"]):
            report.update(reason="Numerical comparison failed", failure_kind="numerical_mismatch")
        else:
            report["status"] = "PASS"
    except Exception as exc:
        report.update(reason=f"{type(exc).__name__}: {exc}", failure_kind="execution_error")
        report["cases"] = [] if action == "compile" else [
            {**row, "status": "FAIL", "failure_kind": "not_completed",
             **({"checks": ["correctness", "performance"]} if action == "validate-task" else {})}
            for row in rows]
    print("ARENA_EVAL_RESULT=" + json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    # Even argument/dependency errors belong to this invocation's envelope.
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        return run(tokens)
    except (Exception, SystemExit) as exc:
        role = tokens[-2] if len(tokens) >= 2 and tokens[-2] in ("baseline", "candidate") else "task"
        action = tokens[-1] if role != "task" else "validate-task"
        result = {"protocol": "arena-eval-v1", "role": role, "action": action,
                  "status": "FAIL", "cases": [], "failure_kind": "execution_error",
                  "reason": f"{type(exc).__name__}: {exc}"}
        print("ARENA_EVAL_RESULT=" + json.dumps(result, allow_nan=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
