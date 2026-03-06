# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import argparse
import copy
import re
import itertools
import torch
import shutil
import sys
from typing import Any, Dict, List, Tuple, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compile import clear_workdir
from utils import load_function_from_path, load_hip_kernel, save_eval_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kernel performance benchmark for HIP kernels (hip2hip) with multiple test cases.")
    parser.add_argument("--py_modu_file", type=str, required=True)
    parser.add_argument("--py_func_file", type=str, required=True)
    parser.add_argument("--hip_file", type=str, required=True)
    parser.add_argument("--ref_hip_file", type=str, required=True)
    return parser.parse_args()


def _canonical_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _candidate_module_keys(func_key: str, module_keys: List[str]) -> List[str]:
    if func_key in module_keys:
        return [func_key]

    func_ck = _canonical_key(func_key)
    exact = [k for k in module_keys if _canonical_key(k) == func_ck]
    if exact:
        return exact

    suffix = [k for k in module_keys if _canonical_key(k).endswith(func_ck)]
    if len(suffix) == 1:
        return suffix

    bi_suffix = [k for k in module_keys if func_ck.endswith(_canonical_key(k))]
    if len(bi_suffix) == 1:
        return bi_suffix

    return []


def _align_state_dict(module_obj: Any, func_obj: Any) -> Tuple[bool, Dict[str, Any]]:
    module_sd = module_obj.state_dict()
    func_sd = func_obj.state_dict()

    aligned_sd = {k: v for k, v in func_sd.items()}
    used_module = set()
    mapped = []
    missing = []
    shape_mismatch = []

    module_keys = list(module_sd.keys())

    for fk, fv in func_sd.items():
        candidates = [k for k in _candidate_module_keys(fk, module_keys) if k not in used_module]
        if not candidates:
            missing.append(fk)
            continue

        mk = candidates[0]
        mv = module_sd[mk]
        if mv.shape != fv.shape:
            shape_mismatch.append({"func_key": fk, "module_key": mk, "func_shape": list(fv.shape), "module_shape": list(mv.shape)})
            continue

        aligned_sd[fk] = mv.detach().clone()
        used_module.add(mk)
        mapped.append({"func_key": fk, "module_key": mk})

    func_obj.load_state_dict(aligned_sd, strict=False)

    param_keys = {k for k, _ in func_obj.named_parameters()}
    unresolved_params = [k for k in missing if k in param_keys]
    ok = len(unresolved_params) == 0 and len(shape_mismatch) == 0

    return ok, {
        "mapped": mapped,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "unresolved_param_keys": unresolved_params,
    }


def load_modu_obj(py_modu_path: str, class_name: str, init_func_name: str) -> Any:
    init_func = load_function_from_path(py_modu_path, init_func_name)
    py_class = load_function_from_path(py_modu_path, class_name)
    init_params = init_func()
    if len(init_params) == 0:
        model = py_class()
    elif len(init_params) == 2 and isinstance(init_params[0], list) and isinstance(init_params[1], dict):
        model = py_class() if len(init_params[1]) == 0 else py_class(**init_params[1])
    else:
        model = py_class(*init_params)
    return model


def load_func_obj(py_func_path: str, class_name: str, init_func_name: str) -> Any:
    init_func = load_function_from_path(py_func_path, init_func_name)
    py_class = load_function_from_path(py_func_path, class_name)
    init_params = init_func()
    if len(init_params) == 0:
        model = py_class()
    elif len(init_params) == 2 and isinstance(init_params[0], list) and isinstance(init_params[1], dict):
        model = py_class() if len(init_params[1]) == 0 else py_class(**init_params[1])
    else:
        model = py_class(*init_params)
    return model


def _compare_results(modu_result: Any, func_result: Any, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    if isinstance(modu_result, dict) and isinstance(func_result, dict):
        if set(modu_result.keys()) != set(func_result.keys()):
            return False
        for k in modu_result:
            if not _compare_results(modu_result[k], func_result[k], rtol=rtol, atol=atol):
                return False
        return True
    if isinstance(modu_result, (list, tuple)) and isinstance(func_result, (list, tuple)):
        if len(modu_result) != len(func_result):
            return False
        return all(_compare_results(a, b, rtol=rtol, atol=atol) for a, b in zip(modu_result, func_result))
    if torch.is_tensor(modu_result) and torch.is_tensor(func_result):
        return torch.allclose(modu_result, func_result, rtol=rtol, atol=atol)
    return modu_result == func_result


def _write_perf_report(report: Dict[str, Any]) -> None:
    report_dir = os.path.join(os.getcwd(), "build")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "performance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def cal_hip_latency(kernel_hip: Any, inputs: List[Any], hip_fn: Any, n_iter: int = 100, n_warmup: int = 10) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(n_warmup):
        kernel_hip(*inputs, fn=hip_fn)

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        kernel_hip(*inputs, fn=hip_fn)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    avg_time = elapsed / n_iter
    return avg_time


def _normalize_get_inputs_result(inputs_result: Any) -> Any:
    """
    Normalize get_inputs() result to handle both single return and generator patterns.
    Returns a generator that yields test cases.
    """
    # If it's already a generator (has __iter__ and not a list/tuple), return as-is
    if hasattr(inputs_result, '__iter__') and not isinstance(inputs_result, (list, tuple, str)):
        # Check if it's a generator (has __next__ or is a generator function result)
        if hasattr(inputs_result, '__next__') or hasattr(inputs_result, 'send'):
            return inputs_result
    
    # If it's a list/tuple, wrap in a generator
    if isinstance(inputs_result, (list, tuple)):
        def _gen():
            yield inputs_result
        return _gen()
    
    # Otherwise, wrap single result
    def _gen():
        yield inputs_result
    return _gen()


def cal_kernel_perf(
    py_modu_path: str,
    py_func_path: str,
    hip_kernel_path: str,
    ref_hip_kernel_path: str,
    build_dir: str = "temp",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    auto_cleanup: bool = True,
) -> Tuple[Any, Any, Any]:
    failed_ret: Tuple[Any, Any, Any] = (None, None, None)

    # Create separate build directories for reference and optimized kernels
    ref_hip_dir = os.path.join(build_dir, "hip_ref")
    opt_hip_dir = os.path.join(build_dir, "hip_opt")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(ref_hip_dir, exist_ok=True)
    os.makedirs(opt_hip_dir, exist_ok=True)
    
    shutil.copy(ref_hip_kernel_path, ref_hip_dir)
    shutil.copy(hip_kernel_path, opt_hip_dir)

    hip_file_name = os.path.basename(hip_kernel_path)
    ref_hip_file_name = os.path.basename(ref_hip_kernel_path)
    # Extract kernel name from file (e.g., hip_102_ItemQueryAttention.hip -> ItemQueryAttention)
    kernel_name = hip_file_name.split('.hip')[0].split('_', 2)[-1]
    # Use different module names to avoid conflicts when loading both kernels
    ref_module_name = kernel_name + "_ref"
    opt_module_name = kernel_name
    
    report: Dict[str, Any] = {
        "status": "fail",
        "kernel": kernel_name,
        "alignment": {},
        "message": "",
        "speedup": None,
        "ori_time": None,
        "opt_time": None,
        "test_cases": [],
    }

    # Load reference HIP kernel (use different module name to avoid conflicts)
    ref_hip_fn = load_hip_kernel(ref_module_name, ref_hip_dir, ref_hip_file_name)
    if ref_hip_fn is None:
        report["message"] = "Reference HIP kernel failed to compile/load"
        _write_perf_report(report)
        if auto_cleanup:
            clear_workdir(ref_hip_dir)
            clear_workdir(opt_hip_dir)
        return failed_ret

    # Load optimized HIP kernel
    opt_hip_fn = load_hip_kernel(opt_module_name, opt_hip_dir, hip_file_name)
    if opt_hip_fn is None:
        report["message"] = "Optimized HIP kernel failed to compile/load"
        _write_perf_report(report)
        if auto_cleanup:
            clear_workdir(ref_hip_dir)
            clear_workdir(opt_hip_dir)
        return failed_ret

    # Load get_inputs function and normalize to generator
    input_func_from_modu = load_function_from_path(py_modu_path, 'get_inputs')
    inputs_modu_gen = _normalize_get_inputs_result(input_func_from_modu())
    
    input_func_from_func = load_function_from_path(py_func_path, 'get_inputs')
    inputs_func_gen = _normalize_get_inputs_result(input_func_from_func())

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    kernel_modu = load_modu_obj(py_modu_path, kernel_name, 'get_init_inputs').to('cuda')
    kernel_func = load_func_obj(py_func_path, kernel_name, 'get_init_inputs').to('cuda')
    align_ok, align_info = _align_state_dict(kernel_modu, kernel_func)
    report["alignment"] = align_info

    if not align_ok:
        report["message"] = "Failed to align functional model parameters with module model"
        _write_perf_report(report)
        if auto_cleanup:
            clear_workdir(ref_hip_dir)
            clear_workdir(opt_hip_dir)
        return failed_ret

    kernel_modu.eval()
    kernel_func.eval()

    # Process each test case
    sentinel = object()
    ref_times = []
    opt_times = []
    all_correct = True

    for case_idx, (inputs_modu, inputs_func) in enumerate(
        itertools.zip_longest(inputs_modu_gen, inputs_func_gen, fillvalue=sentinel)
    ):
        if inputs_modu is sentinel or inputs_func is sentinel:
            report["message"] = f"Mismatched test case count at case {case_idx}"
            _write_perf_report(report)
            if auto_cleanup:
                clear_workdir(ref_hip_dir)
                clear_workdir(opt_hip_dir)
            return failed_ret

        # Normalize inputs to lists
        if not isinstance(inputs_modu, (list, tuple)):
            inputs_modu = [inputs_modu]
        if not isinstance(inputs_func, (list, tuple)):
            inputs_func = [inputs_func]
        
        inputs_modu = list(inputs_modu)
        inputs_func = list(inputs_func)

        # Move to CUDA
        inputs_modu_cuda = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs_modu]
        inputs_func_cuda = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs_func]

        # Extract params from input shapes for KDLoss: input, target
        params: Dict[str, Any] = {}
        if len(inputs_func) >= 2:
            input_tensor = inputs_func[0]
            target_tensor = inputs_func[1]
            if isinstance(input_tensor, torch.Tensor) and isinstance(target_tensor, torch.Tensor):
                params["input_shape"] = list(input_tensor.shape)
                params["target_shape"] = list(target_tensor.shape)
        # Extract temp_factor from module
        if hasattr(kernel_modu, 'temp_factor'):
            params["temp_factor"] = kernel_modu.temp_factor

        case_entry: Dict[str, Any] = {
            "case_idx": case_idx,
            "correct": False,
            "ref_time": None,
            "opt_time": None,
            "speedup": None,
            "execution_time_ms": None,
            "params": params,
        }

        # Correctness check: compare reference HIP vs optimized HIP
        try:
            torch.manual_seed(1337 + case_idx)
            torch.cuda.manual_seed_all(1337 + case_idx)
            ref_result = kernel_func(*copy.deepcopy(inputs_func_cuda), fn=ref_hip_fn)

            torch.manual_seed(1337 + case_idx)
            torch.cuda.manual_seed_all(1337 + case_idx)
            opt_result = kernel_func(*copy.deepcopy(inputs_func_cuda), fn=opt_hip_fn)

            if not _compare_results(ref_result, opt_result, rtol=rtol, atol=atol):
                print(f"[MISMATCH] {kernel_name} case {case_idx}: reference and optimized results differ.")
                case_entry["error"] = "output_mismatch"
                all_correct = False
                report["test_cases"].append(case_entry)
                continue
            case_entry["correct"] = True
        except Exception as e:
            print(f"[Error] {kernel_name} case {case_idx} raises an exception: {e}")
            case_entry["error"] = f"exception: {e}"
            all_correct = False
            report["test_cases"].append(case_entry)
            continue

        # Performance comparison: reference HIP vs optimized HIP
        try:
            ref_time = cal_hip_latency(kernel_func, inputs_func_cuda, ref_hip_fn)
            opt_time = cal_hip_latency(kernel_func, inputs_func_cuda, opt_hip_fn)
            
            case_entry["ref_time"] = round(ref_time, 5)
            case_entry["opt_time"] = round(opt_time, 5)
            case_entry["speedup"] = round(ref_time / opt_time, 2) if opt_time > 0 else None
            case_entry["execution_time_ms"] = round(opt_time, 5)
            
            ref_times.append(ref_time)
            opt_times.append(opt_time)
            
            print(f"[INFO] Case {case_idx}: ref={ref_time:.5f}ms, opt={opt_time:.5f}ms, speedup={case_entry['speedup']:.2f}x")
        except Exception as e:
            print(f"[Error] {kernel_name} case {case_idx} performance exception: {e}")
            case_entry["error"] = f"perf_exception: {e}"
            all_correct = False

        report["test_cases"].append(case_entry)

    if not all_correct:
        report["message"] = "Some test cases failed correctness or performance checks"
        report["status"] = "partial"
    elif len(ref_times) == 0:
        report["message"] = "No valid test cases processed"
        _write_perf_report(report)
        if auto_cleanup:
            clear_workdir(ref_hip_dir)
            clear_workdir(opt_hip_dir)
        return failed_ret
    else:
        # Calculate average times across all test cases
        avg_ref_time = sum(ref_times) / len(ref_times)
        avg_opt_time = sum(opt_times) / len(opt_times)
        avg_speedup = avg_ref_time / avg_opt_time if avg_opt_time > 0 else None

        print(f"[INFO] HIP kernel {kernel_name} processed {len(ref_times)} test cases.")
        print(f"[INFO] Average: ref={avg_ref_time:.5f}ms, opt={avg_opt_time:.5f}ms, speedup={avg_speedup:.2f}x")
        
        report["status"] = "ok"
        report["message"] = f"Performance benchmark completed for {len(ref_times)} test cases"
        report["speedup"] = round(avg_speedup, 2) if avg_speedup else None
        report["ori_time"] = round(avg_ref_time, 5)
        report["opt_time"] = round(avg_opt_time, 5)
        _write_perf_report(report)

        if auto_cleanup:
            clear_workdir(ref_hip_dir)
            clear_workdir(opt_hip_dir)
        return round(avg_speedup, 2) if avg_speedup else None, round(avg_ref_time, 5), round(avg_opt_time, 5)


if __name__ == "__main__":
    args = parse_args()
    ret_perf = cal_kernel_perf(args.py_modu_file, args.py_func_file, args.hip_file, args.ref_hip_file)
    if ret_perf[0] is not None:
        save_eval_result({"speedup": ret_perf[0], "ori_time": ret_perf[1], "opt_time": ret_perf[2]})
    sys.exit(0 if ret_perf[0] is not None else 1)

