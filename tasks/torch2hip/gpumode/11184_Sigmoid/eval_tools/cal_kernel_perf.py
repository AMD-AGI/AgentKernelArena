# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import argparse
import copy
import re
import torch
import shutil
import sys
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compile import clear_workdir
from utils import load_function_from_path, load_hip_kernel, save_eval_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kernel performance benchmark for PyTorch and HIP kernels.")
    parser.add_argument("--py_modu_file", type=str, required=True)
    parser.add_argument("--py_func_file", type=str, required=True)
    parser.add_argument("--hip_file", type=str, required=True)
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


def cal_hip_latency(kernel_hip: Any, inputs: List[Any], hip_fn: Any, n_iter: int = 1000) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        kernel_hip(*inputs, fn=hip_fn)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    avg_time = elapsed / n_iter
    print("HIP perf:", avg_time, "ms")
    return avg_time


def cal_modu_latency(kernel_modu: Any, inputs: List[Any], n_iter: int = 1000) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        kernel_modu(*inputs)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    avg_time = elapsed / n_iter
    print("PyTorch perf:", avg_time, "ms")
    return avg_time


def cal_kernel_perf(
    py_modu_path: str,
    py_func_path: str,
    hip_kernel_path: str,
    build_dir: str = "temp",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    auto_cleanup: bool = True,
) -> Tuple[Any, Any, Any]:
    failed_ret: Tuple[Any, Any, Any] = (None, None, None)

    hip_dir = os.path.join(build_dir, "hip")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(hip_dir, exist_ok=True)
    shutil.copy(hip_kernel_path, hip_dir)

    hip_file_name = os.path.basename(hip_kernel_path)
    kernel_name = hip_file_name.split('.hip')[0].split('_', 2)[-1]
    report: Dict[str, Any] = {
        "status": "fail",
        "kernel": kernel_name,
        "alignment": {},
        "message": "",
        "speedup": None,
        "ori_time": None,
        "opt_time": None,
    }

    hip_fn = load_hip_kernel(kernel_name, hip_dir, hip_file_name)
    if hip_fn is None:
        report["message"] = "HIP kernel failed to compile/load"
        _write_perf_report(report)
        if auto_cleanup:
            clear_workdir(hip_dir)
        return failed_ret

    input_func_from_modu = load_function_from_path(py_modu_path, 'get_inputs')
    inputs_modu = input_func_from_modu()
    inputs_func = copy.deepcopy(inputs_modu)

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
            clear_workdir(hip_dir)
        return failed_ret

    kernel_modu.eval()
    kernel_func.eval()

    inputs_modu = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs_modu]
    inputs_func = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs_func]

    if not _compare_results(inputs_modu, inputs_func, rtol=rtol, atol=atol):
        report["message"] = "Input cloning mismatch"
        _write_perf_report(report)
        if auto_cleanup:
            clear_workdir(hip_dir)
        return failed_ret

    try:
        torch.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        modu_result = kernel_modu(*inputs_modu)

        torch.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        func_result = kernel_func(*inputs_func, fn=hip_fn)

        if not _compare_results(modu_result, func_result, rtol=rtol, atol=atol):
            print(f"[MISMATCH] {kernel_name} results differ.")
            report["message"] = "Correctness precheck failed"
            _write_perf_report(build_dir, report)
            if auto_cleanup:
                clear_workdir(hip_dir)
            return failed_ret
    except Exception as e:
        print(f"[Error] {kernel_name} raises an exception due to {e}.")
        report["message"] = f"Exception during correctness precheck: {e}"
        _write_perf_report(report)
        if auto_cleanup:
            clear_workdir(hip_dir)
        return failed_ret

    print(f"[INFO] HIP kernel {kernel_name} correctness check passed.")
    torch_time = cal_modu_latency(kernel_modu, inputs_modu)
    hip_time = cal_hip_latency(kernel_func, inputs_func, hip_fn)
    speedup = float(f"{(torch_time / hip_time):.2f}")

    print(f"[INFO] HIP vs PyTorch speedup: {speedup:.2f}x")
    report["status"] = "ok"
    report["message"] = "Performance benchmark completed"
    report["speedup"] = speedup
    report["ori_time"] = round(torch_time, 5)
    report["opt_time"] = round(hip_time, 5)
    _write_perf_report(report)

    if auto_cleanup:
        clear_workdir(hip_dir)
    return speedup, round(torch_time, 5), round(hip_time, 5)


if __name__ == "__main__":
    args = parse_args()
    ret_perf = cal_kernel_perf(args.py_modu_file, args.py_func_file, args.hip_file)
    save_eval_result({"speedup": ret_perf[0], "ori_time": ret_perf[1], "opt_time": ret_perf[2]})
    sys.exit(0 if ret_perf[0] is not None else 1)
