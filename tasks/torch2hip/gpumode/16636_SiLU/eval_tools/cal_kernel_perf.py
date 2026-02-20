# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
import copy
import torch
import shutil
import sys
from typing import Any, List, Tuple, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compile import compile_hip, clear_workdir
from utils import load_function_from_path, load_hip_kernel, save_eval_result


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the kernel performance benchmarking script.

    Returns:
        argparse.Namespace: Object containing paths to modular PyTorch impl,
                            functional wrapper, and HIP kernel source.
    """
    parser = argparse.ArgumentParser(
        description="Cal performance for PyTorch and HIP kernels."
    )
    parser.add_argument(
        "--py_modu_file",
        type=str,
        required=True,
        help="Path to the Python module file (modular PyTorch implementation)."
    )
    parser.add_argument(
        "--py_func_file",
        type=str,
        required=True,
        help="Path to the Python function implementation file (functional + HIP wrapper)."
    )
    parser.add_argument(
        "--hip_file",
        type=str,
        required=True,
        help="Path to the HIP kernel source file (.hip)."
    )
    return parser.parse_args()


def load_modu_obj(py_modu_path: str, class_name: str, init_func_name: str) -> Any:
    """
    Load and instantiate a modular-style kernel class from a Python file.

    Args:
        py_modu_path (str): Path to the module file.
        class_name (str): Name of the kernel class.
        init_func_name (str): Name of the function that returns constructor arguments.

    Returns:
        Any: Instantiated kernel object (usually a torch.nn.Module subclass).
    """
    init_func = load_function_from_path(py_modu_path, init_func_name)
    py_class = load_function_from_path(py_modu_path, class_name)
    init_params = init_func()
    if len(init_params) == 0:
        model = py_class()
    elif len(init_params) == 2 and (isinstance(init_params[0], list) and isinstance(init_params[1], dict)):
        model = py_class() if len(init_params[1]) == 0 else py_class(**(init_params[1]))
    else:
        model = py_class(*(init_params))
    return model


def load_func_obj(py_func_path: str, class_name: str, init_func_name: str) -> Any:
    """
    Load and instantiate a functional-style kernel class that accepts a HIP kernel function.

    Args:
        py_func_path (str): Path to the functional implementation file.
        class_name (str): Name of the functional kernel class.
        init_func_name (str): Name of the init parameter function.

    Returns:
        Any: Instantiated functional kernel object.
    """
    init_func = load_function_from_path(py_func_path, init_func_name)
    py_class = load_function_from_path(py_func_path, class_name)
    init_params = init_func()
    if len(init_params) == 0:
        model = py_class()
    elif len(init_params) == 2 and (isinstance(init_params[0], list) and isinstance(init_params[1], dict)):
        model = py_class() if len(init_params[1]) == 0 else py_class(**(init_params[1]))
    else:
        model = py_class(*(init_params))
    return model


def _compare_results(
    modu_result: Any,
    func_result: Any,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> bool:
    """
    Compare two kernel outputs (tensor or dict of tensors) for numerical equivalence.

    Args:
        modu_result (Any): Output from modular PyTorch kernel.
        func_result (Any): Output from functional + HIP kernel.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Returns:
        bool: True if results are close within tolerance.
    """
    if isinstance(modu_result, dict) and isinstance(func_result, dict):
        for k in modu_result:
            if k not in func_result:
                return False
            if not torch.allclose(modu_result[k], func_result[k], rtol=rtol, atol=atol):
                return False
        return True
    elif torch.is_tensor(modu_result) and torch.is_tensor(func_result):
        return torch.allclose(modu_result, func_result, rtol=rtol, atol=atol)
    else:
        return modu_result == func_result


def cal_hip_latency(
    kernel_hip: Any,
    inputs: List[Any],
    hip_fn: Any,
    n_iter: int = 1000
) -> float:
    """
    Measure average latency of the HIP kernel implementation.

    Args:
        kernel_hip (Any): Functional kernel object.
        inputs (List[Any]): List of input arguments (already on CUDA).
        hip_fn (Any): Loaded HIP kernel function from cpp_extension.
        n_iter (int): Number of warmup + measurement iterations.

    Returns:
        float: Average time per call in milliseconds.
    """
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


def cal_modu_latency(
    kernel_modu: Any,
    inputs: List[Any],
    n_iter: int = 1000
) -> float:
    """
    Measure average latency of the pure PyTorch (modular) implementation.

    Args:
        kernel_modu (Any): Modular kernel object.
        inputs (List[Any]): List of input arguments (already on CUDA).
        n_iter (int): Number of iterations.

    Returns:
        float: Average time per call in milliseconds.
    """
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
    auto_cleanup: bool = True
) -> str:
    """
    Full performance + correctness benchmark between PyTorch and HIP implementations.

    Returns speedup as a formatted string (e.g., "5.23"). Returns "0.0" on any failure.

    Args:
        py_modu_path (str): Path to modular PyTorch implementation.
        py_func_path (str): Path to functional wrapper.
        hip_kernel_path (str): Path to .hip source file.
        build_dir (str): Temporary build directory.
        rtol (float): Relative tolerance for correctness check.
        atol (float): Absolute tolerance for correctness check.
        auto_cleanup (bool): Remove build directory after run.

    Returns:
        str: Speedup factor formatted to 2 decimal places, or "0.0" on error.
    """
    failed_ret = [None, None, None]

    hip_dir = os.path.join(build_dir, "hip")
    # Prepare dirs
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(hip_dir, exist_ok=True)
    shutil.copy(hip_kernel_path, hip_dir)

    # Load HIP kernel via JIT (single in-process compilation, no subprocess)
    hip_file_name = os.path.basename(hip_kernel_path)
    kernel_name = hip_file_name.split('.hip')[0].split('_', 2)[-1]
    hip_fn = load_hip_kernel(kernel_name, hip_dir, hip_file_name)
    if hip_fn is None:
        print(f"[INFO] the hip kernel {hip_kernel_path} fail to compile.")
        if auto_cleanup:
            clear_workdir(build_dir)
        return failed_ret

    # get inputs for py_modu and py_func
    input_func_from_modu = load_function_from_path(py_modu_path, 'get_inputs')
    inputs_modu = input_func_from_modu()
    input_func_from_func = load_function_from_path(py_func_path, 'get_inputs')
    inputs_func = input_func_from_func()
    # inputs_func = copy.deepcopy(inputs_modu)
    for idx in range(len(inputs_modu)):
        inputs_func[idx] = copy.deepcopy(inputs_modu[idx])
   
    # get objs for py_modu and py_func
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    kernel_modu = load_modu_obj(py_modu_path, kernel_name, 'get_init_inputs').to('cuda')
    kernel_func = load_func_obj(py_func_path, kernel_name, 'get_init_inputs').to('cuda')
    kernel_func.load_state_dict(kernel_modu.state_dict(), strict=False)
    kernel_modu.eval()
    kernel_func.eval()

    # move inputs to cuda
    inputs_modu = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs_modu]
    inputs_func = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs_func]
  
    assert _compare_results(inputs_modu[0], inputs_func[0], rtol=rtol, atol=atol), '[Error] Inputs for pytorch and hip kernel differ.'

    try:
        modu_result = kernel_modu(*inputs_modu)
        func_result = kernel_func(*inputs_func, fn=hip_fn)
        # compare the difference
        if not _compare_results(modu_result, func_result, rtol=rtol, atol=atol):
            print(f"[MISMATCH] {kernel_name} results differ.")
            if auto_cleanup:
                clear_workdir(build_dir)
            return failed_ret
    except Exception as e:
        print(f"[Error] {kernel_name} raises an exception due to {e}.")
        if auto_cleanup:
            clear_workdir(build_dir)
        return failed_ret
    
    print(f"[INFO] HIP kernel {kernel_name} correctness check passed.")
    torch_time = cal_modu_latency(kernel_modu, inputs_modu)
    hip_time = cal_hip_latency(kernel_func, inputs_func, hip_fn)
    speedup = torch_time / hip_time
    print(f"[INFO] HIP vs PyTorch speedup: {speedup:.2f}x")
    speedup = float(f"{speedup:.2f}")
    
    if auto_cleanup:
        clear_workdir(build_dir)
    return speedup, round(torch_time, 5), round(hip_time, 5)


if __name__ == "__main__":
    args = parse_args()
    ret_perf = cal_kernel_perf(args.py_modu_file, args.py_func_file, args.hip_file)
    ret_dict = {'speedup': ret_perf[0], 'ori_time': ret_perf[1], 'opt_time': ret_perf[2]}
    save_eval_result(ret_dict)
    sys.exit(0 if ret_perf[0] is not None else 1)


