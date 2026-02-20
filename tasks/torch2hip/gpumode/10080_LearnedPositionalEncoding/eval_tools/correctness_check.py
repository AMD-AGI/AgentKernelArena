# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
import copy
import torch
import shutil
import sys
from typing import Any, Dict, List, Tuple, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compile import compile_hip, clear_workdir
from utils import load_function_from_path, load_hip_kernel, save_eval_result


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the correctness checking script.

    Returns:
        argparse.Namespace: Parsed arguments containing paths to Python module,
                            function implementation, and HIP kernel files.
    """
    parser = argparse.ArgumentParser(
        description="Correctness check for PyTorch and HIP kernels."
    )
    parser.add_argument(
        "--py_modu_file",
        type=str,
        required=True,
        help="Path to the Python module file (containing modular-style kernel)."
    )
    parser.add_argument(
        "--py_func_file",
        type=str,
        required=True,
        help="Path to the Python function implementation file (functional-style kernel)."
    )
    parser.add_argument(
        "--hip_file",
        type=str,
        required=True,
        help="Path to the HIP kernel file (.hip)."
    )
    return parser.parse_args()


def load_modu_obj(py_modu_path: str, class_name: str, init_func_name: str) -> Any:
    """
    Load and instantiate a modular-style kernel class from a Python file.

    The target file must contain:
    - A function named `init_func_name` that returns initialization arguments
    - A class named `class_name` that can be instantiated with those args

    Args:
        py_modu_path (str): Path to the Python file containing the class.
        class_name (str): Name of the kernel class to instantiate.
        init_func_name (str): Name of the function returning init parameters.

    Returns:
        Any: Instantiated kernel object (typically a torch.nn.Module).
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
    Load and instantiate a functional-style kernel class from a Python file.

: Same logic as load_modu_obj.

    Args:
        py_func_path (str): Path to the Python file containing the functional kernel.
        class_name (str): Name of the functional kernel class.
        init_func_name (str): Name of the function returning init parameters.

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
    Compare two computation results (tensors or dicts of tensors) for numerical closeness.

    Supports:
        - Single torch.Tensor
        - Dict[str, torch.Tensor]
        - Exact equality fallback for other types

    Args:
        modu_result (Any): Result from modular kernel.
        func_result (Any): Result from functional + HIP kernel.
        rtol (float): Relative tolerance for torch.allclose.
        atol (float): Absolute tolerance for torch.allclose.

    Returns:
        bool: True if results are close enough, False otherwise.
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


def correctness_check(
    py_modu_path: str,
    py_func_path: str,
    hip_kernel_path: str,
    build_dir: str = "temp",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    auto_cleanup: bool = True
) -> bool:
    """
    Perform end-to-end correctness check between a PyTorch (modular) implementation
    and a functional + HIP kernel implementation.

    Steps:
        1. Compile the HIP kernel
        2. Load inputs from the modular file
        3. Instantiate both kernel objects
        4. Run both implementations on GPU
        5. Compare outputs with torch.allclose

    Args:
        py_modu_path (str): Path to modular PyTorch implementation.
        py_func_path (str): Path to functional PyTorch + HIP wrapper.
        hip_kernel_path (str): Path to the .hip source file.
        build_dir (str): Temporary directory for compilation.
        rtol (float): Relative tolerance for comparison.
        atol (float): Absolute tolerance for comparison.
        auto_cleanup (bool): Whether to delete build directory after check.

    Returns:
        bool: True if compilation and correctness check both pass, False otherwise.
    """
   
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
        return False

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

    # get outputs from py_modu and py_func
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
            return False
    except Exception as e:
        print(f"[Error] {kernel_name} raises an exception due to {e}.")
        if auto_cleanup:
            clear_workdir(build_dir)
        return False
    print(f"[INFO] HIP kernel {kernel_name} correctness check passed.")
    if auto_cleanup:
        clear_workdir(build_dir)
    return True


if __name__ == "__main__":
    args = parse_args()
    ret_correctness = correctness_check(args.py_modu_file, args.py_func_file, args.hip_file)
    ret_dict = {'correctness': ret_correctness}
    save_eval_result(ret_dict)
    sys.exit(0 if ret_correctness else 1)
