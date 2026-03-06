# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import yaml
import importlib.util
from typing import Callable, Optional, Any
import torch
from torch.utils.cpp_extension import load


def load_function_from_path(file_path: str, func_name: str) -> Callable[..., Any]:
    """
    Dynamically load a function from a specified Python file path.

    Args:
        file_path (str): Full path to the .py file containing the target function.
        func_name (str): Name of the function to retrieve from the module.

    Returns:
        Callable[..., Any]: The requested function object that can be called directly.

    Raises:
        AttributeError: If the specified function name does not exist in the module.
        Other import-related exceptions may also be raised by importlib (e.g., SyntaxError, ImportError).
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise AttributeError(f"Function '{func_name}' not found in {file_path}")
    return getattr(module, func_name)


def load_hip_kernel(
    kernel_name: str,
    code_dir: str,
    hip_src: str,
) -> Optional[Callable[..., Any]]:
    """
    Compile and load a HIP (ROCm) kernel as a PyTorch C++ extension at runtime.

    Args:
        kernel_name (str): Name of the compiled extension module (used in `load(..., name=...)`).
        code_dir (str): Directory containing the kernel source and include files.
        hip_src (str): Filename of the HIP source file (relative to code_dir, e.g., "kernel.hip").

    Returns:
        Optional[Callable[..., Any]]:
            On success: the `forward` function from the compiled extension (kernel entry point).
            On failure: None (compilation or loading error).

    Note:
        Errors are caught and printed; the function returns None instead of raising,
        allowing the caller to gracefully fall back to CPU or alternative implementations.
    """
    hip_fn: Optional[Callable[..., Any]] = None
    try:
        hip_kernel_ext = load(name=f"{kernel_name}",
                              sources=[f"{code_dir}/{hip_src}"],
                              verbose=True)
        hip_fn = hip_kernel_ext.forward
    except Exception as e:
        print(f"[Error] Failed to load hip kernel of {hip_src} due to: {e}")
        return hip_fn
    print(f"[INFO] HIP kernel in {hip_src} loading passed.")
    return hip_fn


def save_eval_result(updates: dict, path="eval_result.yaml"):
    """
    Save the provided dictionary to eval_result.yaml.
    - If the file does not exist: create it and write the updates
    - If the file exists: load its content and update it with the provided values
    """
    data = {}

    # If the file exists, load existing content first
    if os.path.exists(path):
        with open(path, "r") as f:
            loaded = yaml.safe_load(f)
            if loaded is not None:
                data = loaded

    # Update the data
    data.update(updates)

    # Write back to YAML
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Saved/updated YAML file: {path}")
