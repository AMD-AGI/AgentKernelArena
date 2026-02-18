# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
import subprocess
import sys
import torch
import shutil
from typing import Optional
from kernel_loader_template import kernel_loader_template
from utils import save_eval_result


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing the path to the HIP kernel file.
    """
    parser = argparse.ArgumentParser(
        description="Compile check for hip kernel."
    )
    parser.add_argument(
        "--hip_file",
        type=str,
        required=True,
        help="Path to the HIP kernel file."
    )
    return parser.parse_args()


def clear_workdir(work_dir: str) -> None:
    """
    Remove the temporary working directory and all its contents.

    Args:
        work_dir (str): Path to the directory to be deleted.

    Note:
        Errors during deletion are caught and warned, but not raised.
    """
    try:
        shutil.rmtree(work_dir)
    except Exception as e:
        print(f"[WARN] Failed to cleanup work dir {work_dir}: {e}")


def compile_hip(
    hip_file_path: str,
    build_dir: str = "temp",
    auto_cleanup: bool = True
) -> bool:
    """
    Compile a single HIP kernel file by generating a temporary loader script and running it.

    This function:
    - Creates a temporary build directory
    - Copies the .hip file into it
    - Generates a Python loader script using kernel_loader_template
    - Executes the script to trigger JIT compilation via torch.utils.cpp_extension
    - Reports success/failure and optionally cleans up

    Args:
        hip_file_path (str): Full path to the input .hip source file.
        build_dir (str, optional): Temporary directory for build artifacts. Defaults to "temp".
        auto_cleanup (bool, optional): Whether to delete the build directory after compilation.
                                       Defaults to True.

    Returns:
        bool: True if compilation succeeded, False otherwise.
    """
    hip_dir = os.path.join(build_dir, "hip")
    # Prepare dirs
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(hip_dir, exist_ok=True)
   
    # Copy HIP file
    shutil.copy(hip_file_path, hip_dir)
   
    hip_file_name = os.path.basename(hip_file_path)
    kernel_name = hip_file_name.replace(".hip", "")
   
    # Generate compile script
    hip_kernel_call_code = kernel_loader_template.format(
        kernel_name=kernel_name,
        code_dir=hip_dir,
        code_file=hip_file_name,
    )
   
    hip_comp_file = os.path.join(build_dir, "compile_kernel.py")
    with open(hip_comp_file, "w") as f:
        f.write(hip_kernel_call_code)
   
    # Compile
    try:
        print(f"[INFO] Compiling HIP kernel {kernel_name}...")
        proc = subprocess.run([sys.executable, hip_comp_file], capture_output=True, text=True)
   
        if proc.returncode != 0:
            print(f"[ERROR] Compilation failed:\n{proc.stderr}")
            if auto_cleanup:
                clear_workdir(build_dir)
            return False
   
    except Exception as e:
        print(f"[ERROR] Compilation exception: {e}")
        if auto_cleanup:
            clear_workdir(build_dir)
        return False
   
    # Cleanup
    if auto_cleanup:
        clear_workdir(build_dir)
    print(f"[INFO] HIP kernel {kernel_name} compile passed.")
    return True


if __name__ == "__main__":
    args = parse_args()
    ret_compiled = compile_hip(args.hip_file)
    ret_dict = {'compiled': ret_compiled}
    save_eval_result(ret_dict)
