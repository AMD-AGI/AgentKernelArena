# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import argparse
import shutil
import subprocess
import sys
from typing import Dict, Any

from kernel_loader_template import kernel_loader_template
from utils import save_eval_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a HIP kernel source file.")
    parser.add_argument(
        "--hip_file",
        type=str,
        required=True,
        help="Path to the .hip source file to compile."
    )
    return parser.parse_args()


def clear_workdir(work_dir: str) -> None:
    try:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
    except Exception as e:
        print(f"[WARN] Failed to cleanup work dir {work_dir}: {e}")


def _write_compile_report(report: Dict[str, Any]) -> None:
    report_dir = os.path.join(os.getcwd(), "build")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "compile_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def compile_hip(
    hip_file_path: str,
    build_dir: str = "temp",
    auto_cleanup: bool = True
) -> bool:
    hip_dir = os.path.join(build_dir, "hip")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(hip_dir, exist_ok=True)

    shutil.copy(hip_file_path, hip_dir)

    hip_file_name = os.path.basename(hip_file_path)
    kernel_name = hip_file_name.replace(".hip", "")

    hip_kernel_call_code = kernel_loader_template.format(
        kernel_name=kernel_name,
        code_dir=hip_dir,
        code_file=hip_file_name,
    )

    hip_comp_file = os.path.join(build_dir, "compile_kernel.py")
    with open(hip_comp_file, "w", encoding="utf-8") as f:
        f.write(hip_kernel_call_code)

    report: Dict[str, Any] = {
        "status": "fail",
        "kernel": kernel_name,
        "command": [sys.executable, hip_comp_file],
        "stdout": "",
        "stderr": "",
        "returncode": None,
    }

    try:
        print(f"[INFO] Compiling HIP kernel {kernel_name}...")
        proc = subprocess.run([sys.executable, hip_comp_file], capture_output=True, text=True)
        report["stdout"] = proc.stdout
        report["stderr"] = proc.stderr
        report["returncode"] = proc.returncode

        if proc.returncode != 0:
            print(f"[ERROR] Compilation failed:\n{proc.stderr}")
            _write_compile_report(report)
            if auto_cleanup:
                clear_workdir(hip_dir)
            return False
    except Exception as e:
        report["stderr"] = f"Compilation exception: {e}"
        print(f"[ERROR] Compilation exception: {e}")
        _write_compile_report(report)
        if auto_cleanup:
            clear_workdir(hip_dir)
        return False

    report["status"] = "ok"
    _write_compile_report(report)

    if auto_cleanup:
        clear_workdir(hip_dir)
    print(f"[INFO] HIP kernel {kernel_name} compile passed.")
    return True


if __name__ == "__main__":
    args = parse_args()
    ret_compiled = compile_hip(args.hip_file)
    ret_dict = {"compiled": ret_compiled}
    save_eval_result(ret_dict)
    sys.exit(0 if ret_compiled else 1)
