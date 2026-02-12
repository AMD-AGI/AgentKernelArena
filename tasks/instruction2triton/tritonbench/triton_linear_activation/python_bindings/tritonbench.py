# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import traceback
import os
import argparse
import traceback
from tb_eval.evaluators.interface import get_evaluators

def tritonbench_eval(kernel_name, code_path):
    evaluator = get_evaluators['tbg']()

    with open(code_path, "r") as f:
        code = f.read()
    
    root, extension = os.path.splitext(code_path)
    tmp_dir = f"{root}_tmp"
    exe_dir = f"{root}_pass_exe"
   
    try:
        pass_call, pass_exe, speedup, call_stdout, call_stderr = evaluator(code, tmp_dir, exe_dir, kernel_name, atol=1e-3, rtol=1e-3, custom_tests_path=None)
    except Exception as e:
            print(f"failed to test the code for {kernel_name} due to {e}")
            import traceback
            print(traceback.format_exc())
            pass_call=False; pass_exe=False; speedup=0; call_stdout=""; call_stderr=str(e)
            print(traceback.format_exc())
            pass_call=False; pass_exe=False; speedup=0; call_stdout=""; call_stderr=str(e)
    
    if not pass_call:
        print(call_stderr)
    elif not pass_exe:
        err_msg = call_stderr if call_stderr else call_stdout
        print(f"""
Call Status: True
Exec Status: False
{err_msg}      
""")
    else:
        print(f"""
Call Status: True
Exec Status: True
Speedup: {speedup}
""")


def test():
    kernel_name = "matrix_vector_multip.py"
    code_path = "/mnt/raid0/jianghui/projects/kernel_agent/swe_agent/mini-swe-agent/src/minisweagent/run/output/20251112/matrix_vector_multip.py"
    tritonbench_eval(kernel_name, code_path)

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_name", type=str)
    parser.add_argument("code_path", type=str)
    args = parser.parse_args()
    tritonbench_eval(args.kernel_name, args.code_path)
