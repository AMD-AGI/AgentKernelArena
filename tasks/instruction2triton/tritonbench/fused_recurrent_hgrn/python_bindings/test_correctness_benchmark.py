# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import sys
import os
import re

if len(sys.argv) < 2:
    print("Usage: python test_benchmark.py <bench_name> [workdir] [logdir]")
    print("Example: python test_benchmark.py benchmark_device_merge_sort /mnt/raid0/jianghui/projects/kernel_agent/rocPRIM build")
    print("If workdir and logdir are not provided, current working directory will be used.")
    sys.exit(1)

bench_name = sys.argv[1]
workdir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
logdir = sys.argv[3] if len(sys.argv) > 3 else os.getcwd()

print(f"🔧 Benchmark: {bench_name}")
print(f"📁 Work directory: {workdir}")
print(f"📝 Log directory: {logdir}")

kernel_name = bench_name.split("benchmark_")[-1]

log_idx = 0
while True:
    log_file = f"{logdir}/{bench_name}_{log_idx}.log"
    if not os.path.exists(os.path.join(workdir, log_file)):
        break
    log_idx += 1

log_path = f"{logdir}/{bench_name}_{log_idx}.log"
print(f"➡️ Log file will be saved to: {log_path}")

commands = [
    "ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DBUILD_TEST=ON -DAMDGPU_TARGETS=gfx942 .",
    # save git patch first
    f"git diff > {logdir}/git_patch_{bench_name}_{log_idx}.patch",
    f"make -j test_{kernel_name}",
    f"./test/rocprim/test_{kernel_name}",
    f"make -j8 {bench_name}",
    f"./benchmark/{bench_name} --trials 20 2>&1 | tee {log_path}"
]

if not os.path.exists(workdir):
    os.makedirs(workdir)
for idx, cmd in enumerate(commands):
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=workdir)
    if result.returncode != 0:
        if idx == 3:
            print("correctness test failed")
        else:
            print(f"fail: {cmd}")
        break
