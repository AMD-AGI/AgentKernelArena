"""Run ONE case of the int4 MoE GEMM N times (for rocprofv3 counter collection).
usage: python3 profile_one.py <case_name> <iters>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, host
from kernel_jit import fused_moe_kernel_gptq_awq as CAND

CASES = {
    "gemm1_gateup_M64": (64, 1024, 7168, False),
    "gemm2_down_M64": (64, 7168, 512, False),
    "gemm1_gateup_M2048": (2048, 1024, 7168, False),
    "gemm2_down_M2048": (2048, 7168, 512, False),
    "gemm1_gateup_M64_zp": (64, 1024, 7168, True),
}
name = sys.argv[1] if len(sys.argv) > 1 else "gemm1_gateup_M2048"
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
M, N, K, has_zp = CASES[name]
inp = host.build_inputs(M, N, K, has_zp=has_zp, seed=1234)
for _ in range(3):
    host.invoke(CAND, inp)
torch.cuda.synchronize()
for _ in range(iters):
    host.invoke(CAND, inp)
torch.cuda.synchronize()
print(f"ran {name} x{iters}")
