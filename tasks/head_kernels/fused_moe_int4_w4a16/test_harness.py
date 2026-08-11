"""K1 harness: int4 W4A16 fused-MoE GEMM (fused_moe_kernel_gptq_awq).

Golden reference = the INSTALLED-ORIGINAL vLLM kernel; candidate = the local
editable kernel in kernel_jit.py. Identical inputs -> outputs must match
(bit-tight, tiny tolerance for any benign reassociation). Mirrors the 4-mode
contract used by the other PerfSkills harnesses.

Live Kimi-K2.6 MoE GEMM shapes (TP4, EP1):
  gemm1 (gate_up): N=1024, K=7168
  gemm2 (down)   : N=7168, K=512
"""
import os
import sys
import importlib.util
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import host  # noqa: E402
from kernel_jit import fused_moe_kernel_gptq_awq as CAND  # noqa: E402

from _bench_common import make_argparser, run_modes  # noqa: E402  (vendored, same dir)


def _load_golden():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "canonical",
        "source_golden",
        "kernel_jit.py",
    )
    spec = importlib.util.spec_from_file_location("fused_moe_int4_golden", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_moe_kernel_gptq_awq


GOLD = _load_golden()

# WORKLOAD REGIME: seqlen=1024, concurrency B in {2,32,64}. This is a
# token-parallel MoE GEMM, so the variable dim M = num_tokens = B*1024.
# Model dims kept from the captured live Kimi-K2.6 base case: the dominant
# gemm1 (gate_up) shard, N=1024, K=7168, group_size=32.
SEQLEN = 1024
# (name, M, N, K, has_zp) -- performance cases (one per concurrency).
PERF_CASES = [
    ("c2", 2 * SEQLEN, 1024, 7168, False),
    ("c32", 32 * SEQLEN, 1024, 7168, False),
    ("c64", 64 * SEQLEN, 1024, 7168, False),
]
# Correctness keeps the broad oracle (both gemm1/gemm2 + has_zp path) AND the
# regime shapes, so edited-vs-golden verification stays strong.
CORRECTNESS_CASES = [
    ("gemm1_gateup_M64", 64, 1024, 7168, False),
    ("gemm2_down_M64", 64, 7168, 512, False),
    ("gemm1_gateup_M2048", 2048, 1024, 7168, False),
    ("gemm2_down_M2048", 2048, 7168, 512, False),
    ("gemm1_gateup_M64_zp", 64, 1024, 7168, True),
    ("c2", 2 * SEQLEN, 1024, 7168, False),
    ("c32", 32 * SEQLEN, 1024, 7168, False),
    ("c64", 64 * SEQLEN, 1024, 7168, False),
]


def make_check(M, N, K, has_zp):
    def check():
        inp = host.build_inputs(M, N, K, has_zp=has_zp, seed=1234)
        out_gold = host.invoke(GOLD, inp)
        out_cand = host.invoke(CAND, inp)
        # tight: outputs should be (near) identical under identical inputs
        ok = torch.allclose(out_cand.float(), out_gold.float(), rtol=2e-2, atol=2e-2)
        if not ok:
            d = (out_cand.float() - out_gold.float()).abs()
            print(f"    max_abs={d.max().item():.4e} mean_abs={d.mean().item():.4e}")
        return ok
    return check


def make_run(M, N, K, has_zp):
    inp = host.build_inputs(M, N, K, has_zp=has_zp, seed=1234)
    return lambda: host.invoke(CAND, inp)


def main():
    args = make_argparser("K1 fused_moe int4 gptq_awq").parse_args()
    # Correctness uses the broad oracle set; timing modes use the regime set.
    selected = CORRECTNESS_CASES if args.correctness else PERF_CASES
    cases = []
    for (name, M, N, K, has_zp) in selected:
        cases.append({
            "name": name,
            "run": make_run(M, N, K, has_zp),
            "check": make_check(M, N, K, has_zp),
        })
    run_modes(args, cases)


if __name__ == "__main__":
    main()
