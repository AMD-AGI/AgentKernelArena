#!/usr/bin/env python3
"""Standalone MoE fused gate benchmark for AgentKernelArena.

Tests moe_fused_gate.cu by calling aiter.moe_fused_gate() and comparing
against the PyTorch reference aiter.biased_grouped_topk_torch().

Self-contained: does not depend on aiter's op_tests/test_moeTopkSoftmax.py.
Only depends on the installed aiter package (import aiter).

Usage:
  python3 test_moe_fused_gate_bench.py --mode correctness
  python3 test_moe_fused_gate_bench.py --mode performance
  python3 test_moe_fused_gate_bench.py --mode correctness --token 1 4 --expert 128
"""
from __future__ import annotations

import argparse
import sys
import torch

# DeepSeek V3 style defaults
GROUP = 8
TOPK = 8
TOPK_GROUP = 3
N_SHARE_EXPERTS_FUSION = 0
ROUTED_SCALING_FACTOR = 1.0
NEED_RENORM = True


def _run_reference(gating: torch.Tensor, bias: torch.Tensor, topk: int):
    """Run PyTorch reference implementation."""
    import aiter
    w_ref, id_ref, _ = aiter.biased_grouped_topk_torch(
        gating, bias, topk, NEED_RENORM, GROUP, TOPK_GROUP, True
    )
    w_ref = w_ref * ROUTED_SCALING_FACTOR
    return w_ref, id_ref


def _run_kernel(gating: torch.Tensor, bias: torch.Tensor, topk: int):
    """Run moe_fused_gate HIP kernel."""
    import aiter
    token = gating.shape[0]
    # moe_fused_gate uses strided tensors (stride = topk + 10) matching
    # the convention in aiter's test_moeTopkSoftmax.py
    w_out = torch.empty_strided(
        (token, topk), (topk + 10, 1),
        dtype=torch.float32, device=gating.device,
    )
    id_out = torch.empty_strided(
        (token, topk), (topk + 10, 1),
        dtype=torch.int32, device=gating.device,
    )
    aiter.moe_fused_gate(
        gating, bias, w_out, id_out,
        GROUP, TOPK_GROUP, topk,
        N_SHARE_EXPERTS_FUSION, ROUTED_SCALING_FACTOR,
    )
    return w_out, id_out


def _compare(w_ref, id_ref, w_kernel, id_kernel):
    """Compare results after sorting by expert id (topk order may differ)."""
    id_ref_s, perm_ref = torch.sort(id_ref, dim=1)
    id_ker_s, perm_ker = torch.sort(id_kernel, dim=1)
    w_ref_s = w_ref.gather(1, perm_ref)
    w_ker_s = w_kernel.gather(1, perm_ker)

    id_match = torch.equal(id_ref_s, id_ker_s)
    w_err = (w_ref_s.float() - w_ker_s.float()).abs().max().item()
    return id_match, w_err


def run_correctness(tokens, experts):
    """Run correctness tests and print results."""
    passed = failed = 0
    for expert in experts:
        for token in tokens:
            tag = f"token={token} expert={expert} group={GROUP} topk={TOPK}"
            try:
                gating = torch.randn(
                    (token, expert), dtype=torch.float32, device="cuda"
                )
                bias = torch.randn((expert,), dtype=torch.float32, device="cuda")

                w_ref, id_ref = _run_reference(gating, bias, TOPK)
                w_ker, id_ker = _run_kernel(gating, bias, TOPK)

                id_match, w_err = _compare(w_ref, id_ref, w_ker, id_ker)
                ok = id_match and w_err < 0.01

                if ok:
                    print(f"[RESULT] {tag} status=PASS err={w_err:.2e}")
                    passed += 1
                else:
                    print(
                        f"[RESULT] {tag} status=FAIL err={w_err:.2e}"
                        f" id_match={id_match}"
                    )
                    failed += 1
            except Exception as e:
                print(f"[RESULT] {tag} status=FAIL error={e}")
                failed += 1
            finally:
                torch.cuda.empty_cache()

    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    return failed == 0


def run_performance(tokens, experts, num_warmup=5, num_iters=20):
    """Run performance benchmarks and print results."""
    results = []
    for expert in experts:
        for token in tokens:
            tag = f"token={token} expert={expert} group={GROUP} topk={TOPK}"
            try:
                gating = torch.randn(
                    (token, expert), dtype=torch.float32, device="cuda"
                )
                bias = torch.randn((expert,), dtype=torch.float32, device="cuda")

                # Warmup
                for _ in range(num_warmup):
                    _run_kernel(gating, bias, TOPK)
                torch.cuda.synchronize()

                # Timed runs
                times_us = []
                for _ in range(num_iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _run_kernel(gating, bias, TOPK)
                    end.record()
                    torch.cuda.synchronize()
                    times_us.append(start.elapsed_time(end) * 1000.0)

                times_us.sort()
                median_us = times_us[len(times_us) // 2]
                print(f"[RESULT] {tag} us={median_us:.2f}")
                results.append({
                    "token": token, "expert": expert, "us": median_us,
                })
            except Exception as e:
                print(f"[RESULT] {tag} us=nan error={e}")
            finally:
                torch.cuda.empty_cache()

    if results:
        avg = sum(r["us"] for r in results) / len(results)
        print(f"\nSUMMARY: {len(results)} cases, avg {avg:.2f} us")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MoE fused gate benchmark (moe_fused_gate.cu)"
    )
    parser.add_argument(
        "--mode", choices=["correctness", "performance"], required=True,
    )
    parser.add_argument(
        "--token", type=int, nargs="+", default=[1, 4, 16, 64],
    )
    parser.add_argument(
        "--expert", type=int, nargs="+", default=[128, 256],
    )
    args = parser.parse_args()

    if args.mode == "correctness":
        ok = run_correctness(args.token, args.expert)
        sys.exit(0 if ok else 1)
    else:
        results = run_performance(args.token, args.expert)
        sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
