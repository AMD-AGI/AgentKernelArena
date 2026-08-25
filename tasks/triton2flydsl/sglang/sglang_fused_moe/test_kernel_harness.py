#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/sglang_fused_moe.

Standalone harness for sglang's Triton fused MoE grouped-GEMM kernel
(``fused_moe_kernel``), the editable kernel the Qwen3.5-35B-A3B expert FFN
lowers to when aiter is OFF. The ``fused_moe`` host driver runs the two GEMM
launches (gate/up then down) with a host-side SiLU-and-mul in between and a
top-k combine, matching sglang's ``_fused_moe_kernel_sequence``.

Real Qwen3.5-35B-A3B MoE config (config.json):
  hidden_size=2048, moe_intermediate_size=512, num_experts=256,
  num_experts_per_tok=8, dtype=bfloat16, hidden_act=silu (gated).
  -> w1: [256, 1024, 2048], w2: [256, 2048, 512]. M (tokens) is swept.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton fused_moe vs torch fp32 reference, assert close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json
"""
import sys
import os
import json
import time
import argparse
import importlib.util
from _aka_benchmark import benchmark_cuda_graph_or_events

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/sglang/sglang_fused_moe"
SOURCE_FILE = os.path.join(TASK_DIR, "sglang_fused_moe.py")

# Qwen3.5-35B-A3B MoE: K(hidden)=2048, I(moe_inter)=512, E=256, topk=8.
K_HIDDEN = 2048
I_INTER = 512
N_EXPERTS = 256
TOPK = 8

# Test configs: M (number of tokens). decode -> small M; prefill -> large M.
TEST_SHAPES = [
    (16, K_HIDDEN, I_INTER, N_EXPERTS, TOPK),     # decode-ish batch
    (64, K_HIDDEN, I_INTER, N_EXPERTS, TOPK),
    (256, K_HIDDEN, I_INTER, N_EXPERTS, TOPK),
    (1024, K_HIDDEN, I_INTER, N_EXPERTS, TOPK),    # chunked prefill
    (2048, K_HIDDEN, I_INTER, N_EXPERTS, TOPK),
    (4096, K_HIDDEN, I_INTER, N_EXPERTS, TOPK),    # large prefill chunk
    (1024, K_HIDDEN, I_INTER, 128, TOPK),          # fewer experts (TP/EP shard)
    (512, K_HIDDEN, I_INTER, N_EXPERTS, 4),        # smaller top-k
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
DTYPE_NAME = os.environ.get("MOE_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("sglang_fused_moe_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err):
    return "out of memory" in str(err).lower()


def _retry_oom(fn):
    import torch
    delay = 1.0
    for attempt in range(MAX_OOM_RETRIES):
        try:
            return fn()
        except RuntimeError as e:
            if _is_oom(e) and attempt < MAX_OOM_RETRIES - 1:
                torch.cuda.empty_cache()
                time.sleep(delay)
                delay *= 2.0
                continue
            raise


def route_topk(logits, topk):
    """Softmax router + top-k with renormalized weights (matches sglang/topk)."""
    import torch
    gate = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(gate, topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float().contiguous(), ids.to(torch.int32).contiguous()


def make_test_data(M, K, I, E, topk, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    hidden = torch.randn(M, K, device=device, dtype=dtype)
    # weights scaled like the GEAK MoE reference (randn/10) to keep bf16 sane.
    w1 = (torch.randn(E, 2 * I, K, device=device, dtype=torch.float32) / 10).to(dtype)
    w2 = (torch.randn(E, K, I, device=device, dtype=torch.float32) / 10).to(dtype)
    logits = torch.randn(M, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = route_topk(logits, topk)
    return {"M": M, "K": K, "I": I, "E": E, "topk": topk,
            "hidden": hidden.contiguous(), "w1": w1.contiguous(),
            "w2": w2.contiguous(), "topk_weights": topk_weights, "topk_ids": topk_ids}


def _make_prepared_fused_moe_runner(mod, inp):
    """Build a capture-safe runner for the fixed benchmark input.

    The public ``fused_moe`` wrapper deliberately includes host-side routing
    preparation (including a device-to-host ``.item()``) and allocates all
    intermediates on every call.  Those are setup costs, not the editable
    Triton grouped-GEMM kernels.  Prepare the routing layout and fixed-shape
    buffers once, then launch the same two kernels plus the same activation and
    top-k combine from the timed callback.
    """
    import torch

    M, K, I, E, topk = (
        inp["M"], inp["K"], inp["I"], inp["E"], inp["topk"]
    )
    dtype = inp["hidden"].dtype
    config = mod._default_config()
    compute_type = mod.tl.bfloat16 if dtype == torch.bfloat16 else mod.tl.float16

    # Data-dependent routing and its host-visible padded size are fixed for a
    # benchmark case, so compute them before capture.
    sorted_ids, expert_ids, num_tokens_post_padded = mod.moe_align_block_size(
        inp["topk_ids"], config["BLOCK_SIZE_M"], E
    )

    # Stable storage for both grouped GEMMs and the intervening pointwise work.
    device = inp["hidden"].device
    ic1 = torch.empty((M * topk, 2 * I), device=device, dtype=dtype)
    gate_fp32 = torch.empty((M * topk, I), device=device, dtype=torch.float32)
    up_fp32 = torch.empty_like(gate_fp32)
    activated_fp32 = torch.empty_like(gate_fp32)
    ic2 = torch.empty((M * topk, I), device=device, dtype=dtype)
    ic3 = torch.empty((M, topk, K), device=device, dtype=dtype)
    ic3_fp32 = torch.empty((M, topk, K), device=device, dtype=torch.float32)
    summed_fp32 = torch.empty((M, K), device=device, dtype=torch.float32)
    out = torch.empty((M, K), device=device, dtype=dtype)

    def run():
        mod.invoke_fused_moe_kernel(
            inp["hidden"],
            inp["w1"],
            ic1,
            inp["topk_weights"],
            inp["topk_ids"],
            sorted_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=topk,
            config=config,
            compute_type=compute_type,
        )

        # Match ``(silu(gate.float()) * up.float()).to(dtype)`` without
        # allocating new tensors inside the timed/captured invocation.
        gate_fp32.copy_(ic1[:, :I])
        up_fp32.copy_(ic1[:, I:])
        torch.nn.functional.silu(gate_fp32, inplace=True)
        torch.mul(gate_fp32, up_fp32, out=activated_fp32)
        ic2.copy_(activated_fp32)

        mod.invoke_fused_moe_kernel(
            ic2,
            inp["w2"],
            ic3,
            inp["topk_weights"],
            inp["topk_ids"],
            sorted_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            config=config,
            compute_type=compute_type,
        )

        # Match ``ic3.float().sum(dim=1).to(dtype)`` with stable outputs.
        ic3_fp32.copy_(ic3)
        torch.sum(ic3_fp32, dim=1, out=summed_fp32)
        out.copy_(summed_fp32)
        return out

    return run, out


def reference_moe(inp):
    """Pure-torch fp32 golden mirroring the two-GEMM gated-SiLU MoE FFN.

    Casts to bf16 at the same points the kernel does (ic1 after GEMM1, ic2 after
    SiLU-and-mul, ic3 after GEMM2+routed-weight) so the comparison is a true
    bf16-accumulation check.
    """
    import torch
    import torch.nn.functional as F
    M, K, I, E, topk = inp["M"], inp["K"], inp["I"], inp["E"], inp["topk"]
    dtype = inp["hidden"].dtype
    h = inp["hidden"].float()
    w1 = inp["w1"].float()
    w2 = inp["w2"].float()
    topk_ids = inp["topk_ids"]
    topk_weights = inp["topk_weights"]
    N = 2 * I

    hrep = h.view(M, 1, K).expand(M, topk, K)
    s1 = torch.zeros(M, topk, N, device=h.device, dtype=torch.float32)
    for e in range(E):
        mask = topk_ids == e
        if mask.any():
            s1[mask] = hrep[mask] @ w1[e].transpose(0, 1)
    s1 = s1.to(dtype)  # ic1 (bf16)

    gate, up = s1[..., :I], s1[..., I:]
    s2 = (F.silu(gate.float()) * up.float()).to(dtype)  # ic2 (bf16)

    s3 = torch.zeros(M, topk, K, device=h.device, dtype=torch.float32)
    s2f = s2.float()
    for e in range(E):
        mask = topk_ids == e
        if mask.any():
            s3[mask] = s2f[mask] @ w2[e].transpose(0, 1)
    s3 = s3 * topk_weights.view(M, topk, 1)
    s3 = s3.to(dtype)  # ic3 (bf16, routed-weight applied)

    out = s3.float().sum(dim=1).to(dtype)
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "fused_moe"), "Missing entry fused_moe"
        assert hasattr(mod, "fused_moe_kernel"), "Missing @triton.jit fused_moe_kernel"
        assert hasattr(mod, "invoke_fused_moe_kernel"), "Missing invoke_fused_moe_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    dtype = getattr(torch, DTYPE_NAME)
    atol = 1e-4 if dtype == torch.float32 else 3e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    max_ratio = 0.0 if dtype == torch.float32 else 0.02
    details = []
    for i, (M, K, I, E, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            inp = make_test_data(M, K, I, E, topk, "cuda", dtype)
            out_t = _retry_oom(lambda: mod.fused_moe(
                inp["hidden"], inp["w1"], inp["w2"],
                inp["topk_weights"], inp["topk_ids"]))
            torch.cuda.synchronize()
            out_r = reference_moe(inp)
            diff = (out_t.float() - out_r.float()).abs().max().item()
            isclose = torch.isclose(out_t.float(), out_r.float(), atol=atol, rtol=rtol)
            err_ratio = (~isclose).float().mean().item()
            passed = err_ratio <= max_ratio
            details.append({"shape_id": i + 1, "shape": [M, K, I, E, topk],
                            "max_diff": diff, "err_ratio": err_ratio,
                            "passed": bool(passed)})
            if not passed:
                return False, (f"Shape {i+1} {TEST_SHAPES[i]}: max_diff={diff:.4e} "
                               f"err_ratio={err_ratio:.4f}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": [M, K, I, E, topk], "error": str(e)})
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    dtype = getattr(torch, DTYPE_NAME)
    test_cases = []
    for ti, (M, K, I, E, topk) in enumerate(TEST_SHAPES):
        params = {"M": M, "K": K, "I": I, "E": E, "topk": topk}
        try:
            torch.manual_seed(42 + ti)
            inp = make_test_data(M, K, I, E, topk, "cuda", dtype)
            fn, _ = _make_prepared_fused_moe_runner(mod, inp)

            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                fn()
            torch.cuda.synchronize()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS
            )
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": elapsed_ms,
                               **bench_meta,
                               "params": params})
        except Exception:
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": -1.0,
                               "benchmark_method": "benchmark_failed",
                               "benchmark_fallback_reason": "performance case failed before timing completed",
                               "params": params})
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(build_dir, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err,
                   "num_shapes": len(TEST_SHAPES), "details": details},
                  open(os.path.join(build_dir, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "passed" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: max_diff={d['max_diff']:.4e} "
                      f"err_ratio={d['err_ratio']:.4f} -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        json.dump(test_cases, open(os.path.join(build_dir, "performance_report.json"), "w"), indent=2)
        if test_cases:
            total = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} case(s), total {total:.4f} ms")
            for c in test_cases:
                print(f"  {c['test_case_id']} {c['params']}: {c['execution_time_ms']:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
