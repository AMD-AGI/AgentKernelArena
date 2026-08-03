#!/usr/bin/env python3
"""Task runner for triton2flydsl/aiter/mha (FORWARD only).

Self-contained harness mirroring the triton2flydsl/unified_attention template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the triton forward on TEST_SHAPES, assert finite output
                   (fp16), causal + non-causal, incl. GQA. No torch comparison:
                   the flydsl-vs-triton comparison is added when the FlyDSL target
                   lands (the Triton kernel is the reference here).
  - performance  : graph-first GPU timing, write build/performance_report.json

The kernel under test is the Triton MHA forward (`_attn_fwd`). Forward-only:
the backward kernels were intentionally dropped (out of scope for this task).
Public entry: `flash_attn_func(q, k, v, softmax_scale, causal, window_size, ...)`.
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

TASK_NAME = "triton2flydsl/aiter/mha"
SOURCE_FILE = os.path.join(TASK_DIR, "mha.py")

# Test configurations:
# (batch, seqlen, num_query_heads, num_kv_heads, head_size, causal)
# seqlen_q == seqlen_k keeps causal masking aligned with torch SDPA is_causal.
TEST_SHAPES = [
    (2, 128, 8, 8, 64, False),    # MHA, non-causal
    (2, 128, 8, 8, 64, True),     # MHA, causal
    (1, 256, 16, 16, 128, True),  # MHA, causal, hs=128
    (4, 64, 16, 4, 64, True),     # GQA (4:1), causal
    (1, 512, 8, 2, 128, False),   # GQA (4:1), non-causal, hs=128
    (2, 384, 12, 12, 64, True),   # non-power-of-2 heads, causal
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

MAX_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("mha_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(batch, seqlen, nqh, nkvh, hs, device="cuda", dtype=None):
    """Q,K,V in bshd layout: [batch, seqlen, nheads, head_size]."""
    import torch
    if dtype is None:
        dtype = torch.float16
    q = torch.randn(batch, seqlen, nqh, hs, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, nkvh, hs, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, nkvh, hs, device=device, dtype=dtype)
    scale = 1.0 / (hs ** 0.5)
    return q, k, v, scale


def _call_kernel(mod, q, k, v, scale, causal):
    return mod.flash_attn_func(
        q, k, v,
        softmax_scale=scale,
        causal=causal,
        window_size=(-1, -1),
    )


# Numerical gate: pass if normalized max error <= this.
NORM_ERR_TOL = 1e-2
# Also reported (not gating): allclose(atol=rtol=ALLCLOSE_TOL).
ALLCLOSE_TOL = 1e-2


def torch_mha_ref(q, k, v, scale, causal):
    """fp32-upcast torch reference for the forward MHA the harness invokes.

    Ported from aiter/op_tests/test_mha.py::run_torch -> attention_ref
    (aiter/aiter/test_mha_common.py::attention_ref) for the EXACT config the
    harness uses: bshd layout [batch, seqlen, nheads, head_dim], dense (no
    query/key padding), no bias, no alibi, no dropout, no sink, no sliding
    window. Only the causal flag, softmax scale, and GQA head-repetition are
    configurable (mirrors the flash_attn_func call in _call_kernel).

      * scale: the same softmax_scale passed to the kernel (== 1/sqrt(head_dim)),
        applied as (q * scale) @ k^T -> exactly attention_ref's q/sqrt(d) since
        head_dim == d.
      * GQA: k/v have nkvh heads, repeated g = nqh // nkvh times along the head
        axis (attention_ref uses einops `repeat b s h d -> b s (h g) d`;
        repeat_interleave on the head dim is the equivalent grouping).
      * causal: attention_ref maps causal -> window_size=(-1, 0), i.e.
        mask where col_idx > row_idx + seqlen_k - seqlen_q (bottom-right
        aligned). seqlen_q == seqlen_k here, so this is the standard lower-tri.
    """
    import torch

    q_ = q.float()
    k_ = k.float()
    v_ = v.float()
    b, sq, hq, d = q_.shape
    _, sk, hk, dv = v_.shape
    g = hq // hk
    if g != 1:
        k_ = k_.repeat_interleave(g, dim=2)
        v_ = v_.repeat_interleave(g, dim=2)

    scores = torch.einsum("bthd,bshd->bhts", q_ * scale, k_)
    if causal:
        row = torch.arange(sq, device=q.device).view(sq, 1)
        col = torch.arange(sk, device=q.device).view(1, sk)
        local_mask = col > (row + sk - sq)
        scores = scores.masked_fill(local_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", attn, v_)
    return out.to(q.dtype)


def _compare(ref, out):
    """Return (norm_err, max_abs_err, allclose) comparing kernel out vs ref."""
    import torch

    ref_f = ref.float()
    out_f = out.float()
    denom = ref_f.abs().max().item()
    max_abs = (out_f - ref_f).abs().max().item()
    norm_err = max_abs / denom if denom > 0 else max_abs
    allclose = bool(
        torch.allclose(out_f, ref_f, atol=ALLCLOSE_TOL, rtol=ALLCLOSE_TOL)
    )
    return norm_err, max_abs, allclose


def _with_oom_retry(fn):
    """Retry on transient CUDA OOM (other workers share the GPU)."""
    import torch
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            last = e
            torch.cuda.empty_cache()
            time.sleep(2.0 * (attempt + 1))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                last = e
                torch.cuda.empty_cache()
                time.sleep(2.0 * (attempt + 1))
            else:
                raise
    raise last


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "flash_attn_func"), "Missing flash_attn_func entry"
        assert hasattr(mod, "_attn_fwd"), "Missing _attn_fwd kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    dtype = torch.float16
    details = []

    for i, (batch, seqlen, nqh, nkvh, hs, causal) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, k, v, scale = make_test_data(batch, seqlen, nqh, nkvh, hs, device, dtype)

            result = _with_oom_retry(lambda: _call_kernel(mod, q, k, v, scale, causal))
            torch.cuda.synchronize()

            finite = bool(torch.isfinite(result).all().item())

            ref = torch_mha_ref(q, k, v, scale, causal)
            norm_err, max_abs, allclose = _compare(ref, result)
            numeric_ok = finite and (norm_err <= NORM_ERR_TOL)
            ok = numeric_ok
            details.append({
                "shape_id": i + 1,
                "shape": [batch, seqlen, nqh, nkvh, hs, causal],
                "out_shape": list(result.shape),
                "finite": finite,
                "norm_err": norm_err,
                "max_abs_err": max_abs,
                "allclose@1e-2": allclose,
                "passed": ok,
            })
            if not finite:
                return False, f"Shape {i+1} {TEST_SHAPES[i]}: non-finite output", details
            if not numeric_ok:
                return (
                    False,
                    f"Shape {i+1} {TEST_SHAPES[i]}: norm_err={norm_err:.3e} "
                    f"> tol={NORM_ERR_TOL:.1e} (max_abs={max_abs:.3e})",
                    details,
                )
        except Exception as e:
            details.append({
                "shape_id": i + 1,
                "shape": [batch, seqlen, nqh, nkvh, hs, causal],
                "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details

    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (batch, seqlen, nqh, nkvh, hs, causal) in enumerate(TEST_SHAPES):
        params = {
            "batch": batch, "seqlen": seqlen, "num_query_heads": nqh,
            "num_kv_heads": nkvh, "head_size": hs, "causal": causal,
        }
        try:
            torch.manual_seed(42 + test_idx)
            q, k, v, scale = make_test_data(batch, seqlen, nqh, nkvh, hs, device, dtype)

            for _ in range(WARMUP_ITERATIONS):
                _with_oom_retry(lambda: _call_kernel(mod, q, k, v, scale, causal))
            torch.cuda.synchronize()

            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                lambda: _call_kernel(mod, q, k, v, scale, causal),
                warmup=0,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **bench_meta,
                "params": params,
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": "performance case failed before timing completed",
                "params": params,
            })
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
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
            "details": details,
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "finite" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: out={d['out_shape']} "
                      f"finite={d['finite']} norm_err={d['norm_err']:.3e} "
                      f"max_abs={d['max_abs_err']:.3e} allclose@1e-2={d['allclose@1e-2']} "
                      f"-> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
