#!/usr/bin/env python3
"""Task runner for triton2flydsl/aiter/fav3_sage.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the triton source on TEST_SHAPES, assert finite output. No torch
                   comparison: the flydsl-vs-triton comparison is added when the FlyDSL
                   target lands (the Triton kernel is the reference here).
  - performance  : graph-first GPU timing, write build/performance_report.json

The kernel under test is SageAttention v1 (INT8 Q/K + FP8 V flash attention).
Public entry: `fav3_sage(q, k, v, ...)` (high-precision BF16/FP16/FP32 in -> BF16 out;
quantizes internally). @triton.jit kernels: `sage_fwd` (attention) and
`sage_quant_kernel` (smooth-K + per-block INT8 Q/K + per-channel FP8 V quantization).
"""
import sys
import os
import json
import argparse
import importlib.util
from _aka_benchmark import benchmark_cuda_graph_or_events

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/aiter/fav3_sage"
SOURCE_FILE = os.path.join(TASK_DIR, "fav3_sage.py")

# Small kernel config so several seqlen blocks are exercised cheaply on a shared GPU.
# BLKQ == BLOCK_M and BLKK == BLOCK_N must stay consistent (the descale tables are
# indexed per BLOCK_M / BLOCK_N block).
CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "waves_per_eu": 2,
    "PRE_LOAD_V": False,
    "num_stages": 2,
    "num_warps": 4,
}

# (batch, seqlen, num_q_heads, num_kv_heads, head_dim, causal, window)
# window > 0 selects a causal sliding window of `window` keys; 0 disables it.
TEST_SHAPES = [
    (1, 64, 4, 4, 64, True, 0),     # single block, causal, MHA, d=64
    (1, 128, 8, 8, 64, True, 0),    # 2 blocks, causal
    (2, 128, 8, 2, 64, True, 0),    # GQA causal
    (1, 128, 8, 8, 128, True, 0),   # d=128, causal
    (1, 128, 8, 8, 64, False, 0),   # non-causal (full) attention
    (1, 256, 8, 8, 64, True, 0),    # 4 blocks, causal
    (2, 128, 8, 8, 64, True, 32),   # causal sliding window (32 keys)
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def load_module():
    spec = importlib.util.spec_from_file_location("fav3_sage_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(batch, seqlen, hq, hk, d, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, hq, d, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, hk, d, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, hk, d, device=device, dtype=dtype)
    scale = 1.0 / (d ** 0.5)
    return q, k, v, scale


def _window_size(window):
    """Causal sliding window of `window` keys -> (left=window-1, right=0)."""
    if window and window > 0:
        return (window - 1, 0)
    return (-1, -1)


def _call_kernel(mod, q, k, v, scale, causal, window):
    return mod.fav3_sage(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        window_size=_window_size(window),
        layout="bshd",
        return_lse=False,
        smooth_k=True,
        config=dict(CONFIG),
    )


# ---------------------------------------------------------------------------
# Numerical torch reference (ported from aiter op_tests)
# ---------------------------------------------------------------------------
# Ported from /workspaces/meta/aiter/op_tests/triton_tests/attention/test_fav3_sage.py
# (test_sage), which validates SageAttention against `attention_ref` from
# aiter.test_mha_common. `attention_ref` is plain full-precision SDPA (with
# causal / sliding-window / GQA support); SageAttention's smooth-K only shifts
# every qk row by a per-row constant (subtracting mean(K)), which is invariant
# under softmax, so the correct numerical reference is un-quantized attention.
#
# The kernel here is called with a REAL softmax_scale, per-shape causal flag,
# a causal sliding window, GQA (hq != hk) and layout="bshd" -- this reference
# mirrors EXACTLY that config (see _attention_reference args / _window_size).
#
# `construct_local_mask` is ported verbatim (single-batch, no padding) from
# aiter.test_mha_common.construct_local_mask; the causal `window_size = (w0, 0)`
# collapse matches attention_ref's `if causal: window_size = (window_size[0], 0)`.


def _construct_local_mask(seqlen_q, seqlen_k, window_size, device):
    """Verbatim port of aiter.test_mha_common.construct_local_mask.

    Returns a boolean (seqlen_q, seqlen_k) mask, True = position disallowed.
    """
    import torch
    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k
    sq = seqlen_q
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk_full = torch.full_like(col_idx, seqlen_k)
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk_full),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def _attention_reference(q, k, v, softmax_scale, causal, window):
    """Full-precision SDPA reference matching attention_ref + the harness config.

    q, k, v: bshd high-precision tensors (B, S, H, D). GQA (hq % hk == 0)
    supported. Returns bshd output like the kernel.
    """
    import torch
    qf = q.float().permute(0, 2, 1, 3)  # (B, Hq, Sq, D)
    kf = k.float().permute(0, 2, 1, 3)  # (B, Hk, Sk, D)
    vf = v.float().permute(0, 2, 1, 3)  # (B, Hk, Sk, D)

    hq, hk = qf.shape[1], kf.shape[1]
    if hq != hk:
        assert hq % hk == 0, f"GQA ratio must be integer: hq={hq} hk={hk}"
        g = hq // hk
        kf = kf.repeat_interleave(g, dim=1)
        vf = vf.repeat_interleave(g, dim=1)

    seqlen_q, seqlen_k = qf.shape[2], kf.shape[2]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_scale  # (B, Hq, Sq, Sk)

    # Mirror attention_ref window handling exactly.
    ws = list(_window_size(window))
    if causal:
        ws = [ws[0], 0]
    use_mask = ws[0] >= 0 or ws[1] >= 0
    if use_mask:
        local_mask = _construct_local_mask(seqlen_q, seqlen_k, ws, qf.device)
        scores = scores.masked_fill(local_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    if use_mask:
        # Rows fully masked out -> 0 (avoid NaN), matching attention_ref.
        attn = attn.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)

    out = torch.matmul(attn, vf)  # (B, Hq, Sq, D)
    return out.permute(0, 2, 1, 3).contiguous()  # bshd


# Tolerance for the quantized SageAttention output vs the full-precision
# reference. SageAttention quantizes Q/K to per-block INT8 and V to per-channel
# FP8, so the output carries real quantization noise. The upstream test
# (test_fav3_sage.py::test_sage) accepts element-wise atol=3e-1 / rtol=2.5e-1
# with up to 0.5% of elements exceeding it (fp8_assert_close). We keep that as
# the primary element-wise gate AND add a stricter normalized-max-error gate
# (max|out-ref| / max|ref|) to catch gross structural errors that a per-element
# outlier budget would miss.
ATOL_FP8 = 3.0e-1
RTOL_FP8 = 2.5e-1
MAX_DIFF_PCT = 0.5
NORM_MAX_ERR_TOL = 5.0e-2


def _fp8_frac_exceeding(current, reference, atol, rtol):
    """Fraction (%) of elements exceeding both atol and rtol (fp8_assert_close)."""
    import torch
    abs_diff = torch.abs(current - reference)
    rel_diff = abs_diff / torch.abs(reference.clamp(min=1e-6))
    failed = torch.logical_and(abs_diff > atol, rel_diff > rtol)
    return failed.sum().item() / failed.numel() * 100.0


def _compare(result, reference):
    """Return a dict of numerical metrics comparing kernel vs reference."""
    import torch
    cur = result.float()
    ref = reference.float()
    abs_diff = torch.abs(cur - ref)
    ref_absmax = ref.abs().max().item()
    norm_max_err = abs_diff.max().item() / (ref_absmax + 1e-12)
    frac_pct = _fp8_frac_exceeding(cur, ref, ATOL_FP8, RTOL_FP8)
    ref_flat = ref.reshape(-1)
    cur_flat = cur.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), cur_flat.unsqueeze(0)
    ).item()
    return {
        "max_abs_err": abs_diff.max().item(),
        "mean_abs_err": abs_diff.mean().item(),
        "ref_absmax": ref_absmax,
        "norm_max_err": norm_max_err,
        "frac_exceeding_pct": frac_pct,
        "cosine_sim": cos,
    }


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fav3_sage"), "Missing fav3_sage entry"
        assert hasattr(mod, "sage_fwd"), "Missing sage_fwd kernel"
        assert hasattr(mod, "sage_quant_kernel"), "Missing sage_quant_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    # Runs the Triton kernel on TEST_SHAPES and compares against a full-precision
    # torch attention reference (ported from aiter test_fav3_sage.py::test_sage),
    # matching the EXACT config the kernel is invoked with (softmax_scale, causal,
    # sliding window, GQA, bshd layout). Keeps the finite check; the numerical gate
    # is a normalized-max-error bound plus the upstream fp8 element-wise budget.
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    details = []
    failures = []

    for i, (b, s, hq, hk, d, causal, window) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, k, v, scale = make_test_data(b, s, hq, hk, d, device)

            result = _call_kernel(mod, q, k, v, scale, causal, window)
            torch.cuda.synchronize()

            finite = bool(torch.isfinite(result).all().item())

            reference = _attention_reference(q, k, v, scale, causal, window)
            assert tuple(result.shape) == tuple(reference.shape), (
                f"shape mismatch: kernel {tuple(result.shape)} vs "
                f"reference {tuple(reference.shape)}"
            )
            m = _compare(result, reference)

            num_ok = m["norm_max_err"] <= NORM_MAX_ERR_TOL and (
                m["frac_exceeding_pct"] <= MAX_DIFF_PCT
            )
            passed = bool(finite and num_ok)

            details.append({
                "shape_id": i + 1,
                "shape": [b, s, hq, hk, d, causal, window],
                "out_shape": list(result.shape),
                "finite": finite,
                "norm_max_err": m["norm_max_err"],
                "max_abs_err": m["max_abs_err"],
                "mean_abs_err": m["mean_abs_err"],
                "ref_absmax": m["ref_absmax"],
                "frac_exceeding_pct": m["frac_exceeding_pct"],
                "cosine_sim": m["cosine_sim"],
                "norm_max_err_tol": NORM_MAX_ERR_TOL,
                "passed": passed,
            })
            if not finite:
                failures.append(f"Shape {i+1} {TEST_SHAPES[i]}: non-finite output")
            elif not num_ok:
                failures.append(
                    f"Shape {i+1} {TEST_SHAPES[i]}: numerical mismatch "
                    f"norm_max_err={m['norm_max_err']:.4e} (tol {NORM_MAX_ERR_TOL:.1e}) "
                    f"frac_exceeding={m['frac_exceeding_pct']:.4f}% (tol {MAX_DIFF_PCT}%)"
                )
        except Exception as e:
            import traceback
            details.append({
                "shape_id": i + 1,
                "shape": [b, s, hq, hk, d, causal, window],
                "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}\n{traceback.format_exc()}", details

    if failures:
        return False, "; ".join(failures), details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (b, s, hq, hk, d, causal, window) in enumerate(TEST_SHAPES):
        params = {
            "batch": b, "seqlen": s, "num_q_heads": hq, "num_kv_heads": hk,
            "head_dim": d, "causal": causal, "window": window,
        }
        try:
            torch.manual_seed(42 + test_idx)
            q, k, v, scale = make_test_data(b, s, hq, hk, d, device)

            for _ in range(WARMUP_ITERATIONS):
                _call_kernel(mod, q, k, v, scale, causal, window)
            torch.cuda.synchronize()

            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                lambda: _call_kernel(mod, q, k, v, scale, causal, window),
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
        for dd in details:
            if "finite" in dd:
                print(f"  shape {dd['shape_id']} {dd['shape']}: out={dd['out_shape']} "
                      f"finite={dd['finite']} norm_max_err={dd['norm_max_err']:.4e} "
                      f"frac_exceeding={dd['frac_exceeding_pct']:.4f}% "
                      f"cos={dd['cosine_sim']:.6f} -> {'PASS' if dd['passed'] else 'FAIL'}")
            elif "error" in dd:
                print(f"  shape {dd['shape_id']} {dd['shape']}: ERROR {dd['error']}")
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
