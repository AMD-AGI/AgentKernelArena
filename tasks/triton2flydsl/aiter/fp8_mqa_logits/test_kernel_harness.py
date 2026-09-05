#!/usr/bin/env python3
"""Test harness for the triton2flydsl/aiter/fp8_mqa_logits task.

Loads the standalone Triton source (`fp8_mqa_logits.py`) and runs it over a set
of FP8 MQA-logits shapes.

Modes:
  --compile        ast-parse + import the source, assert entry/kernel symbols exist
  --correctness    run the Triton kernel on TEST_SHAPES, assert finite in-window output
  --full-benchmark graph-first GPU timing, write build/performance_report.json

The flydsl-vs-triton comparison will be added when the FlyDSL target lands.
"""
import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from _aka_benchmark import benchmark_cuda_graph_or_events

SOURCE_FILE = "fp8_mqa_logits.py"
ENTRY = "fp8_mqa_logits"
KERNEL = "_fp8_mqa_logits_kernel"

# (seq_len, seq_len_kv, num_heads, head_size, window)
#   window: "full"    -> [0, seq_len_kv) for every row
#           "causal"  -> row s sees [0, end_s) with end_s growing across rows
#           "band"    -> row s sees a sliding band [start_s, end_s)
# num_heads / head_size are powers of 2 (kernel asserts this). num_heads is the
# MFMA M dim, so it must be a multiple of matrix_instr_nonkdim (16 for
# seq_len<=1024, else 32). seq_len_kv values include non-128 multiples to
# exercise the masked tail.
TEST_SHAPES = [
    (64, 512, 32, 128, "full"),     # basic full window
    (128, 1000, 64, 128, "full"),   # masked tail (1000 % 128 != 0)
    (256, 2048, 32, 64, "causal"),  # causal windows, head_size=64
    (96, 777, 16, 128, "band"),     # sliding band, ragged size
    (2048, 1024, 64, 128, "full"),  # seq_len>1024 -> matrix_instr_nonkdim=32
]

SEED = 20260617
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def _resolve_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, SOURCE_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, SOURCE_FILE)):
        return cwd
    return here


_TASK_DIR = _resolve_dir()


def load_module():
    entry = os.path.join(_TASK_DIR, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("fp8_mqa_logits_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_windows(seq_len, seq_len_kv, window, device):
    import torch

    if window == "full":
        starts = torch.zeros(seq_len, dtype=torch.int32)
        ends = torch.full((seq_len,), seq_len_kv, dtype=torch.int32)
    elif window == "causal":
        # row s sees [0, end_s); end_s grows monotonically to seq_len_kv
        rows = torch.arange(1, seq_len + 1, dtype=torch.float64)
        ends = (rows / seq_len * seq_len_kv).ceil().clamp(1, seq_len_kv).to(torch.int32)
        starts = torch.zeros(seq_len, dtype=torch.int32)
    elif window == "band":
        band = max(seq_len_kv // 3, 1)
        rows = torch.arange(1, seq_len + 1, dtype=torch.float64)
        ends = (rows / seq_len * seq_len_kv).ceil().clamp(1, seq_len_kv).to(torch.int32)
        starts = (ends - band).clamp(min=0).to(torch.int32)
    else:
        raise ValueError(f"unknown window {window!r}")
    return starts.to(device), ends.to(device)


def make_inputs(seq_len, seq_len_kv, num_heads, head_size, window, mod, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    e4m3 = mod.e4m3_dtype

    # Keep magnitudes small so the fp8 (e4m3) quantization of Q/KV does not
    # saturate and the Q.KV dot stays well-conditioned.
    q = (torch.randn(seq_len, num_heads, head_size, generator=gen, device=device) * 0.3).to(e4m3)
    kv = (torch.randn(seq_len_kv, head_size, generator=gen, device=device) * 0.3).to(e4m3)
    kv_scales = (torch.rand(seq_len_kv, generator=gen, device=device, dtype=torch.float32) + 0.5)
    weights = torch.rand(seq_len, num_heads, generator=gen, device=device, dtype=torch.float32)
    cu_starts, cu_ends = _build_windows(seq_len, seq_len_kv, window, device)
    return q, kv, kv_scales, weights, cu_starts, cu_ends


def _window_mask(seq_len_kv, cu_starts, cu_ends, device):
    import torch

    col = torch.arange(seq_len_kv, device=device)[None, :]
    start = cu_starts.clamp(min=0).long()[:, None]
    end = cu_ends.clamp(max=seq_len_kv).long()[:, None]
    return (col >= start) & (col < end)


# --- numerical reference -----------------------------------------------------
# Ported from aiter/op_tests/triton_tests/attention/test_fp8_mqa_logits.py
# (`ref_fp8_mqa_logits` + `calc_diff`) and adapted to the EXACT arguments the
# kernel is invoked with in this harness.
#
# The kernel (_fp8_mqa_logits_kernel) computes, per query row m and kv col n:
#     score[h, n] = dot(Q[m, h, :], KV[n, :])          # fp8 x fp8, fp32 accum
#     score       = score * kv_scales[n]               # scale BEFORE relu
#     score       = relu(score)
#     score       = score * weights[m, h]
#     logits[m, n] = sum_h score[h, n]                 # then -inf outside window
#
# The aiter reference pre-multiplies KV by its scale and applies relu after the
# (already-scaled) score; that is algebraically identical to the above since the
# scale is applied before relu. We mirror the kernel's ordering exactly and feed
# it the SAME fp8 Q/KV tensors (cast to fp32) so the only residual difference is
# fp8 MFMA rounding + fp32 accumulation order, not an input mismatch.
def _ref_fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends):
    import torch

    seq_len_kv = kv.shape[0]
    q_f = q.float()          # [s, h, d]  (exact fp8-represented values)
    kv_f = kv.float()        # [skv, d]
    # score[m, h, n] = sum_d Q[m,h,d] * KV[n,d]
    score = torch.einsum("mhd,nd->mhn", q_f, kv_f)
    score = score * kv_scales.float()[None, None, :]     # scale before relu
    score = torch.relu(score)
    score = score * weights.float()[:, :, None]
    logits = score.sum(dim=1)                            # [s, skv]

    mask = _window_mask(seq_len_kv, cu_starts, cu_ends, logits.device)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


def _calc_diff(x, y):
    # Cosine-style difference used by the aiter reference test (1 - cos_sim).
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denom
    return float((1 - sim).item())


def run_compile():
    try:
        with open(os.path.join(_TASK_DIR, SOURCE_FILE)) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, ENTRY), f"Missing {ENTRY} entry"
        assert hasattr(mod, KERNEL), f"Missing {KERNEL} kernel"
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def run_correctness(verbose=True):
    import torch

    try:
        mod = load_module()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot load {SOURCE_FILE}: {e}")
        return {"correct": False, "details": []}

    # fp8 (e4m3) is a ~2-3 mantissa-bit format, so the Q.KV MFMA dot carries
    # genuine quantization/rounding error. We gate on a normalized-max-error of
    # 5e-2 (the smallest round threshold that comfortably covers fp8 rounding for
    # these shapes without masking a real mismatch) and additionally require the
    # aiter cosine-diff metric to stay small (< 1e-2), matching the reference
    # test's intent. Out-of-window positions are -inf in both and excluded.
    NME_TOL = 5e-2
    DIFF_TOL = 1e-2
    details = []
    failures = []
    for i, (s, skv, h, d, window) in enumerate(TEST_SHAPES):
        try:
            q, kv, kv_scales, weights, cu_starts, cu_ends = make_inputs(
                s, skv, h, d, window, mod
            )
            out = mod.fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends,
                                     clean_logits=True)
            torch.cuda.synchronize()

            # In-window positions must be finite; no NaNs anywhere. Out-of-window
            # positions are intentionally -inf (causal masking), so they are excluded.
            in_window = _window_mask(skv, cu_starts, cu_ends, out.device)
            no_nan = bool((~torch.isnan(out)).all().item())
            in_window_finite = bool(torch.isfinite(out[in_window]).all().item()) if in_window.any() else True

            # Numerical reference comparison (ported aiter ref_fp8_mqa_logits).
            ref = _ref_fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends)
            # The kernel must produce -inf exactly on the out-of-window positions.
            mask_match = bool(torch.equal(out == float("-inf"), ref == float("-inf")))
            if in_window.any():
                a = out[in_window].float()
                b = ref[in_window].float()
                abs_err = (a - b).abs()
                denom = b.abs().max().item()
                nme = float((abs_err.max() / denom).item()) if denom > 0 else float(abs_err.max().item())
                diff = _calc_diff(a, b)
                allclose = bool(torch.allclose(a, b, atol=5e-2, rtol=5e-2))
            else:
                nme, diff, allclose = 0.0, 0.0, True

            numeric_ok = mask_match and (nme <= NME_TOL) and (diff <= DIFF_TOL)
            ok = no_nan and in_window_finite and numeric_ok
            details.append({
                "shape_id": i + 1,
                "shape": [s, skv, h, d, window],
                "no_nan": no_nan,
                "in_window_finite": in_window_finite,
                "mask_match": mask_match,
                "norm_max_err": nme,
                "cos_diff": diff,
                "allclose": allclose,
                "passed": bool(ok),
            })
            if verbose:
                print(f"  {'PASS' if ok else 'FAIL'}: shape {i+1} "
                      f"(seq={s}, kv={skv}, H={h}, D={d}, {window}) "
                      f"no_nan={no_nan} in_window_finite={in_window_finite} "
                      f"mask_match={mask_match} nme={nme:.3e} cos_diff={diff:.3e} "
                      f"allclose={allclose}")
            if not ok:
                failures.append(i + 1)
        except Exception as e:  # noqa: BLE001
            details.append({"shape_id": i + 1, "shape": [s, skv, h, d, window],
                            "error": str(e)})
            failures.append(i + 1)
            if verbose:
                print(f"  FAIL: shape {i+1} (seq={s}, kv={skv}, H={h}, D={d}, {window}) "
                      f"- {str(e)[:120]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(TEST_SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return {"correct": not failures, "details": details}


def run_benchmark(verbose=True):
    import torch

    try:
        mod = load_module()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot load {SOURCE_FILE}: {e}")
        return {"geomean_latency_ms": -1.0}

    report = []
    latencies = []
    print(f"{'shape (seq,kv,H,D,win)':<34} {'latency(ms)':>12}")
    print("-" * 48)
    for idx, (s, skv, h, d, window) in enumerate(TEST_SHAPES):
        try:
            q, kv, kv_scales, weights, cu_starts, cu_ends = make_inputs(
                s, skv, h, d, window, mod
            )
            for _ in range(WARMUP_ITERATIONS):
                mod.fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends)
            torch.cuda.synchronize()

            ms, bench_meta = benchmark_cuda_graph_or_events(
                lambda: mod.fp8_mqa_logits(
                    q, kv, kv_scales, weights, cu_starts, cu_ends
                ),
                warmup=0,
                repetition=BENCHMARK_ITERATIONS,
            )
        except Exception as e:  # noqa: BLE001
            ms = -1.0
            bench_meta = {
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": str(e),
            }
            if verbose:
                print(f"  shape {idx+1} error: {str(e)[:120]}")
        latencies.append(ms)
        report.append({
            "test_case_id": f"perf{idx + 1}",
            "execution_time_ms": ms,
            **bench_meta,
            "params": {"seq_len": s, "seq_len_kv": skv, "num_heads": h,
                       "head_size": d, "window": window},
        })
        if verbose:
            print(f"(seq={s:>5}, kv={skv:>5}, H={h:>3}, D={d:>3}, {window:<6}) {ms:>12.4f}")
        del q, kv, kv_scales, weights, cu_starts, cu_ends
        torch.cuda.empty_cache()

    build_dir = Path(_TASK_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    valid = [x for x in latencies if x > 0]
    geomean = math.exp(sum(math.log(x) for x in valid) / len(valid)) if valid else -1.0
    print("-" * 48)
    print(f"Geometric mean latency: {geomean:.4f} ms ({len(valid)}/{len(TEST_SHAPES)} measured)")
    return {"geomean_latency_ms": geomean}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triton2flydsl fp8_mqa_logits harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 48)
    print("triton2flydsl FP8 MQA logits")
    print("=" * 48)

    if args.compile:
        ok, err = run_compile()
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.correctness:
        result = run_correctness()
        sys.exit(0 if result.get("correct", False) else 1)
    else:
        run_benchmark()
        sys.exit(0)
