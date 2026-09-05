#!/usr/bin/env python3
"""Task runner for triton2flydsl/aiter/unified_attention.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the triton kernel on TEST_SHAPES, assert finite output (bf16)
  - performance  : graph-first GPU timing, write build/performance_report.json

Paged "unified attention". Public entry: `unified_attention(...)`; @triton.jit
kernels: `kernel_unified_attention_2d`, `kernel_unified_attention_3d`,
`reduce_segments`. Inputs are bf16 (the dtype the 3D config path supports).

The flydsl-vs-triton comparison will be added when the FlyDSL target lands.
"""
import sys
import os
import json
import argparse
import importlib.util
from _aka_benchmark import benchmark_cuda_graph_or_events

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/aiter/unified_attention"
SOURCE_FILE = os.path.join(TASK_DIR, "unified_attention.py")

# Test configurations:
# (num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads, head_size,
#  block_size, sliding_window, softcap)
# Shapes are chosen to exercise both the 2D kernel (seq_len_k <= 512 or sliding
# window or decode) and the 3D split-segment kernel + reduce_segments
# (seq_len_k > 512, no sliding window).
TEST_SHAPES = [
    (2, 4, 64, 8, 8, 64, 16, 0, 0.0),       # 2D, short prefill
    (1, 1, 128, 16, 4, 64, 16, 0, 0.0),     # 2D, decode (ALL_DECODE), GQA
    (4, 1, 256, 32, 8, 128, 16, 0, 0.0),    # 2D, decode, GQA, hs=128
    (2, 8, 128, 16, 16, 64, 32, 32, 0.0),   # 2D, sliding window
    (1, 16, 512, 8, 8, 128, 32, 64, 1.0),   # 2D, sliding window + softcap
    (1, 8, 1024, 8, 8, 128, 32, 0, 0.0),    # 3D split-segment + reduce
    (1, 16, 768, 8, 8, 64, 32, 0, 1.0),     # 3D split-segment + softcap
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def load_module():
    spec = importlib.util.spec_from_file_location("unified_attention_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _window_size(sliding_window):
    """AITER computes SLIDING_WINDOW = 1 + window_size[0].

    window_size = (-1, -1)  -> SLIDING_WINDOW = 0  (disabled)
    window_size = (W-1, 0)  -> SLIDING_WINDOW = W
    """
    if sliding_window and sliding_window > 0:
        return (sliding_window - 1, 0)
    return (-1, -1)


def make_test_data(num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads,
                   head_size, block_size, device="cuda", dtype=None):
    """Create test tensors for unified_attention (paged KV cache)."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16

    total_tokens = num_seqs * seq_len_q

    # Packed Q tensor: [num_tokens, num_query_heads, head_size]
    q = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)

    # Paged KV cache: [num_blocks, block_size, num_kv_heads, head_size]
    num_blocks_per_seq = (seq_len_k + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq + 4  # extra padding blocks
    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size,
                            device=device, dtype=dtype)
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size,
                              device=device, dtype=dtype)

    # Block table: each seq uses contiguous blocks
    block_table = torch.zeros(num_seqs, num_blocks_per_seq, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_table[s, b] = s * num_blocks_per_seq + b

    # cu_seqlens_q: cumulative query lengths [num_seqs + 1]
    cu_seqlens_q = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        cu_seqlens_q[s + 1] = cu_seqlens_q[s] + seq_len_q

    # seqused_k: K sequence lengths [num_seqs]
    seqused_k = torch.full((num_seqs,), seq_len_k, device=device, dtype=torch.int32)

    out = torch.empty_like(q)
    scale = 1.0 / (head_size ** 0.5)

    return q, key_cache, value_cache, out, block_table, cu_seqlens_q, seqused_k, scale


def _call_kernel(mod, q, key_cache, value_cache, out, block_table, cu_seqlens_q,
                 seqused_k, scale, seq_len_q, seq_len_k, sliding_window, softcap):
    return mod.unified_attention(
        q,
        key_cache,
        value_cache,
        out,
        cu_seqlens_q,
        seq_len_q,            # max_seqlen_q
        seqused_k,
        seq_len_k,            # max_seqlen_k
        scale,                # softmax_scale
        True,                 # causal
        _window_size(sliding_window),
        block_table,
        float(softcap),
        None,                 # q_descale
        None,                 # k_descale
        None,                 # v_descale
    )


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "unified_attention"), "Missing unified_attention entry"
        assert hasattr(mod, "kernel_unified_attention_2d"), "Missing kernel_unified_attention_2d"
        assert hasattr(mod, "kernel_unified_attention_3d"), "Missing kernel_unified_attention_3d"
        assert hasattr(mod, "reduce_segments"), "Missing reduce_segments"
        return True, None
    except Exception as e:
        return False, str(e)


def ref_paged_attn(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    out_dtype,
    sliding_window=None,
    soft_cap=None,
    sinks=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    output_scale=None,
    causal=1,
):
    """Torch reference for paged unified attention.

    Ported verbatim from aiter/op_tests/triton_tests/attention/
    test_unified_attention.py::ref_paged_attn so it mirrors the exact
    causal / GQA / sliding-window / softcap semantics the triton kernel
    implements. All accumulation happens in fp32; output is cast to
    ``out_dtype`` at the end, matching the kernel's bf16 store path.
    """
    import torch

    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    start_idx = 0
    query = query.to(torch.float32)
    key_cache = key_cache.to(torch.float32)
    value_cache = value_cache.to(torch.float32)
    if q_descale is not None:
        query = query * q_descale
    if k_descale is not None:
        key_cache = key_cache * k_descale
    if v_descale is not None:
        value_cache = value_cache * v_descale
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if causal:
            attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks[:, None, None].repeat_interleave(attn.shape[-2], dim=-2)
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sinks is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    out = torch.cat(outputs, dim=0)
    if output_scale is not None:
        out = out / output_scale

    return out.to(out_dtype)


def _norm_max_error(ref, out):
    """Normalized max error: max|ref-out| / max(|ref|, eps), computed in fp32."""
    import torch

    ref = ref.to(torch.float32)
    out = out.to(torch.float32)
    denom = ref.abs().max().item()
    denom = denom if denom > 1e-12 else 1e-12
    max_abs = (ref - out).abs().max().item()
    return max_abs / denom, max_abs, denom


# Tight numerical gate (bf16). Do NOT loosen.
NORM_ERR_TOL = 1e-2
ALLCLOSE_ATOL = 1e-2
ALLCLOSE_RTOL = 1e-2


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    dtype = torch.bfloat16
    details = []
    all_pass = True
    first_err = None

    for i, (num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, sliding_window, softcap) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, key_cache, value_cache, out, block_table, cu_seqlens_q, seqused_k, scale = \
                make_test_data(num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, device, dtype)

            result = _call_kernel(
                mod, q, key_cache, value_cache, out, block_table, cu_seqlens_q,
                seqused_k, scale, seq_len_q, seq_len_k, sliding_window, softcap,
            )
            torch.cuda.synchronize()

            finite = bool(torch.isfinite(result.float()).all().item())

            # Numerical reference matching the EXACT kernel config:
            #  causal=True, GQA via repeat_interleave, softmax_scale=1/sqrt(hs),
            #  paged KV via block_table, sliding window = SLIDING_WINDOW value,
            #  softcap, bf16, no sinks / no descales (harness passes None).
            ref = ref_paged_attn(
                query=q,
                key_cache=key_cache,
                value_cache=value_cache,
                query_lens=[seq_len_q] * num_seqs,
                kv_lens=[seq_len_k] * num_seqs,
                block_tables=block_table,
                scale=scale,
                out_dtype=dtype,
                sliding_window=(sliding_window if sliding_window and sliding_window > 0 else None),
                soft_cap=(softcap if softcap and softcap > 0 else None),
                causal=1,
            )

            norm_err, max_abs, denom = _norm_max_error(ref, result)
            allclose = bool(torch.allclose(
                result.to(torch.float32), ref.to(torch.float32),
                atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL,
            ))
            numeric_ok = norm_err <= NORM_ERR_TOL
            passed = bool(finite and numeric_ok)
            all_pass = all_pass and passed

            details.append({
                "shape_id": i + 1,
                "shape": [num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, sliding_window, softcap],
                "finite": finite,
                "norm_max_error": norm_err,
                "max_abs_error": max_abs,
                "ref_absmax": denom,
                "allclose@1e-2": allclose,
                "passed": passed,
            })
            if not passed and first_err is None:
                if not finite:
                    first_err = f"Shape {i+1} {TEST_SHAPES[i]}: non-finite output"
                else:
                    first_err = (f"Shape {i+1} {TEST_SHAPES[i]}: norm_max_error="
                                 f"{norm_err:.3e} > {NORM_ERR_TOL:.0e}")
        except Exception as e:
            details.append({
                "shape_id": i + 1,
                "shape": [num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, sliding_window, softcap],
                "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details

    return all_pass, first_err, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.bfloat16
    test_cases = []

    for test_idx, (num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, sliding_window, softcap) in enumerate(TEST_SHAPES):
        params = {
            "num_seqs": num_seqs, "seq_len_q": seq_len_q, "seq_len_k": seq_len_k,
            "num_query_heads": nqh, "num_kv_heads": nkvh, "head_size": hs,
            "block_size": bs, "sliding_window": sliding_window, "softcap": softcap,
        }
        try:
            torch.manual_seed(42 + test_idx)
            q, key_cache, value_cache, out, block_table, cu_seqlens_q, seqused_k, scale = \
                make_test_data(num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, device, dtype)

            for _ in range(WARMUP_ITERATIONS):
                _call_kernel(mod, q, key_cache, value_cache, out, block_table,
                             cu_seqlens_q, seqused_k, scale, seq_len_q, seq_len_k,
                             sliding_window, softcap)
            torch.cuda.synchronize()

            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                lambda: _call_kernel(mod, q, key_cache, value_cache, out, block_table, cu_seqlens_q, seqused_k, scale, seq_len_q, seq_len_k, sliding_window, softcap),
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
                print(f"  shape {d['shape_id']} {d['shape']}: finite={d['finite']} "
                      f"norm_max_err={d['norm_max_error']:.3e} "
                      f"(max_abs={d['max_abs_error']:.3e}, ref_absmax={d['ref_absmax']:.3e}) "
                      f"allclose@1e-2={d['allclose@1e-2']} "
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
