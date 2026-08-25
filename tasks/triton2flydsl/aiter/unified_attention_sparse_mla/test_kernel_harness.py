#!/usr/bin/env python3
"""Task runner for triton2flydsl/aiter/unified_attention_sparse_mla.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the triton kernel on TEST_SHAPES, assert finite output (bf16)
  - performance  : graph-first GPU timing, write build/performance_report.json

Public entry: `unified_attention_sparse_mla(...)`; @triton.jit
kernel: `_kernel_unified_attention_sparse_mla_2d`.

The flydsl-vs-triton comparison will be added when the FlyDSL target lands.
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

TASK_NAME = "triton2flydsl/aiter/unified_attention_sparse_mla"
SOURCE_FILE = os.path.join(TASK_DIR, "unified_attention_sparse_mla.py")

# Test configurations:
# (num_seqs, tokens_per_seq, num_query_heads, kv_lora_rank, rope_rank,
#  block_size, num_blocks, topk)
# num_query_heads must be a multiple of BLOCK_M (16). kv_lora_rank, rope_rank
# and block_size must be powers of two (Triton arange constraints). block_size
# doubles as the KV-cache block size AND the top-k tile size (TILE_SIZE).
TEST_SHAPES = [
    (1, 1, 16, 128, 64, 64, 8, 64),     # decode, 1 token, topk=64
    (2, 1, 16, 128, 64, 64, 8, 64),     # decode, 2 seqs
    (1, 1, 32, 128, 64, 64, 16, 128),   # decode, 32 heads, topk=128
    (1, 4, 16, 256, 64, 64, 8, 96),     # multi-token (prefill), lora=256, topk=96, w/ -1 pad
    (2, 2, 16, 128, 64, 32, 16, 64),    # block_size=32, 2 seqs x 2 tokens
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Shared-GPU resilience: two other workers contend for the device, so retry the
# kernel launch on transient OOM / contention with exponential backoff.
_OOM_RETRIES = 6
_OOM_BACKOFF_S = 1.5


def load_module():
    spec = importlib.util.spec_from_file_location("sparse_mla_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_transient_gpu_error(exc):
    msg = str(exc).lower()
    return any(
        s in msg
        for s in ("out of memory", "oom", "hip error", "resource", "busy", "contention")
    )


def _retry_gpu(fn):
    """Run fn() with retry/backoff on transient OOM/contention (shared GPU)."""
    import torch

    last = None
    for attempt in range(_OOM_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_transient_gpu_error(e) or attempt == _OOM_RETRIES - 1:
                raise
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(_OOM_BACKOFF_S * (2 ** attempt))
    if last is not None:
        raise last


def make_test_data(num_seqs, tokens_per_seq, num_query_heads, kv_lora_rank,
                   rope_rank, block_size, num_blocks, topk, pad_invalid=False,
                   device="cuda", dtype=None):
    """Create test tensors for unified_attention_sparse_mla."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16

    head_size = kv_lora_rank + rope_rank
    total_tokens = num_seqs * tokens_per_seq
    num_kv_positions = num_blocks * block_size

    # Packed Q: [total_tokens, num_query_heads, head_size]
    q = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)

    # Paged latent KV buffer: [num_blocks, block_size, 1, head_size]
    kv = torch.randn(num_blocks, block_size, 1, head_size, device=device, dtype=dtype)

    # Per-token top-k indices into the flat KV cache [total_tokens, topk].
    # Random unique positions; optionally pad the tail of each row with -1
    # (always keep at least one valid entry per row).
    topk_indices = torch.empty(total_tokens, topk, device=device, dtype=torch.int32)
    for g in range(total_tokens):
        perm = torch.randperm(num_kv_positions, device=device)[:topk].to(torch.int32)
        if pad_invalid:
            n_valid = max(1, topk - (g % (topk // 2 + 1)))
            perm[n_valid:] = -1
        topk_indices[g] = perm

    # block_table: required arg (unused by this simplified kernel)
    max_blocks = num_blocks
    block_table = torch.arange(num_seqs * max_blocks, device=device, dtype=torch.int32)
    block_table = block_table.reshape(num_seqs, max_blocks)

    # cu_seqlens_q: cumulative token counts [num_seqs + 1]
    cu_seqlens_q = torch.arange(0, (num_seqs + 1) * tokens_per_seq, tokens_per_seq,
                                device=device, dtype=torch.int32)

    # seqused_k: required arg (used only for num_seqs here)
    seqused_k = torch.full((num_seqs,), num_kv_positions, device=device, dtype=torch.int32)

    out = torch.empty(total_tokens, num_query_heads, kv_lora_rank, device=device, dtype=dtype)
    scale = 1.0 / (head_size ** 0.5)

    return q, kv, out, cu_seqlens_q, seqused_k, topk_indices, block_table, scale


def _call_kernel(mod, q, kv, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                 max_seqlen_k, scale, topk_indices, block_table, kv_lora_rank):
    return mod.unified_attention_sparse_mla(
        q,
        kv,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        scale,
        topk_indices,
        block_table,
        kv_lora_rank,
    )


def _unpack(shape):
    ns, tps, nqh, lora, rope, bs, nblk, topk = shape
    return ns, tps, nqh, lora, rope, bs, nblk, topk


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "unified_attention_sparse_mla"), "Missing unified_attention_sparse_mla entry"
        assert hasattr(mod, "_kernel_unified_attention_sparse_mla_2d"), \
            "Missing _kernel_unified_attention_sparse_mla_2d"
        return True, None
    except Exception as e:
        return False, str(e)


def ref_sparse_mla(q, kv, topk_indices, block_size, kv_lora_rank, scale):
    """Torch reference for the sparse-MLA kernel (fp32 accumulation).

    Mirrors ``_kernel_unified_attention_sparse_mla_2d`` exactly:
      * Each (token g, head h) is independent (no causal mask; selection is
        purely by the per-token top-k indices).
      * A top-k entry ``pos`` maps to KV position ``pos`` in the flat cache
        (physical_block = pos // block_size, slot = pos % block_size, and
        flat_index = physical_block * block_size + slot == pos).
      * ``pos == -1`` entries are ignored (masked out of the softmax).
      * score S = scale * (Q . K) over the FULL head_size (lora + rope parts,
        exactly what the kernel dots), softmax over valid entries.
      * out = softmax(S) @ V, where V is the lora slice kv[..., :kv_lora_rank].

    This is the sparse-MLA specialization of aiter's test_mla_sparse.py
    ``torch_mla_extend`` / ``ref_masked_attention`` (is_causal=False, per-token
    index selection), adapted to the paged flat-index layout the harness uses.
    """
    import torch

    total_tokens, num_heads, head_size = q.shape
    num_blocks, bsz, _, hs = kv.shape
    assert bsz == block_size and hs == head_size

    qf = q.to(torch.float32)
    kv_flat = kv.reshape(num_blocks * block_size, head_size).to(torch.float32)
    num_positions = kv_flat.shape[0]

    out = torch.zeros(total_tokens, num_heads, kv_lora_rank,
                      device=q.device, dtype=torch.float32)

    for g in range(total_tokens):
        idx = topk_indices[g].to(torch.long)
        valid = (idx != -1) & (idx >= 0) & (idx < num_positions)
        pos = idx[valid]
        if pos.numel() == 0:
            continue
        k = kv_flat[pos]                       # [nvalid, head_size]
        v = k[:, :kv_lora_rank]                # [nvalid, kv_lora_rank]
        qh = qf[g]                             # [num_heads, head_size]
        scores = scale * (qh @ k.t())          # [num_heads, nvalid]
        p = torch.softmax(scores, dim=-1)      # [num_heads, nvalid]
        out[g] = p @ v                         # [num_heads, kv_lora_rank]

    return out


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

    for i, shape in enumerate(TEST_SHAPES):
        ns, tps, nqh, lora, rope, bs, nblk, topk = _unpack(shape)
        pad_invalid = (i == 3)  # shape 4 exercises -1 padding / masking
        try:
            torch.manual_seed(42 + i)
            q, kv, out, cu_seqlens_q, seqused_k, topk_indices, block_table, scale = \
                make_test_data(ns, tps, nqh, lora, rope, bs, nblk, topk,
                               pad_invalid=pad_invalid, device=device, dtype=dtype)
            max_seqlen_q = tps
            max_seqlen_k = nblk * bs

            _retry_gpu(lambda: _call_kernel(
                mod, q, kv, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                max_seqlen_k, scale, topk_indices, block_table, lora,
            ))
            torch.cuda.synchronize()

            # out is written in-place; assert the kernel produced finite output.
            finite = bool(torch.isfinite(out.float()).all().item())

            # Numerical reference matching the EXACT kernel config:
            #  per-token top-k selection, block_size==TILE_SIZE, kv_lora_rank
            #  value slice, full head_size Q.K, scale=1/sqrt(head_size), -1 mask.
            ref = ref_sparse_mla(q, kv, topk_indices, bs, lora, scale)

            norm_err, max_abs, denom = _norm_max_error(ref, out)
            allclose = bool(torch.allclose(
                out.to(torch.float32), ref.to(torch.float32),
                atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL,
            ))
            numeric_ok = norm_err <= NORM_ERR_TOL
            passed = bool(finite and numeric_ok)
            all_pass = all_pass and passed

            details.append({
                "shape_id": i + 1,
                "shape": list(shape),
                "out_shape": list(out.shape),
                "pad_invalid": pad_invalid,
                "finite": finite,
                "norm_max_error": norm_err,
                "max_abs_error": max_abs,
                "ref_absmax": denom,
                "allclose@1e-2": allclose,
                "passed": passed,
            })
            if not passed and first_err is None:
                if not finite:
                    first_err = f"Shape {i+1} {shape}: non-finite output"
                else:
                    first_err = (f"Shape {i+1} {shape}: norm_max_error="
                                 f"{norm_err:.3e} > {NORM_ERR_TOL:.0e}")
        except Exception as e:
            details.append({
                "shape_id": i + 1,
                "shape": list(shape),
                "error": str(e),
            })
            return False, f"Shape {i+1} {shape}: exception: {e}", details

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

    for test_idx, shape in enumerate(TEST_SHAPES):
        ns, tps, nqh, lora, rope, bs, nblk, topk = _unpack(shape)
        params = {
            "num_seqs": ns, "tokens_per_seq": tps, "num_query_heads": nqh,
            "kv_lora_rank": lora, "rope_rank": rope, "block_size": bs,
            "num_blocks": nblk, "topk": topk,
        }
        try:
            torch.manual_seed(42 + test_idx)
            q, kv, out, cu_seqlens_q, seqused_k, topk_indices, block_table, scale = \
                make_test_data(ns, tps, nqh, lora, rope, bs, nblk, topk,
                               pad_invalid=(test_idx == 3), device=device, dtype=dtype)
            max_seqlen_q = tps
            max_seqlen_k = nblk * bs

            for _ in range(WARMUP_ITERATIONS):
                _retry_gpu(lambda: _call_kernel(
                    mod, q, kv, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                    max_seqlen_k, scale, topk_indices, block_table, lora,
                ))
            torch.cuda.synchronize()

            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                lambda: _call_kernel(mod, q, kv, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, scale, topk_indices, block_table, lora),
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
                print(f"  shape {d['shape_id']} {d['shape']} -> out {d['out_shape']}: "
                      f"finite={d['finite']} norm_max_err={d['norm_max_error']:.3e} "
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
