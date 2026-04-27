#!/usr/bin/env python3
"""Standalone MLA decode + reduce benchmark for AgentKernelArena.

Tests mla/reduce.cu by calling aiter.mla.mla_decode_fwd(), which internally
invokes the reduce kernel when num_kv_splits > 1 (for larger ctx_len).

Self-contained: does not depend on aiter's op_tests/test_mla.py.
Only depends on the installed aiter package (import aiter).

Usage:
  python3 test_mla_bench.py --mode correctness
  python3 test_mla_bench.py --mode performance
  python3 test_mla_bench.py --mode correctness --batch 1 --ctx 64 --nhead 16
"""
from __future__ import annotations

import argparse
import sys
import torch

# MLA absorb-mode constants (DeepSeek V2/V3)
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                       # 512
NHEAD_KV = 1
PAGE_SIZE = 1
DECODE_QLEN = 1
DTYPE = torch.bfloat16


def _create_inputs(batch_size: int, ctx_len: int, nhead: int, device: str = "cuda"):
    """Create all input tensors for one MLA decode test case."""
    total_q = batch_size * DECODE_QLEN
    total_kv = batch_size * ctx_len
    num_page = total_kv + 10000

    q = torch.randn(total_q, nhead, QK_HEAD_DIM, dtype=DTYPE, device=device)
    kv_flat = torch.randn(num_page, NHEAD_KV, QK_HEAD_DIM, dtype=DTYPE, device=device)
    kv_buffer = kv_flat.view(num_page, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM)

    seq_lens_q = torch.full((batch_size,), DECODE_QLEN, dtype=torch.int32, device=device)
    seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int32, device=device)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(seq_lens_q, dim=0)
    kv_indptr[1:] = torch.cumsum(seq_lens_kv, dim=0)

    kv_indices = torch.arange(num_page, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    out = torch.empty(total_q, nhead, V_HEAD_DIM, dtype=DTYPE, device=device).fill_(-1)
    sm_scale = 1.0 / (QK_HEAD_DIM ** 0.5)

    return {
        "q": q, "kv_buffer": kv_buffer, "kv_flat": kv_flat, "out": out,
        "qo_indptr": qo_indptr, "kv_indptr": kv_indptr,
        "kv_indices": kv_indices, "kv_last_page_lens": kv_last_page_lens,
        "max_seqlen_q": DECODE_QLEN, "sm_scale": sm_scale,
        "batch_size": batch_size, "ctx_len": ctx_len, "nhead": nhead,
    }


def _run_kernel(d: dict):
    """Call aiter MLA decode (triggers reduce kernel internally)."""
    import aiter
    return aiter.mla.mla_decode_fwd(
        d["q"], d["kv_buffer"], d["out"],
        d["qo_indptr"], d["kv_indptr"], d["kv_indices"], d["kv_last_page_lens"],
        d["max_seqlen_q"],
        page_size=PAGE_SIZE,
        nhead_kv=NHEAD_KV,
        sm_scale=d["sm_scale"],
    )


def _reference_attention(d: dict) -> torch.Tensor:
    """Compute reference MLA attention output using PyTorch.

    In absorb mode: K uses full 576-dim, V uses first 512-dim (kv_lora_rank).
    nhead_kv=1 is broadcast to nhead via matmul broadcasting.
    """
    q = d["q"]
    kv_flat = d["kv_flat"]
    kv_indptr = d["kv_indptr"]
    kv_indices = d["kv_indices"]
    sm_scale = d["sm_scale"]
    batch_size = d["batch_size"]

    outputs = []
    for b in range(batch_size):
        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()
        q_start = b * DECODE_QLEN

        pages = kv_indices[kv_start:kv_end]
        kv = kv_flat[pages]              # [ctx_len, 1, 576]
        k = kv[:, 0, :]                  # [ctx_len, 576]
        v = kv[:, 0, :V_HEAD_DIM]        # [ctx_len, 512]
        q_b = q[q_start:q_start + DECODE_QLEN]  # [1, nhead, 576]

        # Attention with broadcasting (nhead_kv=1 → nhead)
        q_t = q_b.float().permute(1, 0, 2)       # [nhead, 1, 576]
        k_t = k.float().T.unsqueeze(0)            # [1, 576, ctx_len]
        scores = torch.matmul(q_t, k_t) * sm_scale  # [nhead, 1, ctx_len]
        weights = torch.softmax(scores, dim=-1)

        v_t = v.float().unsqueeze(0)              # [1, ctx_len, 512]
        out_b = torch.matmul(weights, v_t)        # [nhead, 1, 512]
        outputs.append(out_b.permute(1, 0, 2).to(DTYPE))

    return torch.cat(outputs, dim=0)


def run_correctness(batches, ctxs, nheads):
    """Run correctness tests and print results."""
    passed = failed = 0
    for nhead in nheads:
        for batch in batches:
            for ctx in ctxs:
                tag = f"batch={batch} ctx={ctx} nhead={nhead}"
                try:
                    d = _create_inputs(batch, ctx, nhead)
                    _run_kernel(d)
                    ref = _reference_attention(d)
                    max_err = (d["out"].float() - ref.float()).abs().max().item()
                    ok = torch.allclose(d["out"].float(), ref.float(), atol=0.01, rtol=0.01)
                    if ok:
                        print(f"[RESULT] {tag} status=PASS err={max_err:.6f}")
                        passed += 1
                    else:
                        print(f"[RESULT] {tag} status=FAIL err={max_err:.6f}")
                        failed += 1
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[RESULT] {tag} status=SKIP (OOM)")
                    else:
                        print(f"[RESULT] {tag} status=FAIL error={e}")
                        failed += 1
                finally:
                    torch.cuda.empty_cache()

    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    return failed == 0


def run_performance(batches, ctxs, nheads, num_warmup=5, num_iters=20):
    """Run performance benchmarks and print results."""
    results = []
    for nhead in nheads:
        for batch in batches:
            for ctx in ctxs:
                tag = f"batch={batch} ctx={ctx} nhead={nhead}"
                try:
                    d = _create_inputs(batch, ctx, nhead)
                    for _ in range(num_warmup):
                        d["out"].fill_(-1)
                        _run_kernel(d)
                    torch.cuda.synchronize()

                    times_us = []
                    for _ in range(num_iters):
                        d["out"].fill_(-1)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        _run_kernel(d)
                        end.record()
                        torch.cuda.synchronize()
                        times_us.append(start.elapsed_time(end) * 1000.0)

                    times_us.sort()
                    median_us = times_us[len(times_us) // 2]
                    print(f"[RESULT] {tag} us={median_us:.2f}")
                    results.append({"batch": batch, "ctx": ctx, "nhead": nhead, "us": median_us})
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[RESULT] {tag} us=nan (OOM)")
                    else:
                        print(f"[RESULT] {tag} us=nan error={e}")
                finally:
                    torch.cuda.empty_cache()

    if results:
        avg = sum(r["us"] for r in results) / len(results)
        print(f"\nSUMMARY: {len(results)} cases, avg {avg:.2f} us")
    return results


def main():
    parser = argparse.ArgumentParser(description="MLA decode + reduce benchmark")
    parser.add_argument("--mode", choices=["correctness", "performance"], required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 128])
    parser.add_argument("--ctx", type=int, nargs="+", default=[64, 256, 1200, 3200])
    parser.add_argument("--nhead", type=int, nargs="+", default=[16, 128])
    args = parser.parse_args()

    if args.mode == "correctness":
        ok = run_correctness(args.batch, args.ctx, args.nhead)
        sys.exit(0 if ok else 1)
    else:
        results = run_performance(args.batch, args.ctx, args.nhead)
        sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
