#!/usr/bin/env python3
"""Real launcher + benchmark harness for the raw @triton.jit kernel
`_fwd_grouped_kernel_stage1` (vLLM v1 triton_decode_attention, MLA-style
decode attention, stage 1 split-KV reduction).

This kernel is COMPILE-ONLY in the captured task (raw @triton.jit, no
launcher). We rebuild a launcher here:

  * regenerate the 3 regime test cases (concurrency B in {2,32,64}),
  * build seeded, deterministic, *semantically valid* inputs (paged KV,
    Req_to_tokens pointing at real pages, B_Seqlen = context length),
  * recompute the grid FORMULA from the regime dims (do NOT reuse the
    captured constant grid),
  * launch the JIT kernel into the in-place Att_Out tensor,
  * run golden-vs-editable correctness on identical seeded inputs,
  * time each case with CUDA events (10 warmup + 100 timed).

WORKLOAD REGIME: seqlen = 1024 (decode-style: q_len=1, kv/context_len=1024),
concurrency B in {2,32,64} -> batch = B.

MODEL DIMS kept from the captured base case (sig_752e5deb6cdc):
  q_head_num=8, kv_group_num=8 (=> 1 kv head), head_dim Lk=576
  (= 512 nope + 64 rope/DPE), Lv=512, PAGE_SIZE=16, NUM_KV_SPLITS=4,
  BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK_DV=512, BLOCK_N=16, BLOCK_H=16,
  logit_cap=0.0, sm_scale from captured (0.144679...).
"""
import os
import importlib.util

import torch
import triton

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_EDIT = os.path.join(TASK_DIR, "source", "triton__fwd_grouped_kernel_stage1.py")
SRC_GOLD = os.path.join(TASK_DIR, "source", "source_golden",
                        "triton__fwd_grouped_kernel_stage1.py")
KERNEL_NAME = "_fwd_grouped_kernel_stage1"

# ---- model dims (from captured base case) -------------------------------
Q_HEAD_NUM = 8
KV_GROUP_NUM = 8           # q heads per kv head  => num_kv_heads = 1
NUM_KV_HEADS = Q_HEAD_NUM // KV_GROUP_NUM
LK = 576                   # full K head dim (nope+rope)
LV = 512                   # V head dim
BLOCK_DMODEL = 512
BLOCK_DPE = 64             # LK - BLOCK_DMODEL
BLOCK_DV = 512
BLOCK_N = 16
BLOCK_H = 16
NUM_KV_SPLITS = 4
PAGE_SIZE = 16
LOGIT_CAP = 0.0
SM_SCALE = 0.14467962580268923
NUM_WARPS = 4
NUM_STAGES = 1

SEQLEN = 1024              # input seqlen == output seqlen == context len
CONCURRENCY = {"c2": 2, "c32": 32, "c64": 64}


def _load(path):
    spec = importlib.util.spec_from_file_location("mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, KERNEL_NAME)


def build_inputs(batch, seqlen, seed=42, device="cuda"):
    """Build seeded, semantically valid paged-decode inputs for one case.

    Returns (args, kwargs, out) where `out` is Att_Out (mutated in place)."""
    g = torch.Generator(device=device).manual_seed(seed)

    # Q: [batch, q_head_num, Lk]  (decode: q_len == 1, folded into batch dim)
    Q = torch.randn(batch, Q_HEAD_NUM, LK, dtype=torch.bfloat16, device=device, generator=g)

    # Paged KV buffer. Each request has `seqlen` context tokens spread over
    # ceil(seqlen/PAGE_SIZE) pages; pages are unique across requests so
    # kv_loc indexing is collision-free.
    pages_per_req = triton.cdiv(seqlen, PAGE_SIZE)
    total_pages = batch * pages_per_req
    total_tokens = total_pages * PAGE_SIZE
    K_Buffer = torch.randn(total_tokens, NUM_KV_HEADS, LK,
                           dtype=torch.bfloat16, device=device, generator=g)
    V_Buffer = torch.randn(total_tokens, NUM_KV_HEADS, LV,
                           dtype=torch.bfloat16, device=device, generator=g)

    # Req_to_tokens: [batch, max_pages] int32 -> page numbers.
    max_pages = pages_per_req
    Req_to_tokens = torch.empty(batch, max_pages, dtype=torch.int32, device=device)
    for b in range(batch):
        Req_to_tokens[b] = torch.arange(b * pages_per_req, (b + 1) * pages_per_req,
                                        dtype=torch.int32, device=device)

    # B_Seqlen: [batch] int32 -> context length per request.
    B_Seqlen = torch.full((batch,), seqlen, dtype=torch.int32, device=device)

    # Att_Out: [batch, q_head_num, NUM_KV_SPLITS, Lv+1] float32.
    Att_Out = torch.zeros(batch, Q_HEAD_NUM, NUM_KV_SPLITS, LV + 1,
                          dtype=torch.float32, device=device)

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # strides
    stride_req_to_tokens_b = Req_to_tokens.stride(0)
    stride_qbs = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_buf_kbs = K_Buffer.stride(0)
    stride_buf_kh = K_Buffer.stride(1)
    stride_buf_vbs = V_Buffer.stride(0)
    stride_buf_vh = V_Buffer.stride(1)
    stride_mid_ob = Att_Out.stride(0)
    stride_mid_oh = Att_Out.stride(1)
    stride_mid_os = Att_Out.stride(2)

    args = [
        Q, K_Buffer, V_Buffer, SM_SCALE, Req_to_tokens, B_Seqlen, Att_Out,
        stride_req_to_tokens_b,
        stride_qbs, stride_qh,
        stride_buf_kbs, stride_buf_kh,
        stride_buf_vbs, stride_buf_vh,
        stride_mid_ob, stride_mid_oh, stride_mid_os,
        k_scale, v_scale,
    ]
    kwargs = dict(
        kv_group_num=KV_GROUP_NUM,
        q_head_num=Q_HEAD_NUM,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=PAGE_SIZE,
        logit_cap=LOGIT_CAP,
        Lk=LK,
        Lv=LV,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return args, kwargs, Att_Out


def grid_for(batch):
    """Grid FORMULA recomputed from regime dims (NOT the captured constant)."""
    head_blocks = triton.cdiv(Q_HEAD_NUM, min(BLOCK_H, KV_GROUP_NUM))
    return (batch, head_blocks, NUM_KV_SPLITS)


def launch(kern, batch, seqlen, seed=42):
    args, kwargs, out = build_inputs(batch, seqlen, seed=seed)
    grid = grid_for(batch)
    kern[grid](*args, **kwargs)
    return out


def run_correctness():
    """Golden-from-original: run the editable source and the frozen golden
    copy on identical seeded inputs, compare Att_Out."""
    kern_edit = _load(SRC_EDIT)
    kern_gold = _load(SRC_GOLD)
    worst = None
    for cid, B in CONCURRENCY.items():
        out_edit = launch(kern_edit, B, SEQLEN, seed=42)
        out_gold = launch(kern_gold, B, SEQLEN, seed=42)
        torch.cuda.synchronize()
        a = out_edit.detach().to(torch.float32)
        b = out_gold.detach().to(torch.float32)
        # mask non-finite (e.g. -inf logsumexp tail at padded splits)
        finite = torch.isfinite(a) & torch.isfinite(b)
        if not finite.any():
            return False, f"{cid}: both outputs all non-finite"
        af, bf = a[finite], b[finite]
        cos = torch.nn.functional.cosine_similarity(
            af.reshape(1, -1), bf.reshape(1, -1)).item()
        denom = bf.abs().clamp_min(1e-6)
        max_rel = ((af - bf).abs() / denom).max().item()
        if not (cos >= 0.99 and max_rel < 1e-2):
            return False, (f"{cid}: cos={cos:.6f} max_rel={max_rel:.4g} "
                           f"(need cos>=0.99 & max_rel<1e-2)")
        worst = f"{cid}: cos={cos:.6f} max_rel={max_rel:.4g}"
    return True, None


def run_performance():
    kern = _load(SRC_EDIT)
    results = []
    for cid, B in CONCURRENCY.items():
        args, kwargs, out = build_inputs(B, SEQLEN, seed=42)
        grid = grid_for(B)
        for _ in range(10):
            kern[grid](*args, **kwargs)
        torch.cuda.synchronize()
        n_iter = 100
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for j in range(n_iter):
            starts[j].record()
            kern[grid](*args, **kwargs)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
        results.append({
            "test_case_id": cid,
            "execution_time_ms": avg,
            "params": {"batch": B, "seqlen": SEQLEN, "q_len": 1,
                       "q_head_num": Q_HEAD_NUM, "kv_group_num": KV_GROUP_NUM,
                       "Lk": LK, "Lv": LV, "grid": list(grid_for(B))},
        })
    return results


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok, err = run_correctness()
        print("Correctness:", "PASS" if ok else "FAIL", "" if ok else err)
    else:
        for r in run_performance():
            print(f"Performance: {r['execution_time_ms']:.4f} ms ({r['test_case_id']})")
