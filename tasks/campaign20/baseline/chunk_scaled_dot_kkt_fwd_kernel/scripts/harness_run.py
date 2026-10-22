#!/usr/bin/env python3
"""Real launcher + benchmark for chunk_scaled_dot_kkt_fwd_kernel (raw @triton.jit).

Workload regime (arena standard): seqlen = 1024, concurrency B in {2,32,64}.
This is an attention-style FLA kernel (chunked scaled dot K @ K^T for gated
DeltaNet). Mapping:
  - PREFILL style: each of B sequences has length T = 1024 (q_len == kv_len).
  - batch  = B  (concurrency)         -> via cu_seqlens prefix sum
  - per-seq seqlen T = 1024
  - chunks per seq NT_per = cdiv(T, BT); total grid dim0 = B * NT_per
  - grid = (NT_total, B * H)
Model dims kept from the captured base case:
  H = 8, Hg = 4 (GQA), K = 128 (head_dim), BT = 64 (chunk size),
  USE_G = True, IS_VARLEN = True   (exactly as captured).

NOTE: the captured kernel source references the bare name ``exp`` inside the
USE_G branch but only imports ``triton``/``triton.language``. That is a
pre-existing bug in the captured source. We do NOT edit the kernel file; instead
we inject ``exp = tl.exp`` into the *module namespace* after import. This is
applied identically to the golden and the editable module, so edited-vs-golden
correctness stays fair.
"""
import os, json, importlib.util
import torch
import triton.language as tl

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_EDIT = os.path.join(TASK_DIR, "source", "triton_chunk_scaled_dot_kkt_fwd_kernel.py")
SRC_GOLD = os.path.join(TASK_DIR, "source", "source_golden", "triton_chunk_scaled_dot_kkt_fwd_kernel.py")
KERNEL_NAME = "chunk_scaled_dot_kkt_fwd_kernel"

# ----- fixed model dims (from captured base case) -----
H, Hg, K, BT = 8, 4, 128, 64
SEQLEN = 1024
CONCURRENCY = {"c2": 2, "c32": 32, "c64": 64}


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"kkt_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # inject missing global referenced by the captured source (USE_G branch).
    if not hasattr(mod, "exp"):
        mod.exp = tl.exp
    return getattr(mod, KERNEL_NAME)


def build_inputs(B, seed=42, dev="cuda"):
    """Deterministic, seeded inputs for concurrency B at seqlen=1024 (prefill)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    T = SEQLEN
    tot = B * T
    NT_per = (T + BT - 1) // BT
    k = (torch.randn(1, tot, Hg, K, generator=g, device=dev, dtype=torch.float32)
         ).to(torch.bfloat16)
    beta = (torch.randn(1, tot, H, generator=g, device=dev, dtype=torch.float32)
            ).to(torch.bfloat16)
    gg = torch.randn(1, tot, H, generator=g, device=dev, dtype=torch.float32) * 0.1
    A = torch.zeros(1, tot, H, BT, device=dev, dtype=torch.float32)
    cu_seqlens = torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32, device=dev)
    chunk_indices = torch.tensor(
        [[b, c] for b in range(B) for c in range(NT_per)], dtype=torch.int32, device=dev)
    NT_total = chunk_indices.shape[0]
    grid = (NT_total, B * H)
    args = (k, beta, gg, A, cu_seqlens, chunk_indices, T)
    meta = dict(H=H, Hg=Hg, K=K, BT=BT)
    return args, meta, A, grid


def launch(kern, B, seed=42):
    args, meta, A, grid = build_inputs(B, seed=seed)
    kern[grid](*args, **meta)
    return A


def run_correctness():
    kern_e = _load(SRC_EDIT, "edit")
    kern_g = _load(SRC_GOLD, "gold")
    worst = None
    for cid, B in CONCURRENCY.items():
        out_e = launch(kern_e, B, seed=42); torch.cuda.synchronize()
        out_g = launch(kern_g, B, seed=42); torch.cuda.synchronize()
        a = out_e.float().flatten()
        b = out_g.float().flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        denom = b.abs().max().clamp_min(1e-6)
        max_rel = ((a - b).abs().max() / denom).item()
        if cos < 0.99 or max_rel > 1e-2:
            return {"status": "fail",
                    "error": f"{cid}: cos={cos:.5f} max_rel={max_rel:.4e}"}
        worst = f"{cid}: cos={cos:.5f} max_rel={max_rel:.4e}"
    return {"status": "ok", "error": None, "detail": worst}


def run_performance(n_warmup=10, n_iter=100):
    kern = _load(SRC_EDIT, "perf")
    cases = []
    for cid, B in CONCURRENCY.items():
        args, meta, A, grid = build_inputs(B, seed=42)
        for _ in range(n_warmup):
            kern[grid](*args, **meta)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for j in range(n_iter):
            starts[j].record(); kern[grid](*args, **meta); ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
        cases.append({"test_case_id": cid, "execution_time_ms": avg,
                      "params": {"B": B, "seqlen": SEQLEN, "H": H, "Hg": Hg,
                                 "K": K, "BT": BT, "USE_G": True, "IS_VARLEN": True}})
    return cases


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        print(json.dumps(run_correctness()))
    else:
        print(json.dumps(run_performance()))
