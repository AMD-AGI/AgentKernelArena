#!/usr/bin/env python3
"""Real launcher + benchmark for the compile-only triton kernel `_topk_forward`
(multi-peer variant captured from triton_kernels.topk_details).

The captured source references the @triton.jit helper `streaming_topk` (and its
own helper chain) as module globals but does NOT define them in the captured
file. We inject those helpers (imported from the installed triton_kernels
package, which is the upstream origin) into BOTH the editable source module and
the frozen golden module BEFORE launching, so the JIT can resolve them at
compile time. We never edit the kernel source file on disk.

Workload regime (token-parallel router op):
  n_rows (M) = B * 1024 for B in {2, 32, 64}  -> ids c2, c32, c64
Model dims kept from the captured base case:
  n_expts_tot = 128, N_EXPTS_ACT (topk k) = 4, N_EXPTS_PAD = 128,
  BLOCK_M = 32, BLOCK_N = 32, APPLY_SOFTMAX = True, USE_PROVIDED_INDX = False.
grid = (cdiv(n_rows, BLOCK_M),)  -- recomputed per regime (not the captured const).
"""
import os, sys, json, importlib.util
import torch
import triton

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton__topk_forward.py")
GOLDEN_FILE = os.path.join(TASK_DIR, "source_golden", "triton__topk_forward.py")
BUILD_DIR = os.path.join(TASK_DIR, "build")
KERNEL_NAME = "_topk_forward"

# Fixed model dims (from captured base case)
N_EXPTS_TOT = 128
N_EXPTS_ACT = 4
N_EXPTS_PAD = 128
BLOCK_M = 32
BLOCK_N = 32
APPLY_SOFTMAX = True
USE_PROVIDED_INDX = False
DST_OFFS_M = 0

# Regime: B in {2,32,64}; M = B*1024
REGIME = [("c2", 2), ("c32", 32), ("c64", 64)]


def _cdiv(a, b):
    return (a + b - 1) // b


def _load_kernel(path, modname):
    """Load a source file as a module and inject the streaming_topk helper chain
    into its globals so the @triton.jit kernel can resolve them."""
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Inject helper JIT functions (upstream origin of the captured kernel).
    from triton_kernels.topk_details._topk_forward import (
        streaming_topk, fpval_to_key, key_to_fpval, indx_to_key, key_to_indx,
        get_topmask_and_fullmask,
    )
    for name, obj in [
        ("streaming_topk", streaming_topk),
        ("fpval_to_key", fpval_to_key),
        ("key_to_fpval", key_to_fpval),
        ("indx_to_key", indx_to_key),
        ("key_to_indx", key_to_indx),
        ("get_topmask_and_fullmask", get_topmask_and_fullmask),
    ]:
        setattr(mod, name, obj)
    kern = getattr(mod, KERNEL_NAME)
    # The @triton.jit kernel resolves globals from its own __globals__ (the
    # module dict). Ensure the helpers are present there too.
    g = kern.fn.__globals__ if hasattr(kern, "fn") else kern.__globals__
    g.setdefault("streaming_topk", streaming_topk)
    g.setdefault("fpval_to_key", fpval_to_key)
    g.setdefault("key_to_fpval", key_to_fpval)
    g.setdefault("indx_to_key", indx_to_key)
    g.setdefault("key_to_indx", key_to_indx)
    g.setdefault("get_topmask_and_fullmask", get_topmask_and_fullmask)
    return kern


def _build_inputs(n_rows, seed=42):
    """Deterministic seeded inputs for one regime case."""
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn((n_rows, N_EXPTS_TOT), dtype=torch.bfloat16, device=dev, generator=g)
    # outputs (mutated in place) -- one peer
    Yv = torch.zeros((n_rows, N_EXPTS_ACT), dtype=torch.bfloat16, device=dev)
    Yi = torch.zeros((n_rows, N_EXPTS_ACT), dtype=torch.int16, device=dev)
    n_cols_words = _cdiv(N_EXPTS_TOT, 32)  # = 4
    # bitmatrix stored [words, n_rows] then conceptually transposed; kernel uses
    # stride_rm=1 (row stride over rows) and stride_rn=n_rows (word stride).
    Bits = torch.zeros((n_cols_words, n_rows), dtype=torch.uint32, device=dev)
    PeerYvs = (Yv,)
    PeerYis = (Yi,)
    PeerBits = (Bits,)
    stride_xm = N_EXPTS_TOT
    stride_ym = N_EXPTS_ACT
    stride_rm = 1
    stride_rn = n_rows
    args = [
        X, stride_xm,
        PeerYvs, PeerYis, stride_ym,
        USE_PROVIDED_INDX, PeerBits, stride_rm, stride_rn,
        n_rows, N_EXPTS_TOT,
        DST_OFFS_M,
    ]
    kwargs = dict(
        APPLY_SOFTMAX=APPLY_SOFTMAX,
        BLOCK_M=BLOCK_M, N_EXPTS_PAD=N_EXPTS_PAD,
        N_EXPTS_ACT=N_EXPTS_ACT, BLOCK_N=BLOCK_N,
    )
    return args, kwargs, (Yv, Yi, Bits)


def _launch(kern, n_rows, seed=42):
    args, kwargs, outs = _build_inputs(n_rows, seed=seed)
    grid = (max(_cdiv(n_rows, BLOCK_M), 1),)
    kern[grid](*args, **kwargs)
    torch.cuda.synchronize()
    return outs


def _compare(a_outs, b_outs):
    """Compare (Yv float, Yi index, Bits uint) edited-vs-golden.
    Values: cosine>=0.99 & small max-rel; indices/bits: exact."""
    Yv_a, Yi_a, Bits_a = a_outs
    Yv_b, Yi_b, Bits_b = b_outs
    # exact for integer index + bitmatrix
    if not torch.equal(Yi_a, Yi_b):
        nmis = (Yi_a != Yi_b).sum().item()
        return f"index mismatch: {nmis} differing entries"
    if not torch.equal(Bits_a, Bits_b):
        nmis = (Bits_a != Bits_b).sum().item()
        return f"bitmatrix mismatch: {nmis} differing words"
    fa = Yv_a.to(torch.float32).flatten()
    fb = Yv_b.to(torch.float32).flatten()
    cos = torch.nn.functional.cosine_similarity(fa, fb, dim=0).item()
    denom = fb.abs().clamp_min(1e-6)
    max_rel = ((fa - fb).abs() / denom).max().item()
    if cos < 0.99:
        return f"values cosine {cos:.5f} < 0.99"
    if max_rel > 1e-2:
        return f"values max_rel {max_rel:.4e} > 1e-2"
    return None


def run_correctness():
    edit = _load_kernel(SOURCE_FILE, "topk_edit")
    gold = _load_kernel(GOLDEN_FILE, "topk_gold")
    for cid, B in REGIME:
        n_rows = B * 1024
        out_e = _launch(edit, n_rows, seed=42)
        out_g = _launch(gold, n_rows, seed=42)
        err = _compare(out_e, out_g)
        if err:
            return False, f"{cid} (n_rows={n_rows}): {err}"
    return True, None


def run_performance(warmup=10, iters=100):
    edit = _load_kernel(SOURCE_FILE, "topk_perf")
    results = []
    for cid, B in REGIME:
        n_rows = B * 1024
        args, kwargs, _ = _build_inputs(n_rows, seed=42)
        grid = (max(_cdiv(n_rows, BLOCK_M), 1),)
        for _ in range(warmup):
            edit[grid](*args, **kwargs)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for j in range(iters):
            starts[j].record()
            edit[grid](*args, **kwargs)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters
        results.append({
            "test_case_id": cid,
            "execution_time_ms": avg,
            "params": {"B": B, "n_rows": n_rows, "n_expts_tot": N_EXPTS_TOT,
                       "topk": N_EXPTS_ACT, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
        })
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    os.makedirs(BUILD_DIR, exist_ok=True)
    if mode == "correctness":
        ok, err = run_correctness()
        print("Correctness:", "PASS" if ok else "FAIL", err or "")
    else:
        for r in run_performance():
            print(f"Performance: {r['execution_time_ms']:.4f} ms ({r['test_case_id']})")
