#!/usr/bin/env python3
"""Real launcher + benchmark for the raw @triton.jit kernel
``_gemm_a8w8_blockscale_kernel`` (a8w8 block-scale GEMM, aiter).

Workload regime: seqlen = 1024, concurrency B in {2, 32, 64}.
This is a token-parallel GEMM (per-token activation rows). We map:
    M (num_tokens / rows of A) = B * 1024     (prefill token count)
Model dims held from the captured base case:
    K = 6144  (hidden), N = 2624 (intermediate-ish out feature)
    GROUP_K = GROUP_N = BLOCK_SIZE_K = 128, group_size = 128
Scales:
    a_scale : (M, ceil(K/128) = 48)   fp32
    b_scale : (ceil(K/128) = 48, ceil(N/128) = 21)  fp32 (col-major)
A : (M, K) fp8_e4m3fnuz row-major
B : (K, N) fp8_e4m3fnuz col-major  (stride [1, K])
C : (M, N) bf16 row-major   (output, written in-place)

The kernel references module globals ``remap_xcd`` and ``pid_grid``; keep local
faithful copies so this task does not import aiter.
"""
import os, sys, importlib.util, math
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
KERNEL_NAME = "_gemm_a8w8_blockscale_kernel"

# Local copies of aiter.ops.triton.utils._triton.pid_preprocessing helpers.
@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):  # noqa: F821
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@triton.jit
def pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 1):  # noqa: F821
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)  # noqa: F821
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

# Fixed model dims from captured base case
K_DIM = 6144
N_DIM = 2624
GROUP = 128            # GROUP_K == GROUP_N == BLOCK_SIZE_K
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
GROUP_SIZE_M = 1
NUM_KSPLIT = 1
NUM_WARPS = 4
NUM_STAGES = 2
CACHE_MOD = ".cg"

SEQLEN = 1024
CONCURRENCY = [2, 32, 64]


def _load_kernel(source_file):
    spec = importlib.util.spec_from_file_location("kmod", source_file)
    mod = importlib.util.module_from_spec(spec)
    # Inject the helper jit functions the kernel references as globals.
    mod.pid_grid = pid_grid
    mod.remap_xcd = remap_xcd
    spec.loader.exec_module(mod)
    # exec_module rebinds the module dict; re-inject after exec to be safe.
    mod.pid_grid = pid_grid
    mod.remap_xcd = remap_xcd
    kern = getattr(mod, KERNEL_NAME)
    # the JIT fn captures globals from its own __globals__; patch those too.
    kern.fn.__globals__.setdefault("pid_grid", pid_grid)
    kern.fn.__globals__.setdefault("remap_xcd", remap_xcd)
    kern.fn.__globals__["pid_grid"] = pid_grid
    kern.fn.__globals__["remap_xcd"] = remap_xcd
    return kern


def _build_inputs(B, seed=42):
    """Deterministic seeded inputs for concurrency B. Returns (args, meta, c)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    M = B * SEQLEN
    K = K_DIM
    N = N_DIM
    scale_k = (K + GROUP - 1) // GROUP   # 48
    scale_n = (N + GROUP - 1) // GROUP   # 21
    fp8 = getattr(torch, "float8_e4m3fnuz")

    a_f = (torch.randn(M, K, generator=g, device="cuda", dtype=torch.float32) * 0.2)
    b_f = (torch.randn(K, N, generator=g, device="cuda", dtype=torch.float32) * 0.2)
    a = a_f.to(fp8)
    # B is col-major (stride [1, K]) as in capture
    b = b_f.to(fp8).t().contiguous().t()
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    a_scale = (torch.rand(M, scale_k, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.005)
    # b_scale col-major (stride [1, scale_k]) as in capture
    b_scale = (torch.rand(scale_k, scale_n, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.005)
    b_scale = b_scale.t().contiguous().t()

    args = (
        a, b, c, a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        0,                       # stride_ck (NUM_KSPLIT==1)
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
    )
    meta = dict(
        GROUP_K=GROUP, GROUP_N=GROUP,
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_KSPLIT=NUM_KSPLIT,
        SPLITK_BLOCK_SIZE=(K + NUM_KSPLIT - 1) // NUM_KSPLIT,
        cache_modifier=CACHE_MOD,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return args, meta, c, M, N


def _grid(M, N):
    return (NUM_KSPLIT * ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N),)


def _launch(kern, B, seed=42):
    args, meta, c, M, N = _build_inputs(B, seed=seed)
    grid = _grid(M, N)
    c.zero_()
    kern[grid](*args, **meta)
    torch.cuda.synchronize()
    return c


def cases():
    return [(f"c{b}", b) for b in CONCURRENCY]


def run_correctness():
    src_edit = os.path.join(TASK_DIR, "source", "triton__gemm_a8w8_blockscale_kernel.py")
    src_gold = os.path.join(TASK_DIR, "source_golden", "triton__gemm_a8w8_blockscale_kernel.py")
    kern_e = _load_kernel(src_edit)
    kern_g = _load_kernel(src_gold)
    for cid, B in cases():
        out_e = _launch(kern_e, B, seed=42).float()
        out_g = _launch(kern_g, B, seed=42).float()
        cos = torch.nn.functional.cosine_similarity(
            out_e.flatten().unsqueeze(0), out_g.flatten().unsqueeze(0)
        ).item()
        denom = out_g.abs().max().clamp_min(1e-6)
        max_rel = (out_e - out_g).abs().max().item() / denom.item()
        if not (cos >= 0.99 and max_rel <= 1e-2):
            return False, f"{cid}: cos={cos:.5f} max_rel={max_rel:.5f}"
    return True, None


def run_performance(warmup=10, iters=100):
    src_edit = os.path.join(TASK_DIR, "source", "triton__gemm_a8w8_blockscale_kernel.py")
    kern = _load_kernel(src_edit)
    results = []
    for cid, B in cases():
        args, meta, c, M, N = _build_inputs(B, seed=42)
        grid = _grid(M, N)
        for _ in range(warmup):
            kern[grid](*args, **meta)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for j in range(iters):
            starts[j].record()
            kern[grid](*args, **meta)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters
        results.append({
            "test_case_id": cid,
            "execution_time_ms": avg,
            "params": {"B": B, "M": M, "N": N, "K": K_DIM, "seqlen": SEQLEN},
        })
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok, err = run_correctness()
        print("Correctness:", "PASS" if ok else "FAIL", err or "")
    else:
        for r in run_performance():
            print(f"Performance: {r['execution_time_ms']:.4f} ms ({r['test_case_id']})")
