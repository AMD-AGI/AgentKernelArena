#!/usr/bin/env python3
"""Real launcher + benchmark harness for _gemm_a16_w16_kernel (triton2triton).

Compile-only kernel: raw @triton.jit with no launcher. This module builds a
launcher from the captured arg schema, injects the helper symbols the kernel
references from its module globals (remap_xcd, pid_grid), regenerates the 3
workload-regime test cases, runs golden-vs-editable correctness, and times
each case with CUDA events.

Workload regime (token-parallel GEMM):
  - num_tokens M = B*1024 for B in {2,32,64}
  - model dims kept from captured base case: K=2880, N=5120
  - prefill token count semantics (seqlen=1024)
"""
import os
import json
import importlib.util

import torch
import triton
import triton.language as tl

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton__gemm_a16_w16_kernel.py")
GOLDEN_FILE = os.path.join(TASK_DIR, "source_golden", "triton__gemm_a16_w16_kernel.py")
KERNEL_NAME = "_gemm_a16_w16_kernel"
TEST_CASES = os.path.join(TASK_DIR, "test_cases.json")
BUILD_DIR = os.path.join(TASK_DIR, "build")

# ---- model dims from captured base case ----
K_DIM = 2880
N_DIM = 5120
SEQLEN = 1024
CONCURRENCY = [2, 32, 64]

# ---- meta-params from the captured base case kwargs_sig ----
META = dict(
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=128,
    GROUP_SIZE_M=1,
    NUM_KSPLIT=1,
    SPLITK_BLOCK_SIZE=K_DIM,  # == K so EVEN_K stays consistent with single split
    num_warps=4,
    num_stages=2,
    waves_per_eu=2,
    matrix_instr_nonkdim=16,
    cache_modifier=".cg",
)


def regime_cases():
    cases = []
    for b in CONCURRENCY:
        m = b * SEQLEN
        cases.append({
            "test_case_id": f"c{b}",
            "B": b,
            "M": m,
            "N": N_DIM,
            "K": K_DIM,
            "params": {"B": b, "seqlen": SEQLEN, "M": m, "N": N_DIM, "K": K_DIM},
        })
    return cases


def write_test_cases():
    json.dump(regime_cases(), open(TEST_CASES, "w"), indent=2)


# ---- helper symbols the kernel references from module globals ----
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


def _load_kernel(path):
    spec = importlib.util.spec_from_file_location("k_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # inject helper symbols + tl alias into the kernel's module globals so the
    # @triton.jit body can resolve them at JIT time.
    import triton.language as _tl
    inj = mod.__dict__
    inj.setdefault("tl", _tl)
    inj.setdefault("triton", triton)
    inj["remap_xcd"] = remap_xcd
    inj["pid_grid"] = pid_grid
    kern = getattr(mod, KERNEL_NAME)
    # also patch the JITFunction's own global namespace
    try:
        kern.fn.__globals__.update({
            "remap_xcd": remap_xcd,
            "pid_grid": pid_grid,
            "tl": _tl,
            "triton": triton,
        })
    except Exception:
        pass
    return kern


def build_inputs(M, N, K, seed=42, dtype=torch.bfloat16, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    # A: (M, K) row-major ; W stored (N, K) then transposed to (K, N) like the wrapper
    a = torch.randn((M, K), generator=g, dtype=dtype, device=device)
    w = torch.randn((N, K), generator=g, dtype=dtype, device=device)
    b = w.T  # (K, N), strides (1, K)
    bias = torch.randn((N,), generator=g, dtype=dtype, device=device)
    c = torch.empty((M, N), dtype=dtype, device=device)
    return a, b, bias, c


def _grid(M, N, meta):
    return (meta["NUM_KSPLIT"]
            * triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)


def launch(kern, a, b, bias, c, M, N, K, meta):
    grid = _grid(M, N, meta)
    kern[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        0,            # stride_ck (NUM_KSPLIT==1)
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
        GROUP_SIZE_M=meta["GROUP_SIZE_M"],
        NUM_KSPLIT=meta["NUM_KSPLIT"],
        SPLITK_BLOCK_SIZE=meta["SPLITK_BLOCK_SIZE"],
        cache_modifier=meta["cache_modifier"],
        activation="",
        use_activation=False,
        ADD_BIAS=True,
        SKIP_REDUCE=False,
        num_warps=meta["num_warps"],
        num_stages=meta["num_stages"],
        waves_per_eu=meta["waves_per_eu"],
        matrix_instr_nonkdim=meta["matrix_instr_nonkdim"],
    )
    return c


def _cos(x, y):
    x = x.float().flatten()
    y = y.float().flatten()
    return torch.nn.functional.cosine_similarity(x, y, dim=0).item()


def run_correctness():
    edit = _load_kernel(SOURCE_FILE)
    gold = _load_kernel(GOLDEN_FILE)
    for tc in regime_cases():
        M, N, K = tc["M"], tc["N"], tc["K"]
        ae = build_inputs(M, N, K, seed=42)
        ag = build_inputs(M, N, K, seed=42)
        out_e = launch(edit, *ae, M, N, K, META)
        out_g = launch(gold, *ag, M, N, K, META)
        torch.cuda.synchronize()
        cos = _cos(out_e, out_g)
        denom = out_g.float().abs().max().item()
        max_rel = (out_e.float() - out_g.float()).abs().max().item() / (denom + 1e-6)
        if not (cos >= 0.99 and max_rel < 1e-2):
            return False, f"{tc['test_case_id']}: cos={cos:.5f} max_rel={max_rel:.4e}"
    return True, None


def run_performance(warmup=10, iters=100):
    edit = _load_kernel(SOURCE_FILE)
    out = []
    for tc in regime_cases():
        M, N, K = tc["M"], tc["N"], tc["K"]
        args = build_inputs(M, N, K, seed=42)
        for _ in range(warmup):
            launch(edit, *args, M, N, K, META)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for j in range(iters):
            starts[j].record()
            launch(edit, *args, M, N, K, META)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters
        out.append({
            "test_case_id": tc["test_case_id"],
            "execution_time_ms": avg,
            "params": tc["params"],
        })
    return out
