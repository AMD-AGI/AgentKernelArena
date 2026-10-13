#!/usr/bin/env python3
"""Real launcher + correctness + perf harness for the raw @triton.jit
`fused_moe_kernel` (vLLM fused MoE GEMM, fp8 w8a8 block-scale).

Workload regime: input seqlen = output seqlen = 1024; concurrency B in {2,32,64}.
This is a token-parallel MoE GEMM op, so num_tokens M = B*1024 (prefill).
We map onto the captured base case (up-projection):
  K=7168, N=512, num_experts=256, topk=8, group_n=group_k=128, fp8_w8a8 block-scale.

The kernel is launched from BOTH the editable source and a frozen golden copy on
identical seeded inputs; outputs are compared (cosine + max-rel / tolerance).
"""
import os, sys, json, math, importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_EDIT = os.path.join(TASK_DIR, "source", "triton_fused_moe_kernel.py")
SRC_GOLD = os.path.join(TASK_DIR, "source", "source_golden", "triton_fused_moe_kernel.py")
KERNEL_NAME = "fused_moe_kernel"

# ---- fixed model dims from captured base case (up-projection) ----
K = 7168
N = 512
NUM_EXPERTS = 256
TOPK = 8
GROUP_N = 128
GROUP_K = 128
SEQLEN = 1024

# Meta params (from captured base case kwargs_sig)
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128
GROUP_SIZE_M = 1
NUM_WARPS = 8
NUM_STAGES = 2

CONCURRENCY = {"c2": 2, "c32": 32, "c64": 64}

FP8 = getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn)


def _make_write_zeros():
    """The sibling @triton.jit helper `write_zeros_to_output` is referenced by
    the kernel but was NOT captured by inspect.getsource (it lives next to the
    kernel in vllm.model_executor.layers.fused_moe.fused_moe). We inject the
    exact upstream definition into the loaded module's globals so the JIT can
    resolve the name. This does not modify the kernel source file. The branch
    using it is dead under our routing (no expert == -1).
    """
    import triton
    import triton.language as tl

    @triton.jit
    def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                              token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                              compute_type):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    return write_zeros_to_output


def _load_kernel(path):
    spec = importlib.util.spec_from_file_location("k_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # inject sibling dependency not captured by inspect.getsource
    if not hasattr(mod, "write_zeros_to_output"):
        mod.write_zeros_to_output = _make_write_zeros()
        kern = getattr(mod, KERNEL_NAME)
        # ensure the JITFunction sees it in its global namespace
        try:
            kern.fn.__globals__["write_zeros_to_output"] = mod.write_zeros_to_output
        except Exception:
            pass
        try:
            kern.__globals__["write_zeros_to_output"] = mod.write_zeros_to_output
        except Exception:
            pass
    return getattr(mod, KERNEL_NAME)


def _fp8_minmax():
    fi = torch.finfo(FP8)
    return fi.min, fi.max


def build_inputs(B, seed=42):
    """Build a deterministic set of inputs for concurrency B.

    M = B * SEQLEN tokens, each routed to TOPK experts.
    Returns (args, meta) where meta holds grid + launch kwargs.
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    M = B * SEQLEN
    num_valid_tokens = M * TOPK  # entries in expanded token space

    fmin, fmax = _fp8_minmax()

    # A: [M, K] activations in fp8 (block-scale quantized along K in groups of GROUP_K)
    a_f = (torch.randn(M, K, generator=g, device=dev, dtype=torch.float32) * 0.1)
    a_ptr = a_f.clamp(fmin, fmax).to(FP8)

    # B (weights): [E, N, K] in fp8
    b_f = (torch.randn(NUM_EXPERTS, N, K, generator=g, device=dev, dtype=torch.float32) * 0.05)
    b_ptr = b_f.clamp(fmin, fmax).to(FP8)

    # C output: [M, TOPK, N] bf16
    c_ptr = torch.zeros(M, TOPK, N, device=dev, dtype=torch.bfloat16)

    # scales (block-wise): a_scale [M, K/GROUP_K], b_scale [E, N/GROUP_N, K/GROUP_K]
    k_groups = K // GROUP_K
    n_groups = N // GROUP_N
    a_scale = (torch.rand(M, k_groups, generator=g, device=dev, dtype=torch.float32) * 0.01 + 0.005)
    b_scale = (torch.rand(NUM_EXPERTS, n_groups, k_groups, generator=g, device=dev, dtype=torch.float32) * 0.01 + 0.005)

    # ---- MoE routing metadata (moe_align_block_size style) ----
    # Each of M tokens picks TOPK experts. We assign round-robin to keep it
    # deterministic and ensure every expert block is valid (no -1).
    topk_ids = torch.empty(M, TOPK, dtype=torch.int64, device="cpu")
    for t in range(TOPK):
        topk_ids[:, t] = (torch.arange(M) + t) % NUM_EXPERTS
    topk_ids = topk_ids.reshape(-1)  # [M*TOPK], expanded token e maps token e//TOPK

    # group expanded-token indices by expert, pad each expert bucket up to a
    # multiple of BLOCK_SIZE_M with the sentinel `num_valid_tokens`.
    sorted_list = []
    expert_block_ids = []
    for e in range(NUM_EXPERTS):
        idx = (topk_ids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        pad = (-idx.numel()) % BLOCK_SIZE_M
        if pad:
            idx = torch.cat([idx, torch.full((pad,), num_valid_tokens, dtype=torch.int64)])
        nblocks = idx.numel() // BLOCK_SIZE_M
        sorted_list.append(idx)
        expert_block_ids.append(torch.full((nblocks,), e, dtype=torch.int64))

    sorted_token_ids = torch.cat(sorted_list).to(torch.int32).to(dev)
    expert_ids = torch.cat(expert_block_ids).to(torch.int32).to(dev)
    num_tokens_post_padded = torch.tensor([sorted_token_ids.numel()], dtype=torch.int32, device=dev)
    EM = sorted_token_ids.numel()

    # strides
    stride_am, stride_ak = a_ptr.stride()
    stride_be, stride_bn, stride_bk = b_ptr.stride()  # [E,N,K]
    # C is [M,TOPK,N] but kernel indexes flat [M*TOPK, N]: stride_cm=N, stride_cn=1
    stride_cm, stride_cn = N, 1
    stride_asm, stride_ask = a_scale.stride()
    stride_bse, stride_bsn, stride_bsk = b_scale.stride()  # [E,n_groups,k_groups]
    stride_bbe, stride_bbn = 0, 0

    args = [
        a_ptr, b_ptr, c_ptr,
        None,                 # b_bias_ptr
        a_scale, b_scale,
        None,                 # topk_weights_ptr (MUL_ROUTED_WEIGHT=False)
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        stride_am, stride_ak,
        stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        stride_bbe, stride_bbn,
        GROUP_N, GROUP_K,
    ]
    meta = dict(EM=EM, c_view=c_ptr)
    return args, meta


def _kwargs():
    import triton.language as tl
    return dict(
        naive_block_assignment=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=1,
        MUL_ROUTED_WEIGHT=False,
        top_k=TOPK,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        HAS_BIAS=False,
    )


def _grid(EM):
    num_pid_m = math.ceil(EM / BLOCK_SIZE_M)
    num_pid_n = math.ceil(N / BLOCK_SIZE_N)
    return (num_pid_m * num_pid_n,)


def launch(kern, args, meta):
    kwargs = _kwargs()
    grid = _grid(meta["EM"])
    kern[grid](*args, num_warps=NUM_WARPS, num_stages=NUM_STAGES, **kwargs)
    torch.cuda.synchronize()
    return meta["c_view"]


def _cosine(a, b):
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return (a @ b / denom).item()


def run_correctness():
    kern_e = _load_kernel(SRC_EDIT)
    kern_g = _load_kernel(SRC_GOLD)
    for cid, B in CONCURRENCY.items():
        args_e, meta_e = build_inputs(B, seed=42)
        out_e = launch(kern_e, args_e, meta_e).clone()
        args_g, meta_g = build_inputs(B, seed=42)
        out_g = launch(kern_g, args_g, meta_g).clone()
        if out_e.shape != out_g.shape:
            return False, f"{cid}: shape mismatch {tuple(out_e.shape)} vs {tuple(out_g.shape)}"
        cos = _cosine(out_e, out_g)
        a = out_e.to(torch.float32)
        b = out_g.to(torch.float32)
        denom = b.abs().clamp_min(1e-6)
        max_rel = ((a - b).abs() / denom).max().item()
        if not (cos >= 0.99):
            return False, f"{cid}: cosine {cos:.6f} < 0.99 (max_rel {max_rel:.4g})"
        if max_rel > 1e-2 and not torch.allclose(a, b, atol=5e-2, rtol=5e-2):
            return False, f"{cid}: max_rel {max_rel:.4g} too large (cos {cos:.6f})"
    return True, None


def run_performance():
    kern = _load_kernel(SRC_EDIT)
    results = []
    for cid, B in CONCURRENCY.items():
        args, meta = build_inputs(B, seed=42)
        kwargs = _kwargs()
        grid = _grid(meta["EM"])
        for _ in range(10):
            kern[grid](*args, num_warps=NUM_WARPS, num_stages=NUM_STAGES, **kwargs)
        torch.cuda.synchronize()
        n_iter = 100
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for j in range(n_iter):
            starts[j].record()
            kern[grid](*args, num_warps=NUM_WARPS, num_stages=NUM_STAGES, **kwargs)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
        results.append({
            "test_case_id": cid,
            "execution_time_ms": avg,
            "params": {"B": B, "M": B * SEQLEN, "K": K, "N": N,
                       "num_experts": NUM_EXPERTS, "topk": TOPK,
                       "group": GROUP_K, "dtype": "fp8_w8a8_blockscale"},
        })
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok, err = run_correctness()
        print("Correctness:", "PASS" if ok else "FAIL")
        if err:
            print("Error:", err)
    else:
        for r in run_performance():
            print(f"Performance: {r['execution_time_ms']:.4f} ms ({r['test_case_id']})")
