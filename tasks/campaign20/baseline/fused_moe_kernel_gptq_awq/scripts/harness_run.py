#!/usr/bin/env python3
"""Real launcher + benchmark harness for the raw @triton.jit kernel
`fused_moe_kernel_gptq_awq` (vLLM int4 w4a16 AWQ/GPTQ fused MoE GEMM).

This kernel is compile-only in the arena (raw jit, no launcher). We build a
real launcher here:
  * load BOTH the editable source and a frozen golden copy
  * inject the `write_zeros_to_output` @triton.jit helper into each module's
    globals (the captured source references it but does not define it; it lives
    in vllm.fused_moe alongside the kernel). This does NOT modify the source
    file on disk.
  * synthesize deterministic, seeded inputs for the workload regime
  * recompute the launch grid FROM the regime dims
  * launch the jit kernel (writes output in-place into c)
  * run golden-vs-editable correctness (cosine + max-rel-err)
  * time each case with CUDA events (10 warmup + 100 timed)

WORKLOAD REGIME: input seqlen = output seqlen = 1024; concurrency B in {2,32,64}.
This is a token-parallel MoE GEMM, so num_tokens M = B * 1024 (prefill tokens).
All model dims (K, N, num_experts, top_k, group_size, block sizes) are kept from
the captured base case sig_04008e1e3a14.
"""
import os
import sys
import json
import importlib.util

import torch
import triton
import triton.language as tl

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fused_moe_kernel_gptq_awq.py")
GOLDEN_FILE = os.path.join(TASK_DIR, "source_golden", "triton_fused_moe_kernel_gptq_awq.py")
KERNEL_NAME = "fused_moe_kernel_gptq_awq"

# ---- Model dims captured from base case sig_04008e1e3a14 -------------------
K = 7168
N = 512
NUM_EXPERTS = 384
TOP_K = 8
GROUP_SIZE = 32
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 1
SPLIT_K = 1
MUL_ROUTED_WEIGHT = False
HAS_ZP = False
USE_INT4_W4A16 = True
USE_INT8_W8A16 = False
BLOCK_K_DIVIABLE = True
COMPUTE_TYPE = tl.bfloat16

# regime: M = B * seqlen, seqlen = 1024
SEQLEN = 1024
CASES = [("c2", 2), ("c32", 32), ("c64", 64)]


# ---------- the helper the captured source references but does not define ----
@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _load_kernel(path):
    spec = importlib.util.spec_from_file_location(
        "k_" + os.path.basename(os.path.dirname(path)), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Inject the helper into the module globals so @triton.jit can resolve it
    # at compile time. The jit kernel resolves called fns from its __globals__.
    kern = getattr(mod, KERNEL_NAME)
    kern.__globals__.setdefault("write_zeros_to_output", write_zeros_to_output)
    mod.write_zeros_to_output = write_zeros_to_output
    return kern


def _moe_align(topk_ids, num_experts, block_size, device):
    """Pure-torch replica of moe_align_block_size.

    topk_ids: (M, top_k) int32 of expert assignments.
    Returns sorted_token_ids (flat token*top_k slot ids, padded per expert
    to block_size with sentinel = M*top_k), expert_ids (one per BLOCK_SIZE_M
    block), num_tokens_post_padded (scalar tensor).
    """
    M, top_k = topk_ids.shape
    numel = M * top_k
    flat_expert = topk_ids.reshape(-1)  # (M*top_k,)
    # slot id i represents token (i // top_k) repeated; kernel divides by top_k.
    slot_ids = torch.arange(numel, device=device, dtype=torch.int32)

    sorted_token_chunks = []
    expert_id_chunks = []
    for e in range(num_experts):
        mask = flat_expert == e
        cnt = int(mask.sum().item())
        if cnt == 0:
            continue
        toks = slot_ids[mask]
        padded = ((cnt + block_size - 1) // block_size) * block_size
        pad = padded - cnt
        if pad:
            sentinel = torch.full((pad,), numel, device=device, dtype=torch.int32)
            toks = torch.cat([toks, sentinel])
        sorted_token_chunks.append(toks)
        nblocks = padded // block_size
        expert_id_chunks.append(torch.full((nblocks,), e, device=device, dtype=torch.int32))

    sorted_token_ids = torch.cat(sorted_token_chunks)
    expert_ids = torch.cat(expert_id_chunks)
    num_tokens_post_padded = torch.tensor([sorted_token_ids.numel()],
                                          device=device, dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def build_case(B, seed=42):
    """Synthesize deterministic seeded inputs for concurrency B (M=B*1024)."""
    device = "cuda"
    M = B * SEQLEN
    g = torch.Generator(device=device).manual_seed(seed)

    # A: (M, K) bf16 activations
    a = (torch.randn(M, K, generator=g, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    # B packed int4 weights: (E, N, K//2) uint8
    b = torch.randint(0, 256, (NUM_EXPERTS, N, K // 2), generator=g,
                      device=device, dtype=torch.uint8)

    # b_scale: (E, N, K//group_size) bf16
    b_scale = (torch.randn(NUM_EXPERTS, N, K // GROUP_SIZE, generator=g,
                           device=device, dtype=torch.float32) * 0.05 + 0.1).to(torch.bfloat16)

    # b_zp: None (has_zp=False)
    b_zp = None

    # topk_weights: (M, top_k) float32 routing weights
    topk_weights = torch.rand(M, TOP_K, generator=g, device=device, dtype=torch.float32)

    # topk_ids: (M, top_k) expert assignments
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), generator=g,
                             device=device, dtype=torch.int32)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align(
        topk_ids, NUM_EXPERTS, BLOCK_SIZE_M, device)

    EM = int(sorted_token_ids.numel())
    num_valid_tokens = M * TOP_K

    # C output: (M, top_k, N) bf16, zero-initialized
    c = torch.zeros(M, TOP_K, N, device=device, dtype=torch.bfloat16)

    inputs = dict(
        a=a, b=b, c=c, b_scale=b_scale, b_zp=b_zp,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        EM=EM, num_valid_tokens=num_valid_tokens, M=M,
    )
    return inputs


def grid_for(inputs):
    EM = inputs["EM"]
    num_pid_m = triton.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    return (num_pid_m * num_pid_n,)


def launch(kern, inputs):
    a = inputs["a"]; b = inputs["b"]; c = inputs["c"]
    b_scale = inputs["b_scale"]; b_zp = inputs["b_zp"]
    topk_weights = inputs["topk_weights"]
    sorted_token_ids = inputs["sorted_token_ids"]
    expert_ids = inputs["expert_ids"]
    num_tokens_post_padded = inputs["num_tokens_post_padded"]
    EM = inputs["EM"]; num_valid_tokens = inputs["num_valid_tokens"]

    # c is (M, top_k, N); kernel indexes c with stride_cm over (token) and
    # stride_cn over N. offs_token = sorted_token_ids gives the flat (token*topk)
    # index, so c is viewed as (M*top_k, N).
    c2d = c.view(-1, N)

    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_be, stride_bk, stride_bn = b.stride(0), b.stride(2), b.stride(1)
    # NOTE: in the captured layout b is (E, N, K//2); kernel uses stride_bk for
    # the K-direction (packed) and stride_bn for N. b.stride = (N*Kp, Kp, 1).
    # stride_bk moves along K (the last dim, stride 1), stride_bn moves along N.
    stride_be = b.stride(0)
    stride_bk = b.stride(2)
    stride_bn = b.stride(1)
    stride_cm, stride_cn = c2d.stride(0), c2d.stride(1)
    stride_bse = b_scale.stride(0)
    stride_bsk = b_scale.stride(2)
    stride_bsn = b_scale.stride(1)
    # b_zp is None -> strides 0
    stride_bze = stride_bzk = stride_bzn = 0

    grid = grid_for(inputs)
    kern[grid](
        a, b, c2d, b_scale, b_zp,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        stride_am, stride_ak,
        stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_bse, stride_bsk, stride_bsn,
        stride_bze, stride_bzk, stride_bzn,
        BLOCK_K_DIVIABLE, GROUP_SIZE,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M, SPLIT_K=SPLIT_K,
        MUL_ROUTED_WEIGHT=MUL_ROUTED_WEIGHT, top_k=TOP_K,
        compute_type=COMPUTE_TYPE, has_zp=HAS_ZP,
        use_int4_w4a16=USE_INT4_W4A16, use_int8_w8a16=USE_INT8_W8A16,
    )
    return c


def _cos_and_relerr(x, y):
    xf = x.detach().to(torch.float32).reshape(-1)
    yf = y.detach().to(torch.float32).reshape(-1)
    cos = torch.nn.functional.cosine_similarity(xf, yf, dim=0).item()
    denom = yf.abs().clamp_min(1e-6)
    rel = ((xf - yf).abs() / denom)
    # ignore positions where both ~0
    mask = (xf.abs() > 1e-3) | (yf.abs() > 1e-3)
    max_rel = rel[mask].max().item() if mask.any() else 0.0
    return cos, max_rel


def run_correctness():
    kern_e = _load_kernel(SOURCE_FILE)
    kern_g = _load_kernel(GOLDEN_FILE)
    for cid, B in CASES:
        in_e = build_case(B, seed=42)
        in_g = build_case(B, seed=42)
        out_e = launch(kern_e, in_e)
        out_g = launch(kern_g, in_g)
        torch.cuda.synchronize()
        cos, max_rel = _cos_and_relerr(out_e, out_g)
        if not (cos >= 0.99 and max_rel < 0.05):
            return False, f"{cid}: cos={cos:.5f} max_rel={max_rel:.5f}"
    return True, None


def run_performance(n_warmup=10, n_iter=100):
    kern = _load_kernel(SOURCE_FILE)
    results = []
    for cid, B in CASES:
        inputs = build_case(B, seed=42)
        for _ in range(n_warmup):
            launch(kern, inputs)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for j in range(n_iter):
            starts[j].record()
            launch(kern, inputs)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
        results.append({"test_case_id": cid, "execution_time_ms": avg,
                        "params": {"B": B, "M": B * SEQLEN, "K": K, "N": N,
                                   "num_experts": NUM_EXPERTS, "top_k": TOP_K}})
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok, err = run_correctness()
        print("Correctness:", "PASS" if ok else "FAIL", err or "")
    else:
        for r in run_performance():
            print(f"Performance: {r['execution_time_ms']:.4f} ms ({r['test_case_id']})")
