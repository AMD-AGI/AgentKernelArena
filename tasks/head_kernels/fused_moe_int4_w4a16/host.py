"""Host wrapper for the int4 W4A16 fused-MoE GEMM (fused_moe_kernel_gptq_awq).

Builds synthetic-but-realistic inputs for the Kimi-K2.6 MoE GEMM shapes
(E=384, top_k=8, hidden=7168, shard gate_up N=1024 / down K=512, group_size=32),
and launches a *given* kernel callable so the harness can compare the local
(editable) kernel against the installed-original golden under identical inputs.
"""
import torch
import triton
import triton.language as tl

COMPUTE_TYPE = tl.bfloat16
TORCH_DTYPE = torch.bfloat16


def moe_align_block_size(topk_ids, block_size, num_experts):
    """Pure-torch replica of vLLM's moe_align_block_size.

    slot id i represents token i // top_k; padding uses sentinel M * top_k.
    """
    M, top_k = topk_ids.shape
    numel = M * top_k
    flat_expert = topk_ids.reshape(-1)
    slot_ids = torch.arange(numel, device=topk_ids.device, dtype=torch.int32)

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
            sentinel = torch.full((pad,), numel, device=topk_ids.device, dtype=torch.int32)
            toks = torch.cat([toks, sentinel])
        sorted_token_chunks.append(toks)
        expert_id_chunks.append(
            torch.full((padded // block_size,), e, device=topk_ids.device, dtype=torch.int32)
        )

    if not sorted_token_chunks:
        empty = torch.empty((0,), device=topk_ids.device, dtype=torch.int32)
        return empty, empty, torch.tensor([0], device=topk_ids.device, dtype=torch.int32)

    sorted_token_ids = torch.cat(sorted_token_chunks)
    expert_ids = torch.cat(expert_id_chunks)
    num_tokens_post_padded = torch.tensor(
        [sorted_token_ids.numel()], device=topk_ids.device, dtype=torch.int32
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def build_inputs(M, N, K, E=384, top_k=8, group_size=32, has_zp=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    a = (torch.randn(M, K, generator=g, device=dev, dtype=torch.float32) * 0.5).to(TORCH_DTYPE)
    scores = torch.randn(M, E, generator=g, device=dev, dtype=torch.float32)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.rand(M, top_k, generator=g, device=dev, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    Kp = K // 2  # int4 packed along K (2 nibbles per byte)
    B = torch.randint(0, 256, (E, N, Kp), generator=g, device=dev, dtype=torch.uint8)
    n_kgroups = K // group_size
    B_scale = (torch.randn(E, N, n_kgroups, generator=g, device=dev, dtype=torch.float32) * 0.01).to(TORCH_DTYPE)
    B_zp = None
    if has_zp:
        B_zp = torch.randint(0, 256, (E, N // 2, n_kgroups), generator=g, device=dev, dtype=torch.uint8)

    if M <= 20:
        bsm = 16
    elif M <= 40:
        bsm = 32
    else:
        bsm = 64
    # Tuned config (from benchmark_moe autotune) so the kernel-level pass stacks
    # on top of GEMM tuning rather than the weak default (BK=32, GROUP=1).
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8 if M >= 64 else 1,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 0,
    }
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )
    return {
        "a": a, "B": B, "B_scale": B_scale, "B_zp": B_zp,
        "topk_weights": topk_weights, "topk_ids": topk_ids,
        "sorted_ids": sorted_ids, "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
        "N": N, "K": K, "E": E, "top_k": top_k, "group_size": group_size,
        "has_zp": has_zp, "config": config, "M": M,
    }


def invoke(kernel_fn, inp, mul_routed_weight=True):
    a, B, C_scale = inp["a"], inp["B"], inp["B_scale"]
    B_zp = inp["B_zp"]
    cfg = inp["config"]
    M, N, K, top_k = inp["M"], inp["N"], inp["K"], inp["top_k"]
    sorted_ids = inp["sorted_ids"]
    expert_ids = inp["expert_ids"]
    num_tokens_post_padded = inp["num_tokens_post_padded"]
    num_tokens = M * top_k
    C = torch.zeros(num_tokens, N, device="cuda", dtype=TORCH_DTYPE)

    EM = sorted_ids.size(0)
    if M < cfg["BLOCK_SIZE_M"]:
        EM = min(sorted_ids.size(0), M * top_k * cfg["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    # Constexpr / meta keyword args. Some are kernel-variant-specific (e.g.
    # SPLIT_K exists in the local ROCm-patched candidate but NOT in the
    # installed-original vLLM golden). Filter by the kernel's actual arg_names
    # so the SAME inputs launch BOTH golden and candidate without a signature
    # mismatch (passing an unknown constexpr raises KeyError in Triton).
    meta_kwargs = {
        "block_k_diviable": (K % cfg["BLOCK_SIZE_K"] == 0),
        "group_size": inp["group_size"],
        "BLOCK_SIZE_M": cfg["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": cfg["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": cfg["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": cfg["GROUP_SIZE_M"],
        "SPLIT_K": cfg["SPLIT_K"],
        "MUL_ROUTED_WEIGHT": mul_routed_weight,
        "top_k": top_k,
        "compute_type": COMPUTE_TYPE,
        "has_zp": B_zp is not None,
        "use_int4_w4a16": True,
        "use_int8_w8a16": False,
    }
    arg_names = set(getattr(kernel_fn, "arg_names", []))
    if arg_names:
        meta_kwargs = {k: v for k, v in meta_kwargs.items() if k in arg_names}

    # Launch-control kwargs (not in arg_names): always allowed by the Triton
    # launcher. Only forward AMD-specific knobs that the kernel actually needs.
    launch_kwargs = {
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
    }
    if cfg.get("matrix_instr_nonkdim") is not None:
        launch_kwargs["matrix_instr_nonkdim"] = cfg["matrix_instr_nonkdim"]
    if cfg.get("kpack") is not None:
        launch_kwargs["kpack"] = cfg["kpack"]
    if cfg.get("waves_per_eu") is not None:
        launch_kwargs["waves_per_eu"] = cfg["waves_per_eu"]

    kernel_fn[grid](
        a, B, C, C_scale, B_zp,
        inp["topk_weights"], sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_tokens,
        a.stride(0), a.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        C_scale.stride(0), C_scale.stride(2), C_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        **meta_kwargs,
        **launch_kwargs,
    )
    return C
