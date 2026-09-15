#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.environ.setdefault("AITER_REBUILD", "1")
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "jit"))
    if (WORKSPACE / "aiter_meta").is_dir():
        os.environ["AITER_META_DIR"] = str(WORKSPACE / "aiter_meta")
    if (WORKSPACE / "aiter").is_dir():
        sys.path.insert(0, str(WORKSPACE))
        os.environ.setdefault(
            "AITER_META_DIR",
            "/usr/local/lib/python3.12/dist-packages/aiter_meta",
        )
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<


def _write_report(rows: list[dict]) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    return torch


def _import_aiter():
    import aiter

    return aiter


def _make_attention(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    params = dict(case["params"])
    ctx_len = min(params["ctx_len"], 128) if correctness else params["ctx_len"]
    num_seqs = params["q_tokens"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_size = params["head_size"]
    block_size = params["block_size"]
    pages_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = num_seqs * pages_per_seq

    torch.manual_seed(7)
    query = torch.randn(
        (num_seqs, num_q_heads, head_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        (num_blocks, block_size, num_kv_heads, head_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if params["kv_dtype"] == "fp8":
        key = kv.to(torch.float8_e4m3fn)
        value = (kv * 0.7).to(torch.float8_e4m3fn)
    else:
        key = kv
        value = kv * 0.7

    output = torch.empty_like(query)
    cu_seqlens_q = torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32)
    seqused_k = torch.full(
        (num_seqs,), ctx_len, device="cuda", dtype=torch.int32
    )
    block_table = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).view(num_seqs, pages_per_seq)
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    return {
        "cfg": case,
        "query": query,
        "key": key,
        "value": value,
        "output": output,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "block_table": block_table,
        "ctx_len": ctx_len,
        "scale": head_size**-0.5,
        "one": one,
    }


def _run_attention(inputs: dict):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    unified_attention(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["output"],
        inputs["cu_seqlens_q"],
        1,
        inputs["seqused_k"],
        inputs["ctx_len"],
        inputs["scale"],
        True,
        (-1, -1),
        inputs["block_table"],
        0.0,
        inputs["one"],
        inputs["one"],
        inputs["one"],
    )
    return inputs["output"]


def _attention_reference(inputs: dict):
    torch = _torch()
    query = inputs["query"].float()
    key = inputs["key"].float()
    value = inputs["value"].float()
    outputs = []
    for seq_idx in range(query.shape[0]):
        block_ids = inputs["block_table"][seq_idx]
        key_seq = key[block_ids].reshape(-1, key.shape[2], key.shape[3])
        value_seq = value[block_ids].reshape(-1, value.shape[2], value.shape[3])
        key_seq = key_seq[: inputs["ctx_len"]]
        value_seq = value_seq[: inputs["ctx_len"]]
        ratio = query.shape[1] // key_seq.shape[1]
        key_seq = key_seq.repeat_interleave(ratio, dim=1)
        value_seq = value_seq.repeat_interleave(ratio, dim=1)
        scores = (
            torch.einsum("hd,khd->hk", query[seq_idx], key_seq)
            * inputs["scale"]
        )
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probs, value_seq))
    return torch.stack(outputs).to(inputs["output"].dtype)


def _make_gemm(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    params = dict(case["params"])
    m = min(params["m"], 64) if correctness else params["m"]
    n = params["n"]
    k = params["k"]
    torch.manual_seed(9)
    x = (torch.rand((m, k), device="cuda") * 0.2 - 0.1).to(
        torch.float8_e4m3fn
    )
    weight = (torch.rand((n, k), device="cuda") * 0.2 - 0.1).to(
        torch.float8_e4m3fn
    )
    x_scale = (
        torch.rand((m, k // 128), device="cuda", dtype=torch.float32) * 0.1
        + 0.01
    )
    w_scale = (
        torch.rand(
            (math.ceil(n / 128), k // 128),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.1
        + 0.01
    )
    return {
        "cfg": case,
        "x": x,
        "weight": weight,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "shape": [m, n, k],
    }


def _run_gemm(inputs: dict):
    return _import_aiter().gemm_a8w8_blockscale(
        inputs["x"],
        inputs["weight"],
        inputs["x_scale"],
        inputs["w_scale"],
        _torch().bfloat16,
    )


def _gemm_reference(inputs: dict):
    torch = _torch()
    m, n, k = inputs["shape"]
    x = (
        inputs["x"].float()
        * inputs["x_scale"].repeat_interleave(128, dim=1)[:, :k]
    )
    weight = (
        inputs["weight"].float()
        * inputs["w_scale"]
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)[:n, :k]
    )
    return (x @ weight.t()).to(torch.bfloat16)


def _make_quant(case: dict) -> dict:
    torch = _torch()
    torch.manual_seed(11)
    shape = tuple(case["params"]["shape"])
    input_tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, device="cuda", dtype=torch.float32)
    return {
        "cfg": case,
        "input": input_tensor,
        "output": output,
        "scale": scale,
    }


def _run_quant(inputs: dict):
    _import_aiter().dynamic_per_tensor_quant(
        inputs["output"], inputs["input"], inputs["scale"]
    )
    return inputs["output"]


def _load_mhc_module():
    # Import the installed package first so its custom ops are registered once.
    # Then suppress registration while loading the editable workspace copy;
    # otherwise both copies call direct_register_custom_op with the same names.
    import vllm.model_executor.kernels.mhc.tilelang_kernels  # noqa: F401
    import vllm.utils.torch_utils as torch_utils

    path = WORKSPACE / "mhc" / "tilelang.py"
    spec = importlib.util.spec_from_file_location("ka_mhc_tilelang", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    original_register = torch_utils.direct_register_custom_op
    torch_utils.direct_register_custom_op = lambda *args, **kwargs: None
    try:
        spec.loader.exec_module(module)
    finally:
        torch_utils.direct_register_custom_op = original_register
    return module


def _make_mhc(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    params = dict(case["params"])
    tokens = min(params["tokens"], 64) if correctness else params["tokens"]
    hidden_size = params["hidden_size"]
    hc_mult = params["hc_mult"]
    torch.manual_seed(13)
    x = torch.randn(
        (tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn(
        (tokens, hc_mult, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    post_mix = torch.randn(
        (tokens, hc_mult, 1), device="cuda", dtype=torch.float32
    )
    comb_mix = torch.softmax(
        torch.randn(
            (tokens, hc_mult, hc_mult),
            device="cuda",
            dtype=torch.float32,
        ),
        dim=-1,
    )
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    fn = (
        torch.randn(
            (hc_mult3, hc_mult * hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.001
    )
    hc_scale = torch.ones(3, device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device="cuda", dtype=torch.float32)
    return {
        "cfg": case,
        "params": params,
        "x": x,
        "residual": residual,
        "post_mix": post_mix,
        "comb_mix": comb_mix,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "module": _load_mhc_module(),
    }


def _run_mhc(inputs: dict):
    params = inputs["params"]
    return inputs["module"].mhc_fused_post_pre_tilelang(
        inputs["x"],
        inputs["residual"],
        inputs["post_mix"],
        inputs["comb_mix"],
        inputs["fn"],
        inputs["hc_scale"],
        inputs["hc_base"],
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult"],
        params["sinkhorn_repeat"],
        1,
        1,
        None,
        0.0,
    )


def _mhc_reference(inputs: dict):
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

    params = inputs["params"]
    residual = mhc_post_torch(
        inputs["x"],
        inputs["residual"],
        inputs["post_mix"],
        inputs["comb_mix"],
    )
    post_mix, comb_mix, layer_input = mhc_pre_torch(
        residual,
        inputs["fn"],
        inputs["hc_scale"],
        inputs["hc_base"],
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult"],
        params["sinkhorn_repeat"],
    )
    return residual, post_mix, comb_mix, layer_input


def _make_mla(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    params = dict(case["params"])
    batch = min(params["batch"], 64) if correctness else params["batch"]
    ctx_len = min(params["ctx_len"], 128) if correctness else params["ctx_len"]
    capacity = (
        min(params["kv_capacity"], batch * ctx_len + 1024)
        if correctness
        else params["kv_capacity"]
    )
    torch.manual_seed(17)
    query = torch.randn(
        (batch, params["num_heads"], params["qk_dim"]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        (capacity, params["page_size"], params["kv_heads"], params["qk_dim"]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (batch, params["num_heads"], params["v_dim"]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    qo_indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    kv_indptr = torch.arange(
        0, (batch + 1) * ctx_len, ctx_len, device="cuda", dtype=torch.int32
    )
    kv_indices = (
        torch.arange(batch * ctx_len, device="cuda", dtype=torch.int32)
        % capacity
    )
    last_page_lens = torch.ones(batch, device="cuda", dtype=torch.int32)
    return {
        "cfg": case,
        "params": params,
        "query": query,
        "kv": kv,
        "output": output,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_lens": last_page_lens,
        "ctx_len": ctx_len,
    }


def _run_mla(inputs: dict):
    params = inputs["params"]
    return _import_aiter().mla.mla_decode_fwd(
        inputs["query"],
        inputs["kv"],
        inputs["output"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["last_page_lens"],
        1,
        params["page_size"],
        params["kv_heads"],
        params["qk_dim"] ** -0.5,
        num_kv_splits=None,
        return_lse=False,
    )


def _mla_reference(inputs: dict):
    torch = _torch()
    query = inputs["query"].float()
    outputs = []
    for seq_idx in range(query.shape[0]):
        start = seq_idx * inputs["ctx_len"]
        end = (seq_idx + 1) * inputs["ctx_len"]
        indices = inputs["kv_indices"][start:end]
        kv = inputs["kv"][indices].reshape(
            -1,
            inputs["params"]["kv_heads"],
            inputs["params"]["qk_dim"],
        )
        key = kv
        value = kv[..., : inputs["params"]["v_dim"]]
        ratio = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(ratio, dim=1)
        value = value.repeat_interleave(ratio, dim=1)
        scores = (
            torch.einsum("hd,khd->hk", query[seq_idx], key.float())
            * (inputs["params"]["qk_dim"] ** -0.5)
        )
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probs, value.float()))
    return torch.stack(outputs).to(inputs["output"].dtype)


def _moe_enums(params: dict, aiter):
    quant_type = {
        "per_Tensor": aiter.QuantType.per_Tensor,
        "per_1x128": aiter.QuantType.per_1x128,
        "per_1x32": aiter.QuantType.per_1x32,
    }[params["quant_type"]]
    activation = {
        "silu": aiter.ActivationType.Silu,
        "swiglu": aiter.ActivationType.Swiglu,
    }[params["activation"]]
    return quant_type, activation


def _quantize_moe_weight(weight, quant_type, weight_dtype, aiter):
    if quant_type != aiter.QuantType.per_1x128:
        return aiter.get_torch_quant(quant_type)(weight, quant_dtype=weight_dtype)
    # A1x128 uses W128x128 scales, independently for each expert. The runtime's
    # get_torch_quant(per_1x128) is a two-dimensional activation quantizer.
    experts, rows, columns = weight.shape
    if rows % 128 or columns % 128:
        raise ValueError("CK FP8 weight dimensions must be divisible by 128")
    blocks = weight.reshape(experts, rows // 128, 128, columns // 128, 128)
    blocks = blocks.permute(0, 1, 3, 2, 4).reshape(-1, 128 * 128)
    quantized, scales = aiter.pertoken_quant(blocks, quant_dtype=weight_dtype)
    quantized = quantized.reshape(experts, rows // 128, columns // 128, 128, 128)
    quantized = quantized.permute(0, 1, 3, 2, 4).reshape_as(weight)
    return quantized, scales.reshape(experts, rows // 128, columns // 128)


def _quantize_moe_activation(value, quantize, dtype):
    # Flatten token/top-k axes only; each row keeps its original 128-wide groups.
    quantized, scales = quantize(value.reshape(-1, value.shape[-1]), quant_dtype=dtype)
    return (quantized.reshape_as(value),
            scales.reshape(*value.shape[:-1], value.shape[-1] // 128))


def _prepare_moe(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    aiter = _import_aiter()
    from aiter import dtypes
    from aiter.fused_moe import fused_topk
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility import fp4_utils

    params = dict(case["params"])
    token = min(params["token"], 64) if correctness else params["token"]
    experts = params["experts"]
    model_dim = params["model_dim"]
    inter_dim = params["inter_dim"]
    topk = params["topk"]
    quant_type, activation = _moe_enums(params, aiter)
    activation_dtype = {
        "fp8": dtypes.fp8,
        "fp4": dtypes.fp4x2,
        "bf16": dtypes.bf16,
    }[params["a_dtype"]]
    weight_dtype = {"fp8": dtypes.fp8, "fp4": dtypes.fp4x2}[
        params["w_dtype"]
    ]

    torch.manual_seed(19)
    hidden = (
        torch.randn(
            (token, model_dim), device="cuda", dtype=dtypes.bf16
        )
        * 0.1
    )
    w1 = (
        torch.randn(
            (experts, inter_dim * 2, model_dim),
            device="cuda",
            dtype=dtypes.bf16,
        )
        * 0.03
    )
    w2 = (
        torch.randn(
            (experts, model_dim, inter_dim),
            device="cuda",
            dtype=dtypes.bf16,
        )
        * 0.03
    )
    score = torch.randn((token, experts), device="cuda", dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)
    if quant_type == aiter.QuantType.per_Tensor:
        w1_quant, w1_scale = aiter.pertoken_quant(
            w1.view(experts, -1), quant_dtype=weight_dtype
        )
        w2_quant, w2_scale = aiter.pertoken_quant(
            w2.view(experts, -1), quant_dtype=weight_dtype
        )
        w1_quant = w1_quant.view(w1.shape)
        w2_quant = w2_quant.view(w2.shape)
    else:
        w1_quant, w1_scale = _quantize_moe_weight(w1, quant_type, weight_dtype, aiter)
        w2_quant, w2_scale = _quantize_moe_weight(w2, quant_type, weight_dtype, aiter)

    if quant_type == aiter.QuantType.per_1x32:
        w1_quant = w1_quant.view(
            experts, w1.shape[1], w1.shape[2] // 2
        )
        w2_quant = w2_quant.view(
            experts, w2.shape[1], w2.shape[2] // 2
        )

    w1_reference = w1_quant
    w2_reference = w2_quant
    if (
        quant_type == aiter.QuantType.per_1x32
        and activation_dtype in (dtypes.bf16, dtypes.fp16, dtypes.fp8)
        and weight_dtype == dtypes.fp4x2
    ):
        w1_runtime = shuffle_weight_a16w4(w1_quant, 16, True)
        w1_scale_runtime = shuffle_scale_a16w4(
            w1_scale, experts, True
        )
        w2_runtime = shuffle_weight_a16w4(w2_quant, 16, False)
        w2_scale_runtime = shuffle_scale_a16w4(
            w2_scale, experts, False
        )
    else:
        w1_runtime = shuffle_weight(w1_quant, layout=(16, 16))
        w2_runtime = shuffle_weight(w2_quant, layout=(16, 16))
        w1_scale_runtime = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_runtime = fp4_utils.e8m0_shuffle(w2_scale)

    return {
        "cfg": case,
        "params": params,
        "hidden": hidden,
        "w1": w1_runtime,
        "w2": w2_runtime,
        "w1_reference": w1_reference,
        "w2_reference": w2_reference,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1_scale_runtime": w1_scale_runtime,
        "w2_scale_runtime": w2_scale_runtime,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "quant_type": quant_type,
        "activation": activation,
        "activation_dtype": activation_dtype,
        "weight_dtype": weight_dtype,
    }


def _run_moe(inputs: dict):
    from ck_dispatch import run_ck_moe

    return run_ck_moe(
        inputs["hidden"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        w1_scale=inputs["w1_scale_runtime"],
        w2_scale=inputs["w2_scale_runtime"],
        quant_type=inputs["quant_type"],
        activation=inputs["activation"],
        dtype=_torch().bfloat16,
        activation_dtype=inputs["activation_dtype"],
    )


def _moe_reference(inputs: dict):
    torch = _torch()
    aiter = _import_aiter()
    from aiter import dtypes
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    torch_quant = aiter.get_torch_quant(inputs["quant_type"])
    params = inputs["params"]
    if inputs["quant_type"] == aiter.QuantType.per_1x128:
        a1_quant, a1_scale = _quantize_moe_activation(
            inputs["hidden"], torch_quant, inputs["activation_dtype"],
        )
    elif (
        inputs["quant_type"] == aiter.QuantType.per_1x32
        and inputs["activation_dtype"]
        in (dtypes.bf16, dtypes.fp16, dtypes.fp8)
        and inputs["weight_dtype"] == dtypes.fp4x2
    ):
        a1_quant = inputs["hidden"].to(inputs["activation_dtype"])
        a1_scale = None
    else:
        a1_quant, a1_scale = torch_quant(
            inputs["hidden"], quant_dtype=inputs["activation_dtype"]
        )

    stage1 = torch_moe_stage1(
        a1_quant,
        inputs["w1_reference"],
        inputs["w2_reference"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        dtype=torch.bfloat16,
        activation=inputs["activation"],
        quant_type=inputs["quant_type"],
        a1_scale=a1_scale,
        w1_scale=inputs["w1_scale"],
    )
    if inputs["quant_type"] == aiter.QuantType.per_1x128:
        a2_quant, a2_scale = _quantize_moe_activation(
            stage1, torch_quant, inputs["activation_dtype"],
        )
    elif (
        inputs["quant_type"] == aiter.QuantType.per_1x32
        and inputs["activation_dtype"]
        in (dtypes.bf16, dtypes.fp16, dtypes.fp8)
        and inputs["weight_dtype"] == dtypes.fp4x2
    ):
        a2_quant = stage1
        a2_scale = None
    else:
        a2_quant, a2_scale = torch_quant(
            stage1, quant_dtype=inputs["activation_dtype"]
        )
    a2_quant = a2_quant.view(stage1.shape[0], params["topk"], -1)
    return torch_moe_stage2(
        a2_quant,
        inputs["w1_reference"],
        inputs["w2_reference"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        dtype=torch.bfloat16,
        quant_type=inputs["quant_type"],
        w2_scale=inputs["w2_scale"],
        a2_scale=a2_scale,
    )


def _make(case: dict, correctness: bool = False) -> dict:
    if OPERATOR == "unified_attention":
        return _make_attention(case, correctness)
    if OPERATOR == "a8w8_blockscale_gemm":
        return _make_gemm(case, correctness)
    if OPERATOR == "dynamic_per_tensor_quant":
        return _make_quant(case)
    if OPERATOR == "mhc_fused_post_pre":
        return _make_mhc(case, correctness)
    if OPERATOR == "mla_decode":
        return _make_mla(case, correctness)
    if OPERATOR in ("ck_moe_2stage", "cktile_moe_2stage"):
        return _prepare_moe(case, correctness)
    raise KeyError(OPERATOR)


def _run(inputs: dict):
    return {
        "unified_attention": _run_attention,
        "a8w8_blockscale_gemm": _run_gemm,
        "dynamic_per_tensor_quant": _run_quant,
        "mhc_fused_post_pre": _run_mhc,
        "mla_decode": _run_mla,
        "ck_moe_2stage": _run_moe,
        "cktile_moe_2stage": _run_moe,
    }[OPERATOR](inputs)


def run_compile() -> None:
    inputs = _make(CASES[0], correctness=True)
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def _assert_output_contract(inputs, output):
    torch = _torch()
    if OPERATOR == "a8w8_blockscale_gemm":
        source = inputs["x"]
        expected_shape = tuple(inputs["shape"][:2])
    else:
        source = inputs["hidden"]
        expected_shape = tuple(source.shape)
    assert tuple(output.shape) == expected_shape, "Wrong CK output shape"
    assert output.dtype == torch.bfloat16, "CK output must be BF16"
    assert output.device == source.device, "Wrong CK output device"
    assert torch.isfinite(output).all(), "Nonfinite CK output"


def _ck_reference(inputs):
    return (_gemm_reference(inputs) if OPERATOR == "a8w8_blockscale_gemm"
            else _moe_reference(inputs))


def _prepare_timed_check(inputs):
    # Take copies before the first candidate/warmup invocation. HIP candidates
    # only receive the live tensors, never these reference/snapshot buffers.
    torch = _torch()
    originals = {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    snapshots = {key: value.detach().clone() for key, value in originals.items()}
    reference_inputs = {**inputs, **snapshots}
    expected = _ck_reference(reference_inputs)
    perturbed = dict(snapshots)
    if OPERATOR == "a8w8_blockscale_gemm":
        # Positive scales expose degenerate zero outputs even when the original
        # absolute gate accepts tiny outputs. This adds no measured score point.
        perturbed["x_scale"] = snapshots["x_scale"] * 64
        perturbed["w_scale"] = snapshots["w_scale"] * 64
    else:
        perturbed["hidden"] = -snapshots["hidden"]
    perturbed_expected = _ck_reference({**inputs, **perturbed})
    return {"originals": originals, "snapshots": snapshots, "expected": expected,
            "perturbed": perturbed, "perturbed_expected": perturbed_expected}


def _assert_readonly_inputs(inputs, expected, originals):
    torch = _torch()
    for key, before in expected.items():
        actual = inputs[key]
        assert actual is originals[key], (key, "Input tensor was replaced")
        assert actual.shape == before.shape and actual.dtype == before.dtype, key
        assert actual.device == before.device, key
        # Byte comparison supports FP8 and detects any input write; it does not
        # apply a floating-point tolerance to the immutable input contract.
        assert torch.equal(actual.contiguous().view(torch.uint8),
                           before.contiguous().view(torch.uint8)), (key, "Readonly input was modified")


def _assert_moe_magnitude(observed, expected):
    # Keep the original cosine error < .03, and extend its equal-norm
    # squared-distance bound 2*(1-cosine) to unequal output magnitudes.
    # This fixed bound is not calibrated from a baseline's measured error.
    torch = _torch()
    actual = observed.float().flatten()
    reference = expected.float().flatten()
    assert torch.isfinite(reference).all(), "Nonfinite CK MoE reference"
    signal = float(reference.square().sum())
    error = float((actual - reference).square().sum())
    assert math.isfinite(signal) and math.isfinite(error), "Nonfinite CK MoE error"
    relative_l2_squared = error / signal if signal else (0.0 if error == 0 else float("inf"))
    metrics = {"cosine_error": float(1 - torch.nn.functional.cosine_similarity(actual, reference, dim=0)),
               "relative_l2_squared": relative_l2_squared,
               "norm_ratio": math.sqrt(float(actual.square().sum()) / signal) if signal else None}
    print("CK_MOE_NUMERICS=" + json.dumps(metrics, sort_keys=True))
    assert relative_l2_squared < 0.06, ("Incorrect CK MoE output magnitude", metrics)


def _assert_ck_close(inputs, observed, expected):
    torch = _torch()
    _assert_output_contract(inputs, observed)
    if OPERATOR == "a8w8_blockscale_gemm":
        torch.testing.assert_close(observed, expected, atol=0.15, rtol=0.12)
    else:
        error = 1 - torch.nn.functional.cosine_similarity(
            observed.float().flatten(), expected.float().flatten(), dim=0)
        assert float(error) < 0.03, "Incorrect CK MoE timed output"
        _assert_moe_magnitude(observed, expected)


def _assert_timed_outputs(inputs, timed, check):
    try:
        assert timed.bound, "Timing must expose its captured invocation"
        _assert_readonly_inputs(inputs, check["snapshots"], check["originals"])
        # First inspect the buffers actually written by the measured original
        # workload, using an oracle computed before any candidate execution.
        _assert_ck_close(inputs, timed.outputs, check["expected"])
        for key, value in check["perturbed"].items():
            if value is not check["snapshots"][key]:
                inputs[key].copy_(value)
        timed.outputs.fill_(float("nan"))
        observed = timed.rerun()
        _assert_readonly_inputs(inputs, check["perturbed"], check["originals"])
        _assert_ck_close(inputs, observed, check["perturbed_expected"])
    finally:
        # Failure must not leave perturbed or candidate-corrupted inputs behind.
        for key, original in check["originals"].items():
            original.copy_(check["snapshots"][key])
            inputs[key] = original


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case, correctness=True)
        got = _run(inputs)
        torch.cuda.synchronize()
        _assert_output_contract(inputs, got)
        if OPERATOR == "unified_attention":
            torch.testing.assert_close(
                got, _attention_reference(inputs), atol=0.08, rtol=0.08
            )
        elif OPERATOR == "a8w8_blockscale_gemm":
            torch.testing.assert_close(
                got, _gemm_reference(inputs), atol=0.15, rtol=0.12
            )
        elif OPERATOR == "dynamic_per_tensor_quant":
            expected_scale = (
                inputs["input"].abs().float().max()
                / torch.finfo(torch.float8_e4m3fn).max
            )
            torch.testing.assert_close(
                inputs["scale"],
                expected_scale.reshape(1),
                atol=1e-5,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                got.float() * inputs["scale"],
                inputs["input"].float(),
                atol=0.25,
                rtol=0.15,
            )
        elif OPERATOR == "mhc_fused_post_pre":
            for actual, expected in zip(got, _mhc_reference(inputs)):
                torch.testing.assert_close(
                    actual, expected, atol=0.08, rtol=0.08
                )
        elif OPERATOR == "mla_decode":
            torch.testing.assert_close(
                inputs["output"],
                _mla_reference(inputs),
                atol=0.08,
                rtol=0.08,
            )
        else:
            expected = _moe_reference(inputs)
            cosine_error = 1 - torch.nn.functional.cosine_similarity(
                got.float().flatten(),
                expected.float().flatten(),
                dim=0,
            )
            assert torch.isfinite(got).all()
            assert float(cosine_error) < 0.03, (
                case["id"],
                float(cosine_error),
            )
            _assert_moe_magnitude(got, expected)
        print("correctness PASS", case["id"])


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _make(case, correctness=False)
        timed_check = _prepare_timed_check(inputs)
        _run(inputs)
        _torch().cuda.synchronize()
        timed = _TimedRun()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=3,
            repetition=20,
            target_ms=1.0,
            max_graph_repeats=100,
            timed_run=timed,
        )
        _assert_timed_outputs(inputs, timed, timed_check)
        metadata = {
            **case["params"],
            "model": case["model"],
            "session_breakdown_id": case["session_breakdown_id"],
            "kernel_ids": case["kernel_ids"],
            "gpu_pct": case["gpu_pct"],
            "benchmark_method": bench_meta.get("benchmark_method"),
        }
        metadata.update(
            {
                key: value
                for key, value in bench_meta.items()
                if key.startswith("benchmark_")
            }
        )
        row = {
            "test_case_id": case["id"],
            "shape": case["trace_input_shapes"],
            "execution_time_ms": execution_time_ms,
            "metadata": metadata,
        }
        rows.append(row)
        print(
            case["id"],
            f"{execution_time_ms:.6f} ms",
            bench_meta.get("benchmark_method"),
            bench_meta.get("benchmark_fallback_reason", ""),
        )
    _write_report(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["compile", "correctness", "performance", "manifest"]
    )
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    if mode == "compile":
        run_compile()
    elif mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
