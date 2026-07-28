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
    os.environ.setdefault("AITER_REBUILD", "2")  # incremental ninja rebuild: keep object cache, recompile only edited sources (avoids full CK re-compile on every agent edit)
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "jit"))
    if (WORKSPACE / "aiter_meta").is_dir():
        os.environ["AITER_META_DIR"] = str(WORKSPACE / "aiter_meta")
        # Blob codegen (gen_instances.py) runs in a subprocess and imports
        # chip_info from aiter/jit/utils; propagate that dir via PYTHONPATH or the
        # codegen fails silently and the build errors with
        # "gemm_..._lookup.h file not found".
        try:
            # Use find_spec (not import) so we do NOT execute/cache the installed
            # aiter before a task that seeds an editable aiter package prepends the
            # workspace to sys.path (importing here would pin the installed copy).
            import importlib.util as _ilu

            _spec = _ilu.find_spec("aiter")
            if _spec and _spec.submodule_search_locations:
                _utils = str(
                    Path(_spec.submodule_search_locations[0]) / "jit" / "utils"
                )
                os.environ["PYTHONPATH"] = (
                    _utils + os.pathsep + os.environ.get("PYTHONPATH", "")
                )
        except Exception:
            pass
    if (WORKSPACE / "aiter").is_dir():
        sys.path.insert(0, str(WORKSPACE))
        os.environ.setdefault(
            "AITER_META_DIR",
            "/usr/local/lib/python3.12/dist-packages/aiter_meta",
        )
        # aiter.utility.aiter_types locates aiter_enum.h relative to the parent of
        # the (seeded) aiter package: <WORKSPACE>/aiter_meta/csrc/include/... . When
        # a task seeds only the aiter package, symlink the installed aiter_meta there
        # so the task runs without needing to seed the whole aiter_meta tree.
        _meta_ws = WORKSPACE / "aiter_meta"
        if not _meta_ws.exists():
            try:
                _installed_meta = Path(os.environ["AITER_META_DIR"])
                if _installed_meta.is_dir():
                    _meta_ws.symlink_to(_installed_meta, target_is_directory=True)
            except Exception:
                pass
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(fn, repetition):
    import time
    import torch

    times_ms = []
    for _ in range(max(1, int(repetition))):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def _benchmark_cuda_graph_or_events(
    fn,
    warmup=5,
    repetition=30,
    target_ms=1.0,
    max_graph_repeats=200,
    use_cuda_graph=True,
    **_,
):
    import torch

    for _ in range(max(0, int(warmup))):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    metadata = {
        "benchmark_target_ms": float(target_ms),
        "benchmark_samples": int(repetition),
        "benchmark_max_repeats": int(max_graph_repeats),
    }
    if not torch.cuda.is_available() or not use_cuda_graph:
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update(
            benchmark_method=(
                "cpu_timer_fallback"
                if not torch.cuda.is_available()
                else "cuda_event_fallback"
            ),
            benchmark_effective_repeats=int(repetition),
            benchmark_fallback_reason=(
                "cuda_unavailable"
                if not torch.cuda.is_available()
                else "cuda_graph_disabled"
            ),
        )
        return sum(times) / len(times), metadata

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            estimate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(estimate_graph):
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            estimate_graph.replay()
            end_event.record(stream)
            torch.cuda.synchronize()
            estimate_ms = start_event.elapsed_time(end_event) / 3
            repeats = min(
                max_graph_repeats,
                max(1, int(target_ms / max(estimate_ms, 1e-9))),
            )

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    fn()
            torch.cuda.synchronize()

            times = []
            for _ in range(max(1, int(repetition))):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event) / repeats)

        mean_ms = sum(times) / len(times)
        if mean_ms < 1e-4:
            raise RuntimeError("empty_cuda_graph_capture")
        metadata.update(
            benchmark_method="cuda_graph",
            benchmark_effective_repeats=int(repeats),
        )
        return mean_ms, metadata
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update(
            benchmark_method="cuda_event_fallback",
            benchmark_effective_repeats=int(repetition),
            benchmark_fallback_reason=(
                f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}"
            ),
        )
        return sum(times) / len(times), metadata


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
    torch_quant = aiter.get_torch_quant(quant_type)

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
        w1_quant, w1_scale = torch_quant(w1, quant_dtype=weight_dtype)
        w2_quant, w2_scale = torch_quant(w2, quant_dtype=weight_dtype)

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
    from aiter.fused_moe import fused_moe

    return fused_moe(
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
    )


def _moe_reference(inputs: dict):
    torch = _torch()
    aiter = _import_aiter()
    from aiter import dtypes
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    torch_quant = aiter.get_torch_quant(inputs["quant_type"])
    params = inputs["params"]
    if inputs["quant_type"] == aiter.QuantType.per_1x128:
        a1_quant, a1_scale = torch_quant(
            inputs["hidden"].view(inputs["hidden"].shape[0], -1, 128),
            quant_dtype=inputs["activation_dtype"],
        )
        a1_quant = a1_quant.view(inputs["hidden"].shape)
        a1_scale = a1_scale.squeeze(-1)
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
        a2_quant, a2_scale = torch_quant(
            stage1.view(stage1.shape[0], -1, 128),
            quant_dtype=inputs["activation_dtype"],
        )
        a2_scale = a2_scale.view(
            stage1.shape[0], params["topk"], -1
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


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case, correctness=True)
        got = _run(inputs)
        torch.cuda.synchronize()
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
        print("correctness PASS", case["id"])


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _make(case, correctness=False)
        _run(inputs)
        _torch().cuda.synchronize()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=3,
            repetition=20,
            target_ms=1.0,
            max_graph_repeats=100,
        )
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
