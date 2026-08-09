#!/usr/bin/env python3
"""Harness for the vLLM Triton WNA16 (int4/bf16) fused-MoE kernel
``fused_moe_kernel_gptq_awq`` (fused_moe.py).

The kernel is loaded from the editable workspace copy of the in-image source tree
so an optimizing agent's edits take effect. fused_moe.py registers a custom op at
import; registration is suppressed while loading the editable copy so it does not
clash with the already-registered installed copy (mirrors the vLLM MHC harness).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]

REPO_SUBDIR = "vllm_fused_moe"
KERNEL_FILE = "fused_moe.py"
EDIT_MODULE_NAME = "vllm.model_executor.layers.fused_moe._ka_fused_moe"

# How many times run_correctness rotates over the full case suite.
CORRECTNESS_ROUNDS = 3


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
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


def _load_kernel_module():
    # fused_moe.py runs direct_register_custom_op at import; suppress it so loading
    # the editable copy does not clash with the already-registered installed copy.
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa: F401
    import vllm.utils.torch_utils as torch_utils

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    original = torch_utils.direct_register_custom_op
    torch_utils.direct_register_custom_op = lambda *a, **k: None
    try:
        spec.loader.exec_module(module)
    finally:
        torch_utils.direct_register_custom_op = original
    return module


def _quant_pack_int4(w: "object", group_size: int):
    """Symmetric int4 (zero-point=8) group quant along the last (K) dim.

    Returns (packed_uint8 [E,N,K//2], scale_bf16 [E,N,K//group], deq_bf16 [E,N,K]).
    Matches the kernel dequant: deq = (q - 8) * scale, packed low nibble = even k.
    """
    torch = _torch()
    E, N, K = w.shape
    ng = K // group_size
    wg = w.reshape(E, N, ng, group_size).float()
    scale = (wg.abs().amax(dim=-1) / 7.0).clamp(min=1e-4)  # [E,N,ng]
    q = torch.round(wg / scale.unsqueeze(-1)) + 8.0
    q = q.clamp(0, 15)  # [E,N,ng,group_size]
    deq = ((q - 8.0) * scale.unsqueeze(-1)).reshape(E, N, K).to(torch.bfloat16)
    q = q.to(torch.uint8).reshape(E, N, K // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).contiguous()  # [E,N,K//2]
    return packed, scale.to(torch.bfloat16).contiguous(), deq


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    There is deliberately no correctness/performance switch here: a shape that is
    timed must also be the shape that is validated, or the scored code path can
    differ from the checked one.
    """
    torch = _torch()
    p = dict(case["params"])
    tokens = p["tokens"]
    num_experts = p["num_experts"]
    topk = p["topk"]
    hidden = p["hidden"]
    inter = p["inter"]
    group_size = p["group_size"]
    assert hidden % group_size == 0 and inter % group_size == 0

    torch.manual_seed(31)
    module = _load_kernel_module()

    # Scale inputs/weights so the MoE output is O(1); this keeps the correctness
    # tolerance meaningful (tiny outputs would let a broken kernel pass under a
    # fixed atol). Reductions to K are normalized by 1/sqrt(K).
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)

    # w1 gate/up: [E, 2*inter, hidden] ; w2 down: [E, hidden, inter]
    w1 = (
        torch.randn((num_experts, 2 * inter, hidden), device="cuda", dtype=torch.bfloat16)
        / (hidden**0.5)
    )
    w2 = (
        torch.randn((num_experts, hidden, inter), device="cuda", dtype=torch.bfloat16)
        / (inter**0.5)
    )
    w1_packed, w1_scale, w1_deq = _quant_pack_int4(w1, group_size)
    w2_packed, w2_scale, w2_deq = _quant_pack_int4(w2, group_size)
    del w1, w2

    # routing: top-k experts per token
    logits = torch.randn((tokens, num_experts), device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    return {
        "cfg": case,
        "module": module,
        "x": x,
        "w1": w1_packed,
        "w2": w2_packed,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        # The dequantized weights back the reference. They are kept for the
        # performance run too, because the timed invocation is validated as well.
        "w1_deq": w1_deq,
        "w2_deq": w2_deq,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "num_experts": num_experts,
        "group_size": group_size,
        "inter": inter,
    }


def _run(inputs: dict):
    return inputs["module"].fused_experts_impl(
        hidden_states=inputs["x"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_weights=inputs["topk_weights"],
        topk_ids=inputs["topk_ids"],
        activation="silu",
        use_int4_w4a16=True,
        global_num_experts=inputs["num_experts"],
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, inputs["group_size"]],
    )


def _reference(inputs: dict):
    """Dequantized torch reference for the int4 fused MoE.

    Iterates experts rather than (token, slot) pairs: every token routed to an
    expert is one batched matmul instead of a Python step. The loop this replaces
    ran M*topk times and synchronised on ``topk_ids`` each step, which is what
    made a full-shape correctness run impractical.

    The dequantized weights stay bf16 and are cast per expert, so the reference
    never materialises an fp32 copy of the whole expert bank.
    """
    torch = _torch()
    import torch.nn.functional as F

    x = inputs["x"].float()  # [M, hidden]
    w1d = inputs["w1_deq"]  # [E, 2*inter, hidden] bf16
    w2d = inputs["w2_deq"]  # [E, hidden, inter] bf16
    tw = inputs["topk_weights"].float()  # [M, topk]
    tid = inputs["topk_ids"].long()  # [M, topk]
    inter = inputs["inter"]
    M, hidden = x.shape
    topk = tid.shape[1]

    out = torch.zeros((M, hidden), device="cuda", dtype=torch.float32)
    flat_expert = tid.reshape(-1)
    flat_token = torch.arange(
        M, device=x.device
    ).unsqueeze(1).expand(M, topk).reshape(-1)
    flat_weight = tw.reshape(-1)

    order = torch.argsort(flat_expert)
    sorted_expert = flat_expert[order]
    # One boundary per expert, so slicing needs no per-pair host round trip.
    counts = torch.bincount(sorted_expert, minlength=w1d.shape[0])
    offsets = torch.cumsum(counts, dim=0).tolist()
    start = 0
    for expert, stop in enumerate(offsets):
        if stop == start:
            continue
        rows = flat_token[order[start:stop]]
        gate_up = x[rows] @ w1d[expert].float().t()  # [n, 2*inter]
        act = F.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        contrib = (act @ w2d[expert].float().t()) * flat_weight[
            order[start:stop]
        ].unsqueeze(-1)
        out.index_add_(0, rows, contrib)
        start = stop
    return out.to(torch.bfloat16)


def _assert_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(got, _reference(inputs), atol=0.02, rtol=0.02)


def _perturb_inputs(inputs: dict) -> None:
    """Refresh the activation in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input address, so writing through it
    changes what the scored kernel consumes. Only ``x`` is redrawn: the routing
    tensors are kernel inputs rather than something the kernel derives, and the
    packed weights have a dequantized twin that would have to be rebuilt in
    lockstep for no benefit here.
    """
    torch = _torch()
    torch.manual_seed(53)
    inputs["x"].normal_()


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap.

    Only ``compile`` may use this. Correctness and performance must share one
    shape, otherwise the scored path is not the validated path.
    """
    params = dict(case["params"])
    group_size = params["group_size"]
    params["tokens"] = min(params["tokens"], 16)
    params["num_experts"] = min(params["num_experts"], 8)
    params["topk"] = min(params["topk"], 2)
    # Both reductions must stay a whole number of quantization groups.
    params["hidden"] = max(group_size, min(params["hidden"], 512))
    params["inter"] = max(group_size, min(params["inter"], 128))
    return {**case, "params": params}


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against a freshly perturbed
    activation and checks the buffer it wrote, so work that the scored path skips
    cannot hide behind a correctness call that took a different branch.
    """
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    if timed.outputs is not None:
        timed.outputs.fill_(float("nan"))
    _assert_close(inputs, timed.rerun())


def run_compile() -> None:
    inputs = _make(_compile_smoke_case(CASES[0]))
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    """Check every case, rotating over the whole suite CORRECTNESS_ROUNDS times.

    A single pass only samples a kernel once, so an implementation that is wrong
    intermittently — a racy cross-workgroup barrier, a reused global scratch
    buffer — passes whenever the race happens not to fire. Rotating the suite
    also exposes state that leaks from one case into the next, which a case run
    back-to-back with itself would not surface. The suite is cheap next to
    process start-up and JIT, so the extra rounds cost only a few percent.
    """
    torch = _torch()
    for _ in range(CORRECTNESS_ROUNDS):
        for case in CASES:
            inputs = _make(case)
            got = _run(inputs)
            torch.cuda.synchronize()
            _assert_close(inputs, got)
            print("correctness PASS", case["id"])


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        _torch().cuda.synchronize()
        timed = _TimedRun()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=10,
            repetition=100,
            target_ms=1.0,
            max_graph_repeats=1000,
            timed_run=timed,
        )
        _assert_timed_outputs(inputs, timed)
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "session_id": case.get("session_id"),
            "kernel_ids": case.get("kernel_ids"),
            "gpu_pct": case.get("gpu_pct"),
            "benchmark_method": bench_meta.get("benchmark_method"),
        }
        metadata.update(
            {k: v for k, v in bench_meta.items() if k.startswith("benchmark_")}
        )
        rows.append(
            {
                "test_case_id": case["id"],
                "shape": case.get("trace_input_shapes"),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
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
