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


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(fn, repetition):
    import time
    import torch

    repetition = max(1, int(repetition))
    if not torch.cuda.is_available():
        times_ms = []
        for _ in range(repetition):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
        return times_ms

    times_ms = []
    for _ in range(repetition):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    return times_ms


def _benchmark_cuda_graph_or_events(
    fn,
    warmup=10,
    repetition=100,
    target_ms=1.0,
    n_retries=5,
    estimate_reps=5,
    max_graph_repeats=1000,
    use_cuda_graph=True,
    fallback_reason=None,
):
    import torch

    # A captured graph whose measured per-iteration time falls below this floor is
    # treated as empty (it recorded no device work) and rejected in favour of
    # per-launch event timing, so a graph that silently captured nothing is never
    # reported as a fabricated ~0 ms cuda_graph result. Real kernels measure far
    # above this floor; an empty-graph replay measures ~1e-5 ms.
    empty_graph_floor_ms = 1e-4

    for _ in range(max(0, int(warmup))):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    max_graph_repeats = max(1, int(max_graph_repeats))
    metadata = {
        "benchmark_target_ms": float(target_ms),
        "benchmark_samples": int(repetition),
        "benchmark_max_repeats": int(max_graph_repeats),
    }

    if not torch.cuda.is_available():
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cpu_timer_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_unavailable",
        })
        return sum(times) / len(times), metadata

    if not use_cuda_graph:
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_graph_disabled",
        })
        return sum(times) / len(times), metadata

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            estimate_reps = max(1, int(estimate_reps))
            estimate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(estimate_graph):
                for _ in range(estimate_reps):
                    fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            estimate_graph.replay()
            end_event.record(stream)
            torch.cuda.synchronize()

            estimate_ms = start_event.elapsed_time(end_event) / estimate_reps
            if estimate_ms == 0:
                n_repeat = max_graph_repeats
            else:
                n_repeat = min(max_graph_repeats, max(1, int(float(target_ms) / estimate_ms)))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(n_repeat):
                    fn()
            torch.cuda.synchronize()

            retry_times = []
            for _ in range(max(1, int(repetition))):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                torch.cuda.synchronize()
                retry_times.append(start_event.elapsed_time(end_event) / n_repeat)

        graph_mean = sum(retry_times) / len(retry_times)
        if graph_mean < empty_graph_floor_ms:
            times = _measure_cuda_event_fallback(fn, repetition)
            metadata.update({
                "benchmark_method": "cuda_event_fallback",
                "benchmark_effective_repeats": int(repetition),
                "benchmark_fallback_reason": fallback_reason or "empty_cuda_graph_capture",
            })
            return sum(times) / len(times), metadata

        metadata.update({
            "benchmark_method": "cuda_graph",
            "benchmark_effective_repeats": int(n_repeat),
        })
        return graph_mean, metadata
    except Exception as exc:
        # Isolate the aborted capture before re-measuring so the fallback timing is
        # not polluted by the failed attempt (a mid-capture failure can leave the
        # first few launches abnormally slow).
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for _ in range(min(3, max(1, int(warmup)))):
            fn()
        torch.cuda.synchronize()
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}",
        })
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


def _make(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    p = dict(case["params"])
    if correctness:
        tokens = min(p["tokens"], 16)
        num_experts = min(p["num_experts"], 8)
        topk = min(p["topk"], 2)
        hidden = min(p["hidden"], 512)
        inter = min(p["inter"], 128)
    else:
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
        "w1_deq": w1_deq if correctness else None,
        "w2_deq": w2_deq if correctness else None,
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
    torch = _torch()
    import torch.nn.functional as F

    x = inputs["x"].float()  # [M, hidden]
    w1d = inputs["w1_deq"].float()  # [E, 2*inter, hidden]
    w2d = inputs["w2_deq"].float()  # [E, hidden, inter]
    tw = inputs["topk_weights"].float()  # [M, topk]
    tid = inputs["topk_ids"]  # [M, topk]
    inter = inputs["inter"]
    M, hidden = x.shape
    out = torch.zeros((M, hidden), device="cuda", dtype=torch.float32)
    for m in range(M):
        acc = torch.zeros((hidden,), device="cuda", dtype=torch.float32)
        for j in range(tid.shape[1]):
            e = int(tid[m, j].item())
            gu = x[m] @ w1d[e].t()  # [2*inter]
            gate = gu[:inter]
            up = gu[inter:]
            act = F.silu(gate) * up  # [inter]
            acc = acc + float(tw[m, j].item()) * (act @ w2d[e].t())
        out[m] = acc
    return out.to(torch.bfloat16)


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
        torch.testing.assert_close(
            got, _reference(inputs), atol=0.02, rtol=0.02
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
            warmup=10,
            repetition=100,
            target_ms=1.0,
            max_graph_repeats=1000,
        )
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
