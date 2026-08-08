#!/usr/bin/env python3
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


def _assert_operator() -> None:
    """Guard against this runner being pointed at a different operator.

    The file was specialized to ``mhc_fused_post_pre``; the other builders,
    callers and references from the shared template are gone. Failing loudly
    beats silently running the wrong workload.
    """
    if OPERATOR != "mhc_fused_post_pre":
        raise KeyError(
            f"{OPERATOR}: this runner only implements mhc_fused_post_pre"
        )


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


def _make_mhc(case: dict) -> dict:
    torch = _torch()
    params = dict(case["params"])
    tokens = params["tokens"]
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


def _assert_mhc_close(inputs: dict, got) -> None:
    torch = _torch()
    for actual, expected in zip(got, _mhc_reference(inputs)):
        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


def _perturb_mhc_inputs(inputs: dict) -> None:
    """Refresh the data inputs in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input addresses, so writing through
    them changes what the scored kernel consumes. Fresh values stop an output
    buffer that the kernel never wrote from matching the reference by accident,
    which the caching allocator makes surprisingly likely when a recycled block
    still holds a previous output.
    """
    torch = _torch()
    torch.manual_seed(29)
    inputs["x"].normal_()
    inputs["residual"].normal_()
    inputs["post_mix"].normal_()
    inputs["comb_mix"].copy_(
        torch.softmax(torch.randn_like(inputs["comb_mix"]), dim=-1)
    )


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    There is deliberately no correctness/performance switch here: a shape that is
    timed must also be the shape that is validated, or the scored code path can
    differ from the checked one.

    Kept as the entry point the task drivers call, alongside :func:`_run`.
    """
    _assert_operator()
    return _make_mhc(case)


def _run(inputs: dict):
    _assert_operator()
    return _run_mhc(inputs)


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap.

    Only ``compile`` may use this. Correctness and performance must share one
    shape, otherwise the scored path is not the validated path.
    """
    smoke_case = {**case, "params": dict(case["params"])}
    smoke_case["params"]["tokens"] = min(case["params"]["tokens"], 64)
    return smoke_case


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against freshly perturbed
    inputs and checks the buffers it wrote, so work that the scored path skips
    cannot hide behind a correctness call that took a different branch.
    """
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")

    _perturb_mhc_inputs(inputs)
    if timed.outputs is not None:
        for tensor in timed.outputs:
            tensor.fill_(float("nan"))
    _assert_mhc_close(inputs, timed.rerun())


def run_compile() -> None:
    inputs = _make(_compile_smoke_case(CASES[0]))
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case)
        got = _run(inputs)
        torch.cuda.synchronize()
        _assert_mhc_close(inputs, got)
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
            warmup=3,
            repetition=20,
            target_ms=1.0,
            max_graph_repeats=100,
            timed_run=timed,
        )
        _assert_timed_outputs(inputs, timed)
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
