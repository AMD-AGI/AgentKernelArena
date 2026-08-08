#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _import_aiter():
    import aiter

    return aiter


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


def _assert_quant_correct(inputs: dict, got) -> None:
    torch = _torch()
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


def _run_quant_correctness(inputs: dict) -> None:
    torch = _torch()

    # The scale is an output and must not depend on allocator-provided contents.
    inputs["scale"].fill_(torch.finfo(inputs["scale"].dtype).max)
    got = _run_quant(inputs)
    torch.cuda.synchronize()
    _assert_quant_correct(inputs, got)

    # Reuse the same buffers with a smaller input maximum. This catches
    # implementations that retain the previous scale across invocations.
    inputs["input"].mul_(torch.finfo(inputs["input"].dtype).eps)
    got = _run_quant(inputs)
    torch.cuda.synchronize()
    _assert_quant_correct(inputs, got)


def _perturb_quant_inputs(inputs: dict) -> None:
    """Refresh the input in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input address, so writing through it
    changes what the scored kernel consumes. ``_run_quant_correctness`` leaves the
    input scaled down by eps, so this also restores a normal magnitude.
    """
    torch = _torch()
    torch.manual_seed(31)
    inputs["input"].normal_()


def _assert_operator() -> None:
    """Guard against this runner being pointed at a different operator.

    The file was specialized to ``dynamic_per_tensor_quant``; the other builders, callers and
    references from the shared template are gone. Failing loudly beats silently
    running the wrong workload.
    """
    if OPERATOR != "dynamic_per_tensor_quant":
        raise KeyError(f"{OPERATOR}: this runner only implements dynamic_per_tensor_quant")


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    There is deliberately no correctness/performance switch here: a shape that is
    timed must also be the shape that is validated, or the scored code path can
    differ from the checked one.

    Kept as the entry point the task drivers call, alongside :func:`_run`.
    """
    _assert_operator()
    return _make_quant(case)


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against a freshly perturbed
    input and checks the buffers it wrote, so work that the scored path skips
    cannot hide behind a correctness call that took a different branch.
    """
    torch = _torch()
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_quant_inputs(inputs)
    # Both outputs are harness-owned, so poison them directly rather than going
    # through the timed handle. A kernel that skips either one keeps the poison.
    inputs["scale"].fill_(torch.finfo(inputs["scale"].dtype).max)
    inputs["output"].fill_(torch.finfo(torch.float8_e4m3fn).max)
    _assert_quant_correct(inputs, timed.rerun())


def _run(inputs: dict):
    _assert_operator()
    return _run_quant(inputs)


def run_compile() -> None:
    inputs = _make(CASES[0])
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    for case in CASES:
        _run_quant_correctness(_make(case))
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
