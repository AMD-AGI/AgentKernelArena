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


def _make_attention(
    case: dict,
    *,
    ctx_len_override: int | None = None,
    expected_path: str | None = None,
) -> dict:
    """Build the attention case at its scored ``ctx_len``.

    ``ctx_len_override`` exists only so correctness can additionally exercise the
    2d dispatch, which the scored ``ctx_len`` does not reach. It is never used to
    shrink the scored shape itself.
    """
    torch = _torch()
    params = dict(case["params"])
    ctx_len = (
        ctx_len_override if ctx_len_override is not None else params["ctx_len"]
    )
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
        "expected_attention_path": expected_path,
    }


def _attention_correctness_inputs(case: dict) -> list[tuple[str, dict]]:
    full_ctx_len = int(case["params"]["ctx_len"])
    return [
        (
            "2d",
            _make_attention(
                case,
                ctx_len_override=min(full_ctx_len, 128),
                expected_path="2d",
            ),
        ),
        (
            "3d",
            _make_attention(
                case,
                ctx_len_override=full_ctx_len,
                expected_path="3d",
            ),
        ),
    ]


class _KernelLaunchRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.called = False

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def recorded_launch(*args, **kwargs):
            self.called = True
            return launch(*args, **kwargs)

        return recorded_launch


def _run_attention(inputs: dict):
    import aiter.ops.triton.attention.unified_attention as attention

    expected_path = inputs.get("expected_attention_path")
    kernel_names = (
        "kernel_unified_attention_2d",
        "kernel_unified_attention_3d",
        "reduce_segments",
    )
    originals = {}
    recorders = {}
    if expected_path is not None:
        for name in kernel_names:
            originals[name] = getattr(attention, name)
            recorders[name] = _KernelLaunchRecorder(originals[name])
            setattr(attention, name, recorders[name])

    try:
        attention.unified_attention(
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
    finally:
        for name, kernel in originals.items():
            setattr(attention, name, kernel)

    if expected_path == "2d":
        assert recorders["kernel_unified_attention_2d"].called
        assert not recorders["kernel_unified_attention_3d"].called
        assert not recorders["reduce_segments"].called
    elif expected_path == "3d":
        assert not recorders["kernel_unified_attention_2d"].called
        assert recorders["kernel_unified_attention_3d"].called
        assert recorders["reduce_segments"].called
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


def _assert_operator() -> None:
    """Guard against this runner being pointed at a different operator.

    The file was specialized to ``unified_attention``; the other builders, callers and
    references from the shared template are gone. Failing loudly beats silently
    running the wrong workload.
    """
    if OPERATOR != "unified_attention":
        raise KeyError(f"{OPERATOR}: this runner only implements unified_attention")


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    There is deliberately no correctness/performance switch here: a shape that is
    timed must also be the shape that is validated, or the scored code path can
    differ from the checked one.

    Kept as the entry point the task drivers call, alongside :func:`_run`.
    """
    _assert_operator()
    return _make_attention(case)


def _assert_attention_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(
        got, _attention_reference(inputs), atol=0.08, rtol=0.08
    )


def _perturb_attention_inputs(inputs: dict) -> None:
    """Refresh the data inputs in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input addresses, so writing through
    them changes what the scored kernel consumes. Fresh values stop an output
    buffer that the kernel never wrote from matching the reference by accident.
    """
    torch = _torch()
    torch.manual_seed(41)
    inputs["query"].normal_()
    # key/value may be fp8 views of one bf16 draw, so rebuild them the same way
    # _make_attention did instead of writing noise straight into the fp8 buffers.
    kv = torch.randn(
        inputs["key"].shape, device="cuda", dtype=torch.bfloat16
    )
    inputs["key"].copy_(kv.to(inputs["key"].dtype))
    inputs["value"].copy_((kv * 0.7).to(inputs["value"].dtype))


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against freshly perturbed
    inputs and checks the buffer it wrote, so work that the scored path skips
    cannot hide behind a correctness call that took a different branch.
    """
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_attention_inputs(inputs)
    # The output is harness-owned, so poison it directly; a kernel that stops
    # writing keeps the poison instead of a plausible stale result.
    inputs["output"].fill_(float("nan"))
    _assert_attention_close(inputs, timed.rerun())


def _run(inputs: dict):
    _assert_operator()
    return _run_attention(inputs)


def run_compile() -> None:
    if OPERATOR == "unified_attention":
        for _, inputs in _attention_correctness_inputs(CASES[0]):
            _run_attention(inputs)
    else:
        inputs = _make(CASES[0])
        _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        for path, inputs in _attention_correctness_inputs(case):
            got = _run_attention(inputs)
            torch.cuda.synchronize()
            _assert_attention_close(inputs, got)
            print("correctness PASS", case["id"], f"path={path}")


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
