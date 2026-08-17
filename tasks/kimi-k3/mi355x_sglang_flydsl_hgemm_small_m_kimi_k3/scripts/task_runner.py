#!/usr/bin/env python3
"""Image-kernel harness for the Kimi-K3 decode-side FlyDSL small-M split-K HGEMM.

Kernel covered (Hyperloom session 20260814T191522Z, rank0 of TP=8):
  ``hgemm_bf16_16x64x64x7_SPK2_W1x2x1_BLDS1_TN_AS1_0`` -- 4.55% of end-to-end GPU
  time, decode-only, 162 launches per decode step at 12.3 us each (1.99 ms of the
  25.557 ms step). It is the largest attackable non-attention, non-communication
  kernel in the session: the bigger GEMM entries (``Cijk_*``, ~15% E2E) are
  hipBLASLt Tensile code objects with no source in the image.

Where the shapes come from
--------------------------
Decode kernels run inside the CUDA graph, so the trace carries no ``Input Dims``
for them. Two independent artifacts pin M/N/K anyway:

  1. 162 launches/step = 69 + 93, exactly the per-step counts the *prefill* GEMM
     census (trace-read ``aten::mm`` dims) shows for ``[T,7168]x[7168,6144]``
     (69 = KDA layers; N = 4 x 12 heads x 128 for q/k/v/gate) and
     ``[T,1536]x[1536,7168]`` (93 = o_proj over KDA 69 + MLA 24).
  2. ``aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv`` -- the tuned table
     the running stack actually consults -- has ``gfx950 / cu_num=256 / M=8`` rows
     for both (N=6144,K=7168) and (N=7168,K=1536), each selecting ``libtype=flydsl``
     with ``flydsl_gemm7_..._t16x64x64_split_k2_block_m_warp1_block_n_warp2_...``,
     which decodes to precisely the session's kernel name (the ``gemm<N>`` suffix is
     the stage count).

Entry point
-----------
``aiter.tuned_gemm.tgemm.mm(a[M,K], b[N,K])`` -- the *real* dispatcher. The backend
and config are resolved live through ``get_GEMM_A16W16_config``, so the tuned CSV is
part of the edit surface and the per-M dispatch (M=8 -> t16x64x64/split_k2,
M=64 -> t32x64x128/split_k1) is exercised rather than hardcoded. Every case records
the resolved ``libtype``/``kernelName`` in the performance report so a run is
self-documenting.

Golden
------
``a.double() @ b.double().T``. At these sizes (max 64 x 7168 x 7168) it is a single
vectorized matmul -- no chunking, no loop.
"""
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
MODEL_CONFIG = SPEC["model_config"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    seeded = WORKSPACE / "aiter"
    if (seeded / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    else:
        sys.path.insert(0, os.environ.get("AITER_ROOT", "/sgl-workspace/aiter"))
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "aiter_jit"))
    # aiter's get_module() only rebuilds on an ARCH mismatch and never hashes the
    # sources, so any csrc the FlyDSL path pulls in would otherwise be served from a
    # stale prebuilt .so. =2 keeps the ninja build dir (unlike =1) and never touches
    # module_aiter_core, so this cannot trigger a full aiter build.
    os.environ.setdefault("AITER_REBUILD", "2")
    # FlyDSL keys its compilation cache on a fingerprint that, by default, covers
    # only flydsl's own sources -- aiter does NOT register its kernel builders via
    # extra_source_dirs, so a shared cache would serve a stale binary after the agent
    # edits splitk_hgemm.py. Pinning the cache inside the workspace makes every
    # workspace start from an empty cache; combined with the per-run staleness probe
    # in _run(), an edit can never be silently ignored.
    os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR",
                          str(WORKSPACE / "build" / "flydsl_cache"))
    # Never let a tuning sweep write into the workspace CSV mid-run.
    os.environ.setdefault("AITER_TUNE_GEMM", "0")
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


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    return torch


def _gemm_entry():
    from aiter.tuned_gemm import tgemm

    return tgemm


def _resolve_config(M: int, N: int, K: int, dtype) -> dict:
    """Ask the live dispatcher which backend/kernel it will pick, for the report.

    Best-effort: a signature change in aiter must not fail the run, only lose the
    annotation.
    """
    try:
        from aiter.tuned_gemm import get_GEMM_A16W16_config

        cfg = get_GEMM_A16W16_config(M, N, K, False, str(dtype), str(dtype))
        if isinstance(cfg, dict):
            return {"libtype": cfg.get("libtype"), "kernelName": cfg.get("kernelName"),
                    "splitK": cfg.get("splitK"), "solidx": cfg.get("solidx")}
        return {"resolved": str(cfg)}
    except Exception as exc:  # pragma: no cover - annotation only
        return {"resolve_error": f"{type(exc).__name__}: {exc}"[:200]}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    """Build one case at its scored shape (same shape for correctness and perf)."""
    torch = _torch()
    p = case["params"]
    M, N, K = int(p["M"]), int(p["N"]), int(p["K"])
    dtype = getattr(torch, p.get("dtype", "bfloat16"))

    gen = torch.Generator(device="cuda").manual_seed(int(case.get("seed", 6141)))

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", dtype=torch.float32, generator=gen)
                * scale).to(dtype)

    # TN layout: activations [M, K], weights [N, K] (the aten::linear convention the
    # dispatcher expects -- out is [M, N] = a @ b.T).
    a = rnd(M, K, scale=0.5).contiguous()
    b = rnd(N, K, scale=0.05).contiguous()

    resolved = _resolve_config(M, N, K, dtype)
    # HARD GUARD: this task only scores the FlyDSL path. The tuned CSV is inside the
    # edit surface, so without this an agent could raise its score by routing these
    # shapes to hipBLASLt / triton / opus instead of making the FlyDSL kernel faster,
    # and a mis-specified case could silently benchmark another backend. (That is not
    # hypothetical: (M=64, N=6144, K=7168) selects libtype=opus in the shipped CSV,
    # which is why that combination is deliberately NOT a case here.)
    want = case["params"].get("require_libtype", "flydsl")
    got = resolved.get("libtype")
    if got is None:
        print(f"  WARNING {case['id']}: could not resolve the dispatch "
              f"({resolved}); the libtype guard is inactive for this case")
    else:
        assert got == want, (
            case["id"],
            f"dispatch resolved to libtype={got!r}, expected {want!r}. This task "
            f"scores the FlyDSL kernel; do not route around it.",
        )
    return {"cfg": case, "M": M, "N": N, "K": K, "dtype": dtype, "a": a, "b": b,
            "resolved": resolved}


def _run(inp: dict):
    tgemm = _gemm_entry()
    return tgemm.mm(inp["a"], inp["b"])


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def _golden(inp: dict):
    """out = a @ b.T in float64. Single vectorized matmul, no chunking needed."""
    return inp["a"].double() @ inp["b"].double().T


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inp = _prepare(CASES[0])
    out = _run(inp)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)} "
          f"resolved={inp['resolved']}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inp = _prepare(case)
        out = _run(inp)
        torch.cuda.synchronize()
        ref = _golden(inp)

        assert torch.isfinite(out).all(), (case["id"], "non-finite output")
        assert tuple(out.shape) == (inp["M"], inp["N"]), (case["id"], tuple(out.shape))
        got = out.double().flatten()
        gold = ref.flatten()
        cos = torch.nn.functional.cosine_similarity(got, gold, dim=0).item()
        denom = gold.abs().max().clamp_min(1e-8)
        rel_max = ((got - gold).abs().max() / denom).item()
        p = case["params"]
        assert cos > p.get("min_cosine", 0.9995), (
            case["id"], f"cosine {cos:.7f} vs float64 golden too low"
        )
        assert rel_max < p.get("max_rel_err", 0.02), (
            case["id"], f"normalized max err {rel_max:.5f} too high"
        )
        print("correctness PASS", case["id"],
              f"[{case['phase']}] cos={cos:.7f} rel_max_err={rel_max:.5f} "
              f"lib={inp['resolved'].get('libtype')}")
        del inp, out, ref, got, gold
        torch.cuda.empty_cache()


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inp = _prepare(case)
        _run(inp)                       # settle the FlyDSL JIT / config lookup
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda i=inp: _run(i),
            warmup=bench.get("warmup", 5),
            repetition=bench.get("repetition", 50),
            target_ms=bench.get("target_ms", 1.0),
            max_graph_repeats=bench.get("max_graph_repeats", 300),
        )
        tflops = 2.0 * inp["M"] * inp["N"] * inp["K"] / (exec_ms * 1e-3) / 1e12
        metadata = {
            **case["params"],
            "phase": case.get("phase"),
            "model": case.get("model"),
            "kernel_ids": case.get("kernel_ids"),
            "exact_shape_source": case.get("exact_shape_source"),
            "expected_kernel_name": case.get("expected_kernel_name"),
            "resolved_dispatch": inp["resolved"],
            "tflops": tflops,
        }
        metadata.update({k: v for k, v in meta.items() if k.startswith("benchmark_")})
        rows.append({
            "test_case_id": case["id"],
            "shape": [inp["M"], inp["N"], inp["K"]],
            "execution_time_ms": exec_ms,
            **{k: v for k, v in meta.items() if k.startswith("benchmark_")},
            "metadata": metadata,
        })
        print(case["id"], f"{exec_ms:.6f} ms  {tflops:.2f} TFLOP/s",
              meta.get("benchmark_method"), inp["resolved"].get("libtype"),
              meta.get("benchmark_fallback_reason", ""))
        del inp
        torch.cuda.empty_cache()
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    {"compile": run_compile, "correctness": run_correctness,
     "performance": run_performance}[mode]()


if __name__ == "__main__":
    main()
