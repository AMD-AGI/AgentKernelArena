#!/usr/bin/env python3
"""Task runner for repository/aiter_triton_fp8_gemm.

Optimizes the AITER Triton FP8 scaled GEMM kernel that lives in
`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8.py`. The public entry
`aiter.ops.triton.gemm.basic.gemm_a8w8.gemm_a8w8` computes
`Y = (X @ W^T) * (x_scale * w_scale) + bias` where X and W are FP8 tensors and
x_scale / w_scale dequantize the accumulator back to BF16. All GEMM work is done
by the single `@triton.jit` `_gemm_a8w8_kernel` (an optional
`_gemm_a8w8_reduce_kernel` only reduces split-K partials). This maps to the hot
op `aten::_scaled_mm` (FP8 scaled matmul). Editing the kernel file changes the
Triton source that is JIT-compiled at first call, so the agent's edits take
effect. This runner:
  - compile:     builds the op with a small FP8 call (smoke)
  - correctness: runs the Triton op vs a torch dequant+matmul reference
  - performance: benchmarks the Triton op, writes build/performance_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter_triton_fp8_gemm"
REPO_SUBDIR = "aiter"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _workspace_root() / REPO_SUBDIR


def _report_root() -> Path:
    return _workspace_root() / "build"


def _has_torch(python: Path) -> bool:
    try:
        r = subprocess.run(
            [str(python), "-c", "import torch"], capture_output=True, timeout=120
        )
        return r.returncode == 0
    except Exception:
        return False


def _torch_python_candidates() -> list[Path]:
    """Interpreters likely to have torch inside the sglang/ROCm base image.

    We do NOT build a fresh venv: the base image already ships torch + aiter
    deps (triton, einops, ...). A venv created from the wrong python silently
    loses torch. Instead we re-exec into whichever interpreter can import torch.
    """
    cands: list[Path] = [Path(sys.executable)]
    for p in ("/opt/venv/bin/python3", "/opt/venv/bin/python"):
        cands.append(Path(p))
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            cands.append(Path(found))
    # Dedup by literal path. Do NOT resolve(): a venv's bin/python is a symlink
    # to the base interpreter, but only the venv path sees the venv site-packages
    # (torch). Resolving would collapse them and skip the torch-enabled venv.
    seen: set[str] = set()
    out: list[Path] = []
    for c in cands:
        rc = str(c)
        if rc not in seen:
            seen.add(rc)
            out.append(c)
    return out


def _ensure_torch_python() -> None:
    """Re-exec into an interpreter that can import torch, exactly once."""
    if os.environ.get("_KA_TORCH_PY") == "1":
        return
    try:
        import torch  # noqa: F401

        return
    except Exception:
        pass

    script_path = Path(__file__).resolve()
    for cand in _torch_python_candidates():
        if not cand.exists():
            continue
        if str(cand) == str(sys.executable):
            continue
        if _has_torch(cand):
            env = os.environ.copy()
            env["_KA_TORCH_PY"] = "1"
            env.setdefault("AITER_LOG_LEVEL", "WARNING")
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{_repo_root()}{os.pathsep}{existing}"
                if existing
                else str(_repo_root())
            )
            os.execve(str(cand), [str(cand), str(script_path), *sys.argv[1:]], env)
    # No torch found anywhere: fall through so the import raises a clear error.


def _configure_runtime() -> None:
    os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)


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


def _write_performance_report(results: list[dict]) -> None:
    report_root = _report_root()
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "performance_report.json").write_text(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Input construction (mirrors op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py
# generate_gemm_a8w8_inputs + run_torch for the FP8 path). x is (M, K) row-major
# FP8, w is (N, K) FP8 (transposed internally by gemm_a8w8), x_scale is (M, 1)
# and w_scale is (1, N) so the dequant scale is the outer product x_scale @ w_scale.
# ---------------------------------------------------------------------------
def _make_case(*, m: int, n: int, k: int, in_dtype_str: str):
    import torch
    from aiter.ops.triton.utils.types import str_to_torch_dtype

    in_dtype = str_to_torch_dtype[in_dtype_str]
    out_dtype = torch.bfloat16
    device = "cuda:0"
    torch.manual_seed(0)

    dtype_max = torch.finfo(in_dtype).max

    x = torch.randn((m, k), dtype=torch.float32, device=device)
    weight = torch.randn((n, k), dtype=torch.float32, device=device)

    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max
    x = (x / x_scale).to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max
    weight = (weight / w_scale.T).to(in_dtype)

    bias = torch.rand([1, n], dtype=torch.float32, device=device) * 10

    return {
        "params": {"m": m, "n": n, "k": k, "in_dtype": in_dtype_str},
        "x": x,
        "weight": weight,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "bias": bias,
        "out_dtype": out_dtype,
    }


def _run_aiter(case: dict):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    return gemm_a8w8(
        case["x"],
        case["weight"],
        case["x_scale"],
        case["w_scale"],
        case["bias"],
        case["out_dtype"],
    )


def _run_torch(case: dict):
    import torch
    import torch.nn.functional as F

    x = F.linear(case["x"].to(torch.float32), case["weight"].to(torch.float32))
    scale = torch.matmul(case["x_scale"], case["w_scale"])
    out = torch.mul(x, scale)
    out = out.to(case["bias"]) + case["bias"]
    return out.to(case["out_dtype"])


CASES = [
    dict(m=16, n=1024, k=1024, in_dtype_str="fp8e4m3"),
    dict(m=32, n=2048, k=512, in_dtype_str="fp8e5m2"),
]

PERF_CASES = [
    ("fp8_gemm_m4096_n4096_k4096", dict(m=4096, n=4096, k=4096, in_dtype_str="fp8e4m3")),
    ("fp8_gemm_m8192_n8192_k8192", dict(m=8192, n=8192, k=8192, in_dtype_str="fp8e4m3")),
]


def run_compile() -> None:
    case = _make_case(**CASES[0])
    _run_aiter(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch

    # FP8 (e4m3/e5m2) has a tiny mantissa, so use fp8-appropriate tolerances
    # matching the upstream op tests (atol=0.02-0.03, rtol=1e-2).
    # Also validate the exact shapes that are scored (PERF_CASES); otherwise a
    # kernel that is correct on the small shapes but wrong -- or specializing
    # invalid behavior -- on the large scored shapes would still earn a perf score.
    scored_cfgs = [cfg for _id, cfg in PERF_CASES]
    for idx, cfg in enumerate([*CASES, *scored_cfgs]):
        case = _make_case(**cfg)
        out = _run_aiter(case)
        ref = _run_torch(case)
        torch.testing.assert_close(out, ref, atol=0.03, rtol=1e-2)
        print(f"Correctness case {idx} {cfg}: PASS")


def run_performance() -> None:
    results: list[dict] = []
    for test_case_id, cfg in PERF_CASES:
        case = _make_case(**cfg)
        _run_aiter(case)  # warm JIT compile / autotune
        time_ms, bench_meta = _benchmark_cuda_graph_or_events(lambda: _run_aiter(case))
        p = case["params"]
        entry = {
            "test_case_id": test_case_id,
            "shape": [p["m"], p["n"], p["k"]],
            "execution_time_ms": time_ms,
            "metadata": p,
            "benchmark_method": bench_meta.get("benchmark_method"),
        }
        if bench_meta.get("benchmark_fallback_reason"):
            entry["benchmark_fallback_reason"] = bench_meta["benchmark_fallback_reason"]
        results.append(entry)
        print(f"{test_case_id}: {time_ms:.4f} ms [{bench_meta.get('benchmark_method')}]")
    _write_performance_report(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    _ensure_torch_python()
    _configure_runtime()

    if args.mode == "compile":
        run_compile()
    elif args.mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
