#!/usr/bin/env python3
"""Task runner for repository/aiter_triton_gemm.

Optimizes AITER's Triton A16W16 GEMM op `aiter.ops.triton.gemm.basic.gemm_a16w16`
(hot op `aten::mm`). The op computes ``Y = X @ W^T`` and dispatches to the
``@triton.jit`` kernels defined in
``aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py``:
  - ``_gemm_a16_w16_kernel``        (the main blocked / split-K matmul)
  - ``_gemm_a16w16_reduce_kernel``  (split-K partial reduction)
Editing that kernel file re-triggers Triton JIT compilation, so agent changes
take effect. This runner:
  - compile:     builds/launches the op with a small case (smoke)
  - correctness: runs the Triton op vs a torch.matmul reference (assert close)
  - performance: benchmarks the op and writes build/performance_report.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import os
from pathlib import Path

TASK_NAME = "repository/aiter_triton_gemm"
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
    deps (triton, einops, jinja2, pyyaml, ...). A venv created from the wrong
    python (e.g. a bare /usr/bin/python3) silently loses torch. Instead we
    re-exec into whichever interpreter can import torch.
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
# Input construction (mirrors op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py
# generate_gemm_a16w16_inputs with the default "TN" layout). The op computes
# Y = X @ W^T with X:(M, K) and W:(N, K); the torch reference is torch.matmul.
# ---------------------------------------------------------------------------
def _make_case(*, m: int, n: int, k: int, dtype_str: str):
    import torch

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    device = "cuda:0"
    torch.manual_seed(0)

    x = torch.randn((m, k), dtype=dtype, device=device)
    w = torch.randn((n, k), dtype=dtype, device=device)

    return {
        "params": {"m": m, "n": n, "k": k, "dtype": dtype_str},
        "x": x,
        "w": w,
        "dtype": dtype,
    }


def _run_aiter(case: dict):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    # Match output dtype to the input dtype so the torch.matmul reference is
    # comparable (the op defaults to bf16 output otherwise).
    return gemm_a16w16(case["x"], case["w"], dtype=case["dtype"])


def _run_torch(case: dict):
    import torch

    return torch.matmul(case["x"], case["w"].T)


CASES = [
    dict(m=256, n=512, k=512, dtype_str="bfloat16"),
    dict(m=128, n=256, k=512, dtype_str="float16"),
]

PERF_CASES = [
    ("gemm_m4096_n4096_k4096", dict(m=4096, n=4096, k=4096, dtype_str="bfloat16")),
    ("gemm_m8192_n8192_k1024", dict(m=8192, n=8192, k=1024, dtype_str="bfloat16")),
]


def run_compile() -> None:
    case = _make_case(**CASES[0])
    _run_aiter(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch

    # Also validate the exact shapes that are scored (PERF_CASES); otherwise a
    # kernel that is correct on the small shapes but wrong -- or specializing
    # invalid behavior -- on the large scored shapes would still earn a perf score.
    scored_cfgs = [cfg for _id, cfg in PERF_CASES]
    for idx, cfg in enumerate([*CASES, *scored_cfgs]):
        case = _make_case(**cfg)
        out = _run_aiter(case)
        ref = _run_torch(case)
        # bf16/fp16 accumulation is done in fp32 inside the kernel; a relative
        # tolerance of 2e-2 with a small absolute floor covers rounding of the
        # 16-bit inputs/outputs.
        atol = 1e-2 if cfg["dtype_str"] == "bfloat16" else 5e-3
        torch.testing.assert_close(out, ref, atol=atol, rtol=2e-2)
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
