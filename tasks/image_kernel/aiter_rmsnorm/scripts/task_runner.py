#!/usr/bin/env python3
"""Task runner for repository/aiter_hip_rmsnorm.

Optimizes the AITER HIP rmsnorm / add_rmsnorm (+quant) kernel that lives in
`csrc/kernels/rmsnorm_quant_kernels.cu`. All of `aiter.rmsnorm`,
`aiter.add_rmsnorm`, `aiter.rmsnorm_quant` and `aiter.add_rmsnorm_quant` are
JIT-compiled from that single `.cu` (module `module_rmsnorm_quant`, decorated
via `@compile_ops`) and their kernel work is done by the one `__global__`
`add_rmsnorm_quant_kernel`. Editing the `.cu` invalidates the source-hash cache
and forces a recompile, so the agent's changes take effect. This runner:
  - compile:     builds the op with a small case (smoke)
  - correctness: runs the HIP op vs a torch reference (assert close)
  - performance: benchmarks the HIP op and writes build/performance_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter_hip_rmsnorm"
REPO_SUBDIR = "aiter"


def _ck_dir() -> str | None:
    """Locate Composable-Kernel headers for the JIT build.

    aiter's JIT build uses `${CK_DIR}/include` (default
    `<repo>/3rdparty/composable_kernel`). A fresh `git clone` of aiter does NOT
    ship the CK submodule include tree, so the pristine baseline build fails with
    `FileNotFoundError: .../3rdparty/composable_kernel/include` and the baseline
    (and thus speedup) is never measured. Prefer the repo submodule when present;
    otherwise fall back to the ROCm image's bundled CK headers.
    """
    if os.environ.get("CK_DIR"):
        return os.environ["CK_DIR"]
    repo_ck = _repo_root() / "3rdparty" / "composable_kernel" / "include"
    if repo_ck.is_dir():
        return None  # aiter default already works
    for base in sorted(glob.glob("/opt/rocm*"), reverse=True):
        if os.path.isdir(os.path.join(base, "include", "ck_tile")):
            return base
    return None


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
    deps (einops, jinja2, pyyaml, ...). A venv created from the wrong python
    (e.g. a bare /usr/bin/python3) silently loses torch. Instead we re-exec into
    whichever interpreter can import torch.
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
            env.setdefault("ENABLE_CK", "1")
            env.setdefault("AITER_LOG_LEVEL", "WARNING")
            ck = _ck_dir()
            if ck:
                env["CK_DIR"] = ck
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{_repo_root()}{os.pathsep}{existing}"
                if existing
                else str(_repo_root())
            )
            os.execve(str(cand), [str(cand), str(script_path), *sys.argv[1:]], env)
    # No torch found anywhere: fall through so the import raises a clear error.


def _configure_runtime() -> None:
    # AITER normally reuses an existing module without checking the source
    # files. Force a rebuild so an agent's edits are guaranteed to be compiled.
    os.environ["AITER_REBUILD"] = "1"
    os.environ.setdefault("ENABLE_CK", "1")
    os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
    ck = _ck_dir()
    if ck:
        os.environ["CK_DIR"] = ck
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
# Input construction (mirrors op_tests/test_rmsnorm2dFusedAddQuant.py run_hip).
# The no-quant HIP ops (aiter.rmsnorm / aiter.add_rmsnorm) both dispatch to the
# single __global__ add_rmsnorm_quant_kernel and match torch F.rms_norm.
# ---------------------------------------------------------------------------
def _make_case(*, m: int, n: int, dtype_str: str, add_residual: bool):
    import torch

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    device = "cuda:0"
    torch.manual_seed(0)
    torch.set_default_device(device)

    input = torch.randn((m, n), dtype=dtype)
    weight = torch.randn(n, dtype=dtype)
    residual = torch.randn((m, n), dtype=dtype) if add_residual else None

    return {
        "params": {
            "m": m,
            "n": n,
            "dtype": dtype_str,
            "add_residual": add_residual,
        },
        "input": input,
        "weight": weight,
        "residual": residual,
        "eps": 1e-5,
    }


def _run_aiter(case: dict):
    import torch
    import aiter

    inp = case["input"]
    weight = case["weight"]
    eps = case["eps"]
    out = torch.empty_like(inp)
    if case["residual"] is None:
        aiter.rmsnorm(out, inp, weight, eps)
        return out
    residual_out = torch.empty_like(inp)
    aiter.add_rmsnorm(out, inp, case["residual"], residual_out, weight, eps)
    return out


def _run_torch(case: dict):
    import torch.nn.functional as F

    inp = case["input"]
    weight = case["weight"]
    eps = case["eps"]
    if case["residual"] is None:
        norm_in = inp
    else:
        norm_in = inp + case["residual"]
    return F.rms_norm(
        input=norm_in, normalized_shape=(norm_in.shape[-1],), weight=weight, eps=eps
    )


CASES = [
    dict(m=8, n=1024, dtype_str="bfloat16", add_residual=False),
    dict(m=16, n=2048, dtype_str="float16", add_residual=True),
]

PERF_CASES = [
    ("rmsnorm_m8192_n4096", dict(m=8192, n=4096, dtype_str="bfloat16", add_residual=False)),
    ("add_rmsnorm_m4096_n8192", dict(m=4096, n=8192, dtype_str="bfloat16", add_residual=True)),
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
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        print(f"Correctness case {idx} {cfg}: PASS")


def run_performance() -> None:
    results: list[dict] = []
    for test_case_id, cfg in PERF_CASES:
        case = _make_case(**cfg)
        _run_aiter(case)  # warm build
        time_ms, bench_meta = _benchmark_cuda_graph_or_events(lambda: _run_aiter(case))
        p = case["params"]
        entry = {
            "test_case_id": test_case_id,
            "shape": [p["m"], p["n"]],
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
