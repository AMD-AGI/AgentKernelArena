#!/usr/bin/env python3
"""Task runner for repository/sglang_hip_store_cache.

Optimizes the SGLang HIP KV-cache store kernel that lives in
`python/sglang/jit_kernel/csrc/elementwise/kvcache.cuh` (device kernel
`store_kvcache` + warp copy helper `copy_kv_warp`, exposed to Python as
`sglang::store_cache`). The kernel is JIT-compiled on demand by
`tvm_ffi.cpp.load_inline` (see `sglang.jit_kernel.kvcache._jit_kvcache_module`),
keyed by source content, so editing `kvcache.cuh` forces a recompile and the
agent's changes take effect. This runner:
  - compile:     builds the op with a small case (smoke)
  - correctness: runs the HIP op vs a torch scatter reference (assert close)
  - performance: benchmarks the HIP op and writes build/performance_report.json

Unlike the aiter tasks, the sglang JIT build is header-only and only pulls in
`sgl_kernel` headers shipped under `python/sglang/jit_kernel/include`; there is
no Composable-Kernel dependency, so no CK_DIR fallback is required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/sglang_hip_store_cache"
REPO_SUBDIR = "sglang"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _workspace_root() / REPO_SUBDIR


def _repo_python_root() -> Path:
    """The importable root of the sglang package (repo lives under python/)."""
    return _repo_root() / "python"


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

    We do NOT build a fresh venv: the base image already ships torch + the
    sglang deps (tvm_ffi, ...). A venv created from a bare /usr/bin/python3
    silently loses torch. Instead we re-exec into whichever interpreter can
    import torch.
    """
    cands: list[Path] = [Path(sys.executable)]
    for p in ("/opt/venv/bin/python3", "/opt/venv/bin/python"):
        cands.append(Path(p))
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            cands.append(Path(found))
    # Dedup by literal path. Do NOT resolve(): a venv's bin/python is a symlink
    # to the base interpreter, but only the venv path sees the venv
    # site-packages (torch). Resolving would collapse them and skip the
    # torch-enabled venv.
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
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{_repo_python_root()}{os.pathsep}{existing}"
                if existing
                else str(_repo_python_root())
            )
            os.execve(str(cand), [str(cand), str(script_path), *sys.argv[1:]], env)
    # No torch found anywhere: fall through so the import raises a clear error.


def _configure_runtime() -> None:
    # Ensure the (possibly edited) local checkout wins over any installed copy
    # so edits to kvcache.cuh are the sources that get JIT-compiled.
    repo_python = _repo_python_root()
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))
    os.chdir(_repo_root())


def _benchmark_ms(fn, warmup: int = 10, rep: int = 100) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.mean(samples)


def _write_performance_report(results: list[dict]) -> None:
    report_root = _report_root()
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "performance_report.json").write_text(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Input construction (mirrors python/sglang/jit_kernel/tests/test_store_cache.py).
# store_cache scatters rows of k/v into k_cache/v_cache at the given indices:
#   k_cache[indices] = k ; v_cache[indices] = v
# ---------------------------------------------------------------------------
def _make_case(
    *,
    batch_size: int,
    element_dim: int,
    cache_size: int,
    dtype_str: str,
    indices_dtype: str,
):
    import torch

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]
    idx_dtype = {"int32": torch.int32, "int64": torch.int64}[indices_dtype]
    device = "cuda"
    torch.manual_seed(0)

    k = torch.randn(batch_size, element_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, element_dim, dtype=dtype, device=device)
    k_cache = torch.randn(cache_size, element_dim, dtype=dtype, device=device)
    v_cache = torch.randn(cache_size, element_dim, dtype=dtype, device=device)
    indices = torch.randperm(cache_size, device=device)[:batch_size].to(idx_dtype)

    return {
        "params": {
            "batch_size": batch_size,
            "element_dim": element_dim,
            "cache_size": cache_size,
            "dtype": dtype_str,
            "indices_dtype": indices_dtype,
            "row_bytes": element_dim * torch.tensor([], dtype=dtype).element_size(),
        },
        "k": k,
        "v": v,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "indices": indices,
    }


def _run_sglang(case: dict):
    """Run the SGLang HIP store_cache kernel in place on fresh cache copies."""
    from sglang.jit_kernel.kvcache import store_cache

    k_cache = case["k_cache"].clone()
    v_cache = case["v_cache"].clone()
    store_cache(case["k"], case["v"], k_cache, v_cache, case["indices"])
    return k_cache, v_cache


def _run_torch(case: dict):
    """Golden reference: scatter/copy rows into cloned caches."""
    k_cache = case["k_cache"].clone()
    v_cache = case["v_cache"].clone()
    idx = case["indices"].long()
    k_cache[idx] = case["k"]
    v_cache[idx] = case["v"]
    return k_cache, v_cache


CASES = [
    dict(batch_size=8, element_dim=128, cache_size=4096, dtype_str="bfloat16", indices_dtype="int64"),
    dict(batch_size=16, element_dim=64, cache_size=4096, dtype_str="float16", indices_dtype="int32"),
]

PERF_CASES = [
    ("store_cache_bs128_d512", dict(batch_size=128, element_dim=512, cache_size=65536, dtype_str="bfloat16", indices_dtype="int64")),
    ("store_cache_bs4096_d1024", dict(batch_size=4096, element_dim=1024, cache_size=131072, dtype_str="bfloat16", indices_dtype="int64")),
]


def run_compile() -> None:
    from sglang.jit_kernel.kvcache import can_use_store_cache

    case = _make_case(**CASES[0])
    assert can_use_store_cache(case["params"]["row_bytes"])
    _run_sglang(case)  # forces JIT build + one small invocation
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch

    # Also validate the exact shapes that are scored (PERF_CASES); otherwise a
    # kernel that is correct on the small shapes but wrong -- or specializing
    # invalid behavior -- on the large scored shapes would still earn a perf score.
    scored_cfgs = [cfg for _id, cfg in PERF_CASES]
    for idx, cfg in enumerate([*CASES, *scored_cfgs]):
        case = _make_case(**cfg)
        k_out, v_out = _run_sglang(case)
        k_ref, v_ref = _run_torch(case)
        torch.testing.assert_close(k_out, k_ref)
        torch.testing.assert_close(v_out, v_ref)
        print(f"Correctness case {idx} {cfg}: PASS")


def run_performance() -> None:
    results: list[dict] = []
    from sglang.jit_kernel.kvcache import store_cache

    for test_case_id, cfg in PERF_CASES:
        case = _make_case(**cfg)
        # Time the kernel in place on the pre-allocated caches (repeatedly
        # writing the same indices is valid) so timing excludes cache-clone
        # allocation/copy overhead.
        def _call(c=case):
            store_cache(c["k"], c["v"], c["k_cache"], c["v_cache"], c["indices"])

        _call()  # warm build
        time_ms = _benchmark_ms(_call)
        p = case["params"]
        results.append(
            {
                "test_case_id": test_case_id,
                "shape": [p["batch_size"], p["element_dim"], p["cache_size"]],
                "execution_time_ms": time_ms,
                "metadata": p,
            }
        )
        print(f"{test_case_id}: {time_ms:.4f} ms")
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
