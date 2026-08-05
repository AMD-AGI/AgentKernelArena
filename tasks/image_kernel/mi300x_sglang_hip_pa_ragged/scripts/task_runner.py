#!/usr/bin/env python3
"""Task runner for repository/aiter_hip_pa_ragged.

Optimizes the AITER HIP paged-attention kernel in the RAGGED regime. The shared
device code lives in `csrc/cpp_itfs/pa/pa_kernels.cuh`, with its entry kernel in
`csrc/cpp_itfs/pa/pa_ragged.cuh`. It is JIT-compiled from a jinja template plus
the header sources; this runner forces a fresh task-local build so source edits
take effect.

Unlike the sibling `aiter_pa_decode` task (many sequences, one query token each,
long context — the multi-partition decode reduce), this task drives short and
non-page-aligned context lengths (e.g. 4097) that stress the ragged last-page
handling and load balancing across uneven KV histories. Both tasks optimize the
same `pa_kernels.cuh`; they differ only in the benchmarked shape regime, so a
real speedup should hold across both.

This runner:
  - compile:     builds the op with a small ragged case (smoke)
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

TASK_NAME = "repository/aiter_hip_pa_ragged"
REPO_SUBDIR = "aiter"


def _ck_dir() -> str | None:
    """Locate Composable-Kernel headers for the JIT build.

    aiter's `compile_template_op` uses `${CK_DIR}/include` (default
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
    (e.g. a bare /usr/bin/python3) silently loses torch, which is exactly what
    broke the baseline measurement and the codex run. Instead we re-exec into
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
    # AITER's template JIT only checks whether lib.so exists. Force a rebuild
    # so source edits always take effect.
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
# Input construction (mirrors csrc/cpp_itfs/pa/pa_ragged_test.py Random/Shomy).
# Shapes are RAGGED-flavoured: short and non-page-aligned context lengths that
# stress the ragged last-page path and load balancing across uneven KV histories.
# ---------------------------------------------------------------------------
def _make_case(
    *,
    ctx_lens: int,
    num_seqs: int,
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype_str: str,
):
    import torch
    from einops import rearrange
    from csrc.cpp_itfs.pa import pa_ragged_test as T

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    device = "cuda:0"
    torch.manual_seed(0)
    torch.set_default_device(device)

    k_scale = v_scale = torch.tensor([1.0], dtype=torch.float32)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seq_len = ctx_lens
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs

    # Decode: a single query token per sequence.
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-1, 1)

    key_caches, value_caches = T.kv_cache_factory(
        num_blocks, block_size, 1, num_kv_heads, head_size, "auto", dtype, 0, device
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    block_tables = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=num_seqs,
    )
    seq_lens = torch.full(size=(num_seqs,), fill_value=ctx_lens, dtype=torch.int)

    def get_num_blocks(cl):
        return (cl + block_size - 1) // block_size

    def get_last_page_len(cl):
        return cl % block_size if cl % block_size > 0 else block_size

    context_lengths = [ctx_lens] * num_seqs
    num_blocks_list = [get_num_blocks(c) for c in context_lengths]
    last_page_lens = [get_last_page_len(c) for c in context_lengths]
    kv_indptr = torch.tensor([0] + num_blocks_list).cumsum(dim=0, dtype=torch.int)
    kv_last_page_lens = torch.tensor(last_page_lens, dtype=torch.int)

    elements_per_row = kv_indptr[1:] - kv_indptr[:-1]
    col_indices = torch.arange(block_tables.size(1)).expand(block_tables.size(0), -1)
    kv_page_indices = block_tables[col_indices < elements_per_row.unsqueeze(1)]

    # HND cache layout for the golden torch reference.
    key_cache_new = rearrange(key_cache, "b h d1 s d2 -> b h s (d1 d2)")
    value_cache_new = rearrange(value_cache, "b h d s -> b h s d")

    # Pre-allocate the op's output and scratch workspace ONCE (mirroring the setup
    # inside pa_ragged_test.run_aiter). The timed callable then becomes a pure
    # kernel launch — no per-call allocation — which is CUDA-graph capturable.
    _PARTITION_SIZE_ROCM = 256
    assert _PARTITION_SIZE_ROCM % block_size == 0
    max_num_partitions = (max_seq_len + _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM
    output = torch.empty_like(query)
    nbytes_per_qo_elem = torch.finfo(output.dtype).bits // 8
    workspace_buffer = torch.empty(
        (num_seqs * num_query_heads * max_num_partitions * head_size) * nbytes_per_qo_elem
        + 2 * (num_seqs * num_query_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=output.device,
    )

    return {
        "params": {
            "ctx_lens": ctx_lens,
            "num_seqs": num_seqs,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "block_size": block_size,
            "dtype": dtype_str,
        },
        "query": query,
        "key_cache_new": key_cache_new.contiguous(),
        "value_cache_new": value_cache_new.contiguous(),
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "kv_indptr": kv_indptr,
        "kv_page_indices": kv_page_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "max_seq_len": max_seq_len,
        "num_kv_heads": num_kv_heads,
        "scale": scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "num_queries_per_kv": num_queries_per_kv,
        "output": output,
        "workspace_buffer": workspace_buffer,
        "max_num_partitions": max_num_partitions,
    }


def _run_aiter(case: dict):
    # Call the raw op directly. pa_ragged_test.run_aiter is a @perftest-decorated
    # BENCHMARK wrapper (warmup + 101 profiled iters + torch.cuda.synchronize +
    # empty_cache + trace post-processing); timing it measured the harness, not the
    # kernel (~100x inflated) and was not CUDA-graph capturable. The underlying op
    # is a single launch and is capturable.
    from csrc.cpp_itfs.pa.pa_ragged import paged_attention_ragged

    p = case["params"]
    paged_attention_ragged(
        case["output"],
        case["workspace_buffer"],
        case["query"],
        case["key_cache_new"],
        case["value_cache_new"],
        case["scale"],
        case["kv_indptr"],
        case["kv_page_indices"],
        case["kv_last_page_lens"],
        p["block_size"],
        case["max_num_partitions"],
        None,  # alibi_slopes
        "auto",  # kv_cache_dtype
        "HND",  # kv_cache_layout
        0.0,  # logits_soft_cap
        case["k_scale"],
        case["v_scale"],
        None,  # fp8_out_scale
    )
    return case["output"]


def _run_torch(case: dict):
    from csrc.cpp_itfs.pa import pa_ragged_test as T

    out, _ = T.run_torch_new(
        case["query"],
        case["key_cache_new"],
        case["value_cache_new"],
        case["block_tables"],
        case["seq_lens"],
        case["max_seq_len"],
        "auto",
        case["num_kv_heads"],
        case["scale"],
        None,  # alibi_slopes
        0.0,  # logits_soft_cap
        case["k_scale"],
        case["v_scale"],
        case["num_queries_per_kv"],
    )
    return out


# Correctness cases: ragged-shaped (short / uneven context, GQA) kept modest in
# total tokens because the golden torch reference loops over every (seq, token)
# pair in Python.
CASES = [
    dict(ctx_lens=128, num_seqs=8, num_heads=(8, 1), head_size=128, block_size=16, dtype_str="bfloat16"),
    dict(ctx_lens=26, num_seqs=16, num_heads=(4, 2), head_size=64, block_size=16, dtype_str="float16"),
]

# Performance cases: ragged regime -- short and non-page-aligned (4097) contexts.
PERF_CASES = [
    ("pa_ragged_ctx128_s128", dict(ctx_lens=128, num_seqs=128, num_heads=(8, 1), head_size=128, block_size=16, dtype_str="bfloat16")),
    ("pa_ragged_ctx4097_s128", dict(ctx_lens=4097, num_seqs=128, num_heads=(8, 1), head_size=128, block_size=16, dtype_str="bfloat16")),
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
            "shape": [p["num_seqs"], p["num_query_heads"], p["head_size"], p["ctx_lens"]],
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
