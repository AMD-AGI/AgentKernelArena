#!/usr/bin/env python3
"""Task runner for repository/aiter_hip_mha_batch_prefill.

Optimizes the AITER HIP CK batch-prefill MHA kernel whose torch entry lives in
`csrc/py_itfs_ck/mha_batch_prefill_kernels.cu`. That `.cu` implements
`aiter::torch_itfs::mha_batch_prefill` (+ `get_ck_fmha_batch_prefill_args`),
which packs the arguments and dispatches to the CK `aiter::mha_batch_prefill`
fmha kernel. The python op `aiter.mha_batch_prefill_func` is JIT-compiled from
that `.cu` plus the CK fmha codegen (module `module_mha_batch_prefill`, decorated
via `@compile_ops`). This runner forces a fresh task-local JIT build so edits to
the `.cu` take effect. It:
  - compile:     builds the op with a small case (smoke)
  - correctness: runs the HIP op vs a torch SDPA reference (assert close)
  - performance: benchmarks the HIP op and writes build/performance_report.json

The build needs the Composable-Kernel fmha codegen shipped in the CK submodule
(`3rdparty/composable_kernel/example/ck_tile/01_fmha/generate.py`), which is
seeded with the AITER repository from the task image.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter_hip_mha_batch_prefill"
REPO_SUBDIR = "aiter"


def _ck_dir() -> str | None:
    """Locate Composable-Kernel headers/codegen for the JIT build.

    aiter's batch_prefill build uses `${CK_DIR}/include` for headers and
    `${CK_DIR}/example/ck_tile/01_fmha/generate.py` for fmha codegen (default
    `<repo>/3rdparty/composable_kernel`). A fresh `git clone` of aiter does NOT
    ship the CK submodule, so the pristine baseline build fails and the baseline
    (and thus speedup) is never measured. The task's `post_clone_install` fetches
    the submodule; prefer it when present. The ROCm image only bundles CK
    *headers* (no fmha codegen), so it is a last resort that is insufficient for
    batch_prefill on its own -- but we still expose it for parity with the other
    aiter tasks in case a future ROCm ships the example tree.
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
    broke the baseline measurement. Instead we re-exec into whichever interpreter
    can import torch.
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
# Input construction (mirrors op_tests/test_batch_prefill.py: uniform-length
# paged KV cache in the SGLANG 4D linear layout [num_pages, page_size, h_kv, d]).
# All test cases use bf16 + causal + logits soft-cap 30 and no bias / lse /
# dropout / descale / sink. The nonzero soft cap avoids a documented ROCm 7.2
# gfx950 compiler bug in the causal + zero-soft-cap kernel variant.
# ---------------------------------------------------------------------------
def _make_case(
    *,
    batch_size: int,
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    dtype_str: str,
    logits_soft_cap: float,
):
    import torch

    assert kv_len >= qo_len, "causal batch prefill needs kv_len >= qo_len"
    assert num_qo_heads % num_kv_heads == 0
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    device = "cuda:0"
    torch.manual_seed(0)

    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size

    # Q: flat [total_q, h_q, d]; per-sequence slices via cu_seqlens_q.
    total_q = batch_size * qo_len
    query = torch.empty(total_q, num_qo_heads, head_size, dtype=dtype, device=device)
    query.uniform_(-1, 1)

    # Paged KV cache, 4D SGLANG linear layout [num_pages, page_size, h_kv, d].
    kv_cache = torch.empty(
        total_pages, page_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    key_cache = kv_cache.clone().uniform_(-1, 1)
    value_cache = kv_cache.clone().uniform_(-1, 1)

    cu_seqlens_q = torch.arange(
        0, total_q + 1, qo_len, dtype=torch.int32, device=device
    )
    kv_indptr = torch.arange(
        0, total_pages + 1, pages_per_seq, dtype=torch.int32, device=device
    )
    # Page table: each sequence owns a contiguous, disjoint block of pages.
    kv_page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    # +256 padding: the kernel may speculatively read up to one bn0 tile past the
    # last valid page index before the bounds check; pad with 0 to keep reads
    # in-bounds (masked out, never affects output).
    kv_page_indices = torch.nn.functional.pad(kv_page_indices, (0, 256), value=0)
    last_page_len = ((kv_len - 1) % page_size) + 1
    kv_last_page_lens = torch.full(
        (batch_size,), last_page_len, dtype=torch.int32, device=device
    )

    return {
        "params": {
            "batch_size": batch_size,
            "qo_len": qo_len,
            "kv_len": kv_len,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "page_size": page_size,
            "dtype": dtype_str,
            "logits_soft_cap": logits_soft_cap,
        },
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cu_seqlens_q": cu_seqlens_q,
        "kv_indptr": kv_indptr,
        "kv_page_indices": kv_page_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "pages_per_seq": pages_per_seq,
        "max_seqlen_q": qo_len,
        "max_seqlen_k": kv_len,
        "scale": float(1.0 / math.sqrt(head_size)),
        "logits_soft_cap": logits_soft_cap,
    }


def _run_aiter(case: dict):
    import aiter

    out = aiter.mha_batch_prefill_func(
        case["query"],
        case["key_cache"],
        case["value_cache"],
        case["cu_seqlens_q"],
        case["kv_indptr"],
        case["kv_page_indices"],
        case["max_seqlen_q"],
        case["max_seqlen_k"],
        softmax_scale=case["scale"],
        logits_soft_cap=case["logits_soft_cap"],
        causal=True,
        kv_last_page_lens=case["kv_last_page_lens"],
    )
    return out


def _run_torch(case: dict):
    """Bottom-right aligned causal attention over the gathered paged KV cache."""
    import torch

    p = case["params"]
    b = p["batch_size"]
    qo_len = p["qo_len"]
    kv_len = p["kv_len"]
    h_q = p["num_qo_heads"]
    h_kv = p["num_kv_heads"]
    d = p["head_size"]
    ratio = h_q // h_kv
    scale = case["scale"]
    logits_soft_cap = case["logits_soft_cap"]

    q = case["query"]
    key_cache = case["key_cache"]
    value_cache = case["value_cache"]
    pages_per_seq = case["pages_per_seq"]

    out = torch.empty_like(q)
    for i in range(b):
        qi = q[i * qo_len : (i + 1) * qo_len]  # [Sq, h_q, d]
        page_start = i * pages_per_seq
        ki = (
            key_cache[page_start : page_start + pages_per_seq]
            .reshape(-1, h_kv, d)[:kv_len]
            .float()
        )  # [Sk, h_kv, d]
        vi = value_cache[page_start : page_start + pages_per_seq].reshape(-1, h_kv, d)[
            :kv_len
        ].float()
        if ratio > 1:
            ki = ki.repeat_interleave(ratio, dim=1)
            vi = vi.repeat_interleave(ratio, dim=1)

        # [h_q, Sq, Sk]
        attn = scale * torch.einsum("qhd,khd->hqk", qi.float(), ki)
        if logits_soft_cap > 0:
            attn = logits_soft_cap * torch.tanh(attn / logits_soft_cap)
        # Bottom-right aligned causal mask: query row r (abs pos kv_len-qo_len+r)
        # attends key col c iff c <= (kv_len - qo_len) + r.
        row = torch.arange(qo_len, device=q.device).unsqueeze(1)
        col = torch.arange(kv_len, device=q.device).unsqueeze(0)
        mask = col > (kv_len - qo_len) + row  # True => disallowed
        attn.masked_fill_(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        oi = torch.einsum("hqk,khd->qhd", attn, vi)  # [Sq, h_q, d]
        out[i * qo_len : (i + 1) * qo_len] = oi.to(q.dtype)
    return out


CASES = [
    dict(batch_size=2, qo_len=64, kv_len=64, num_qo_heads=8, num_kv_heads=1, head_size=128, page_size=16, dtype_str="bfloat16", logits_soft_cap=30.0),
    dict(batch_size=1, qo_len=48, kv_len=80, num_qo_heads=4, num_kv_heads=2, head_size=128, page_size=16, dtype_str="bfloat16", logits_soft_cap=30.0),
]

PERF_CASES = [
    ("batch_prefill_b4_q512_kv512", dict(batch_size=4, qo_len=512, kv_len=512, num_qo_heads=8, num_kv_heads=1, head_size=128, page_size=16, dtype_str="bfloat16", logits_soft_cap=30.0)),
    ("batch_prefill_b2_q2048_kv2048", dict(batch_size=2, qo_len=2048, kv_len=2048, num_qo_heads=8, num_kv_heads=1, head_size=128, page_size=16, dtype_str="bfloat16", logits_soft_cap=30.0)),
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
            "shape": [
                p["batch_size"],
                p["num_qo_heads"],
                p["head_size"],
                p["qo_len"],
                p["kv_len"],
            ],
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
