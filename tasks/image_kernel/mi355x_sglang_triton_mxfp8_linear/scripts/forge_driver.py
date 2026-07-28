#!/usr/bin/env python3
"""forge-loop measurement driver for the SGLang MXFP8 dense-GEMM task.

Target kernel (PROTECTED, optimized by forge-loop):
    ``_mxfp8_linear_kernel`` (Triton ``tl.dot_scaled``, CDNA4/gfx950) and its
    launcher ``_run_mxfp8_linear_kernel`` in
    ``sglang/kernels/ops/quantization/mxfp8_amd_gfx95.py``.

Why this file exists
--------------------
The Arena forge launcher (``agents/forge/launch_agent.py``) prefers a
task-shipped ``scripts/forge_driver.py`` and copies it VERBATIM to the run
workspace root as ``forge_driver.py``. If it is absent, the launcher generates a
generic shim that delegates to ``arena_task_adapter`` — which does NOT implement
``--profile-run`` — so forge-loop's pre-loop "task preparation" then spends an
LLM agent (minutes) authoring/repairing a profiling-capable driver every run.

Shipping this file makes forge-loop's driver preflight pass on the first check
(correctness + graph-timed bench + profiling), so task preparation is skipped
entirely and no per-run driver authoring can fail.

Contract implemented (forge-loop runs ``python forge_driver.py <args>`` and reads
only stdout — see kernel_agents.mcp_server.tools.{test,bench} and
kernel_agents.loop.task_preparer.DRIVER_CONTRACT_SPEC):

  * Correctness  ``--shape <s> --mode <smoke|stability|determinism|full>``
        prints ``allclose: True/False`` (all cases pass their own relerr tol).
        NOTE: we deliberately do NOT print an ``SNR: <db> dB`` line. This MXFP8
        GEMM has a quantization noise floor below forge's default 30 dB SNR gate,
        so emitting SNR would make even the PRISTINE kernel fail correctness. The
        task's own tolerance is relative error (``max_relerr`` per case in
        session_cases.json), exactly what ``task_runner.run_correctness`` uses;
        ``allclose`` is a valid contract fallback (test tool: SNR preferred,
        allclose otherwise).

  * Benchmark    ``--shape <s> --warmup <n> --iters <n> --bench-mode``
        prints ``case_ms: <case_id> <ms>`` for every case and a single
        ``mean_ms: <ms>`` aggregate (arithmetic mean across cases, matching
        Arena's own evaluator aggregation). Timing runs under a CUDA/HIP graph
        via ``task_runner._benchmark_cuda_graph`` (the GEMM launch is captured
        once and REPLAYED per timed iteration), so forge-loop's graph-replay
        probe is satisfied for real.

  * Profiling    ``--profile-run [--profile-case <case_id>]``
        builds ONE case's inputs and launches ONLY the target launcher
        (``_run``): a few warmups to settle Triton JIT/autotune, a couple of
        profiled launches, one synchronize, exit 0. No timing is printed.

All measurement logic is REUSED from the task's own ``scripts/task_runner.py``
(``_make`` / ``_run`` / ``_reference`` / ``_benchmark_cuda_graph`` /
``_configure``) — no reimplementation, so the driver measures exactly the same op
the Arena harness scores. This file is the correctness ORACLE and perf MEASURER;
forge-loop never edits it.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _import_task_runner():
    """Import the task's own harness (scripts/task_runner.py), robustly.

    In a real run the launcher copies this file to ``<workspace>/forge_driver.py``
    while task_runner stays at ``<workspace>/scripts/task_runner.py``; when run
    in place from the task dir both files sit in ``scripts/``. Search both.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "scripts", here, here.parent / "scripts"):
        tr = cand / "task_runner.py"
        if tr.is_file():
            spec = importlib.util.spec_from_file_location("_forge_task_runner", tr)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_forge_task_runner"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError(f"task_runner.py not found near {here}")


def _case_map(tr):
    return {c["id"]: c for c in tr.CASES}


def _case_size(case: dict) -> int:
    p = case["params"]
    return int(p["m"]) * int(p["n"]) * int(p["k"])


def _run_correctness(tr, mode: str) -> int:
    """Reuse the harness's kernel-vs-reference check; gate on relerr, not SNR."""
    torch = tr._torch()
    all_ok = True
    for case in tr.CASES:
        inputs = tr._make(case)
        got = tr._run(inputs)
        torch.cuda.synchronize()
        err = tr._relerr(got, tr._reference(inputs))
        tol = case["params"].get("max_relerr", 0.06)
        ok = err < tol
        all_ok = all_ok and ok
        # Informational only. Intentionally NOT the "SNR:" token (see module doc).
        print(f"# case {case['id']}: relerr={err:.4f} tol={tol} ok={ok}")
    print(f"allclose: {all_ok}")
    return 0


def _run_bench(tr, warmup: int, iters: int) -> int:
    """Graph-timed bench: one CUDA-graph mean per case (reuses task_runner)."""
    torch = tr._torch()
    means = []
    for case in tr.CASES:
        inputs = tr._make(case)
        tr._run(inputs)  # warm/JIT settle before capture
        torch.cuda.synchronize()
        ms, meta = tr._benchmark_cuda_graph(
            lambda i=inputs: tr._run(i),
            warmup=max(1, warmup),
            repetition=max(1, iters),
        )
        means.append(ms)
        # Per-case time (forge can round-trip the case_id to --profile-case).
        print(f"case_ms: {case['id']} {ms:.6f}")
        print(f"# bench {case['id']}: {ms:.6f} ms method={meta.get('benchmark_method')}"
              f" {meta.get('benchmark_fallback_reason', '')}".rstrip())
    # Aggregate forge uses for keep/revert = arithmetic mean across cases
    # (matches arena_task_adapter / Arena's evaluator). Label it honestly.
    agg = sum(means) / len(means)
    print(f"mean_ms: {agg:.6f}")
    return 0


def _run_profile(tr, case_id: str) -> int:
    """Kernel-only profiling for one case: warm, a few launches, sync, exit 0."""
    torch = tr._torch()
    cases = _case_map(tr)
    if case_id and case_id in cases:
        case = cases[case_id]
    else:
        # Default: the largest / dominant case (biggest M*N*K) when unspecified.
        case = max(tr.CASES, key=_case_size)
    inputs = tr._make(case)
    for _ in range(5):          # settle Triton JIT / autotune selection
        tr._run(inputs)
    torch.cuda.synchronize()
    for _ in range(3):          # profiled launches (profiler replays per group)
        tr._run(inputs)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SGLang MXFP8 dense-GEMM forge driver")
    parser.add_argument("--shape", default="default")  # task owns its shapes
    parser.add_argument("--mode", default="full")       # all modes -> task correctness
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args, _unknown = parser.parse_known_args()

    tr = _import_task_runner()
    tr._configure()  # set gfx950 arch env + make the seeded sglang copy importable

    import torch
    if not torch.cuda.is_available():
        print("error: ROCm GPU (gfx950) is required (torch.cuda.is_available() is False)",
              file=sys.stderr)
        return 1

    if args.profile_run:
        return _run_profile(tr, args.profile_case)
    if args.bench_mode:
        return _run_bench(tr, args.warmup, args.iters)
    return _run_correctness(tr, args.mode)


if __name__ == "__main__":
    sys.exit(main())
