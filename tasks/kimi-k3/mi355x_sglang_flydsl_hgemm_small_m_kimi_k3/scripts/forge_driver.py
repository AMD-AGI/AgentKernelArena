#!/usr/bin/env python3
"""forge-loop measurement driver for the Kimi-K3 FlyDSL small-M HGEMM task.

Targets (PROTECTED, optimized by forge-loop):
    ``hgemm_bf16_16x64x64x7_SPK2_W1x2x1_BLDS1_TN_AS1_0``, built by
    ``aiter/ops/flydsl/kernels/splitk_hgemm.py`` and dispatched through
    ``aiter.tuned_gemm.tgemm.mm``.

Why this file exists
--------------------
The Arena forge launcher (``agents/forge/launch_agent.py``) prefers a task-shipped
``scripts/forge_driver.py`` and copies it VERBATIM to the workspace root. Without
it the launcher generates a generic shim that delegates to ``arena_task_adapter``,
which does NOT implement ``--profile-run``; forge-loop then burns an LLM agent
authoring a profiling-capable driver before every run. Shipping this file makes the
driver preflight pass on the first check, so task preparation is skipped entirely.

Contract implemented (forge-loop runs ``python forge_driver.py <args>`` and reads
only stdout):

  * Correctness  ``--mode <smoke|stability|determinism|full>``
        prints ``SNR: <db> dB`` -- the worst case's signal-to-noise ratio against
        the harness's independent, vectorized float64 golden.

  * Benchmark    ``--warmup <n> --iters <n> --bench-mode``
        prints ``case_ms: <case_id> <ms>`` per case plus one ``mean_ms: <ms>``
        aggregate (arithmetic mean across cases, matching Arena's own evaluator).
        Timing goes through ``task_runner._benchmark_cuda_graph_or_events``, i.e.
        the call is captured once into a HIP graph and REPLAYED per timed
        iteration, so forge-loop's graph-replay probe is satisfied for real.

  * Profiling    ``--profile-run``
        builds the largest case's inputs and launches ONLY the target entry point:
        a few warmups to settle the Triton JIT, a couple of profiled launches, one
        synchronize, exit 0. No timing is printed.

All measurement logic is REUSED from ``scripts/task_runner.py`` (``_configure`` /
``_prepare`` / ``_run`` / ``_golden`` / ``_benchmark_cuda_graph_or_events``), so the
driver measures exactly the same op Arena scores. forge-loop never edits it.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path


def _import_task_runner():
    """Import the task's own harness, whether we run from scripts/ or the root.

    In a real run the launcher copies this file to ``<workspace>/forge_driver.py``
    while task_runner stays at ``<workspace>/scripts/task_runner.py``; run in place
    from the task dir, both sit in ``scripts/``. Search both.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "scripts", here, here.parent / "scripts"):
        runner = cand / "task_runner.py"
        if runner.is_file():
            spec = importlib.util.spec_from_file_location("_forge_task_runner", runner)
            module = importlib.util.module_from_spec(spec)
            sys.modules["_forge_task_runner"] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(f"task_runner.py not found near {here}")


def _case_cost(case: dict) -> int:
    """GEMM work: M * N * K."""
    p = case["params"]
    return int(p["M"]) * int(p["N"]) * int(p["K"])


def _run_correctness(tr) -> int:
    """Per-case SNR against the float64 golden; report the worst case."""
    import torch

    worst_db = math.inf
    for case in tr.CASES:
        inp = tr._prepare(case)
        out = tr._run(inp)
        torch.cuda.synchronize()
        ref = tr._golden(inp)
        got = out.double().flatten()
        gold = ref.flatten()
        noise = (got - gold).norm().item()
        signal = gold.norm().item()
        db = 200.0 if noise <= 0 else 20.0 * math.log10(signal / noise)
        worst_db = min(worst_db, db)
        print(f"# case {case['id']}: SNR={db:.2f} dB finite={bool(torch.isfinite(out).all())}")
        del inp, out, ref, got, gold
        torch.cuda.empty_cache()
    print(f"SNR: {worst_db:.2f} dB")
    return 0


def _run_bench(tr, warmup: int, iters: int) -> int:
    """Graph-timed bench: one HIP-graph mean per case (reuses task_runner)."""
    import torch

    results = []
    for case in tr.CASES:
        inp = tr._prepare(case)
        tr._run(inp)                   # settle the Triton JIT before capture
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        ms, meta = tr._benchmark_cuda_graph_or_events(
            lambda i=inp: tr._run(i),
            warmup=max(1, warmup),
            repetition=max(1, iters),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
        )
        if not math.isfinite(ms) or ms <= 0:
            print(f"error: invalid timing for case {case['id']!r}: {ms!r}", file=sys.stderr)
            return 1
        results.append((case["id"], ms, meta))
        del inp
        torch.cuda.empty_cache()
    if len(results) != len(tr.CASES) or not results:
        print("error: benchmark did not produce every task case", file=sys.stderr)
        return 1
    for case_id, ms, meta in results:
        print(f"case_ms: {case_id} {ms:.6f}")
        print(f"# bench {case_id}: {ms:.6f} ms method={meta.get('benchmark_method')}"
              f" {meta.get('benchmark_fallback_reason', '')}".rstrip())
    means = [ms for _, ms, _ in results]
    print(f"mean_ms: {sum(means) / len(means):.6f}")
    return 0


def _run_profile(tr) -> int:
    """Kernel-only profiling for one case: warm, a few launches, sync, exit 0."""
    import torch

    case = max(tr.CASES, key=_case_cost)
    inp = tr._prepare(case)
    for _ in range(3):                 # settle Triton JIT / autotune selection
        tr._run(inp)
    torch.cuda.synchronize()
    for _ in range(3):                 # profiled launches
        tr._run(inp)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimi-K3 FlyDSL small-M HGEMM forge driver")
    parser.add_argument("--mode", default="full")       # all modes -> task correctness
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    tr = _import_task_runner()
    tr._configure()   # gfx950 env + workspace-seeded aiter first on sys.path

    import torch
    if not torch.cuda.is_available():
        print("error: a ROCm GPU (gfx950) is required (torch.cuda.is_available() is False)",
              file=sys.stderr)
        return 1

    if args.profile_run:
        return _run_profile(tr)
    if args.bench_mode:
        return _run_bench(tr, args.warmup, args.iters)
    return _run_correctness(tr)


if __name__ == "__main__":
    sys.exit(main())
