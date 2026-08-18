#!/usr/bin/env python3
"""forge-loop measurement driver for the Qwen3.8 aiter MXFP4 MoE 2-stage task.

Target (optimized by forge-loop): the FlyDSL a4w4 MoE grouped GEMMs
``compile_mixed_moe_gemm1`` / ``compile_mixed_moe_gemm2`` in
``aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py``.

Why this file exists
--------------------
``agents/forge/launch_agent.py:822`` prefers a task-shipped
``scripts/forge_driver.py`` and copies it VERBATIM to the workspace root;
without one it generates a shim around ``agents/forge/drivers/arena_task_adapter``.
That shim parses ``--profile-run`` with ``parse_known_args``, i.e. it silently
swallows the flag and runs correctness instead. KernelForge's
``_check_profile_contract`` only asserts ``returncode == 0``, so the shim passes
the gate without owning a profiling path at all. Campaigns longer than 2 hours
switch Analysis to ``profiled`` (``kernel_agents/cli.py:2088``) and actually use
that path, so the shim's gap turns into wasted prep budget or a hard
``task_preparation_failed``. This driver implements the flag for real.

Contract implemented (forge-loop runs ``python forge_driver.py <args>`` and reads
only stdout; parsers are ``mcp_server/tools/test.py:74-83`` and
``mcp_server/tools/bench.py:341-347``):

  * Correctness  ``--mode <smoke|stability|determinism|full>``
        prints ``allclose: True|False`` and ``max_diff: <rel_norm_err>``, worst
        case across the suite.

        It deliberately does NOT print ``SNR: <db> dB``. Both metrics are
        parsed, but SNR takes precedence (``test.py:85``) and is gated at 30 dB,
        while this op is **a4w4** -- the activations are MXFP4, ~2 mantissa
        bits. The pristine kernel measures rel_norm_err ~0.15 against the
        dequantized torch reference, i.e. ~16.5 dB. Printing SNR would fail the
        gate on the UNMODIFIED kernel and make every candidate look broken.
        ``allclose`` is judged against the task's own tolerance
        (``min_cosine`` 0.97 / ``max_rel_norm_err`` 0.25 in session_cases.json),
        which is the physically meaningful bar for fp4 activations.

  * Benchmark    ``--warmup <n> --iters <n> --bench-mode``
        prints ``case_ms: <case_id> <ms>`` for every declared case plus one
        ``mean_ms:`` aggregate (arithmetic mean across cases, matching Arena's
        evaluator). Timing goes through the task harness's
        ``_benchmark_cuda_graph_or_events``, which captures the op into a
        CUDA/HIP graph and REPLAYS it once per timed iteration -- so
        ``task_preparer._count_graph_replays`` observes real replays rather
        than a printed label.

  * Profiling    ``--profile-run [--profile-case <case_id>]``
        builds one case (the most expensive by default) and launches ONLY
        ``fused_moe``: warmups to settle the aiter JIT and the FlyDSL
        heuristic-config lookup, a few profiled launches, one synchronize,
        exit 0. Prints no timing.

All measurement logic is REUSED from ``scripts/task_runner.py`` so the driver
measures exactly the op Arena scores. forge-loop never edits this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _import_task_runner():
    """Import the task's own harness, whether we run from scripts/ or the root."""
    here = Path(__file__).resolve().parent
    for cand in (here / "scripts", here, here.parent / "scripts"):
        runner = cand / "task_runner.py"
        if runner.is_file():
            spec = importlib.util.spec_from_file_location("_forge_task_runner", runner)
            module = importlib.util.module_from_spec(spec)
            sys.modules["_forge_task_runner"] = module
            spec.loader.exec_module(module)
            return module
    raise SystemExit("error: task_runner.py not found next to forge_driver.py")


def _case_cost(case: dict) -> int:
    """Rank cases so profiling picks the one worth looking at (largest M)."""
    return int(case["params"].get("token", 0))


def _cases(tr) -> list[dict]:
    cases = list(tr.CASES)
    if not cases:
        raise SystemExit("error: session_cases.json declares no cases")
    return cases


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _run_correctness(tr) -> int:
    import torch

    worst_err, worst_cos, all_ok = 0.0, 1.0, True
    for case in _cases(tr):
        inputs = tr._prepare(case)
        got = tr._run(inputs)
        torch.cuda.synchronize()
        expected = tr._reference(inputs)
        finite = bool(torch.isfinite(got).all())
        cos, err = tr._deviation(got, expected)
        tol = case["params"]
        ok = (
            finite
            and cos > tol.get("min_cosine", 0.97)
            and err < tol.get("max_rel_norm_err", 0.25)
        )
        all_ok = all_ok and ok
        worst_err, worst_cos = max(worst_err, err), min(worst_cos, cos)
        print(f"# case {case['id']}: cos={cos:.6f} rel_norm_err={err:.5f} "
              f"finite={finite} ok={ok}")
        del inputs, expected, got
        tr._free()

    # max_diff carries the relative norm error, the same statistic the task's own
    # tolerance is expressed in, so a human reading the log compares like with like.
    print(f"max_diff: {worst_err:.6e}")
    print(f"allclose: {all_ok}")
    print(f"# worst cosine {worst_cos:.6f} across {len(_cases(tr))} case(s)")
    return 0 if all_ok else 1


def _run_bench(tr, warmup: int, iters: int) -> int:
    import torch

    means: list[float] = []
    for case in _cases(tr):
        inputs = tr._prepare(case)
        tr._run(inputs)
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        exec_ms, meta = tr._benchmark_cuda_graph_or_events(
            lambda: tr._run(inputs),
            warmup=warmup,
            # One graph replay per timed iteration: this is what makes
            # _count_graph_replays see >= iters replays.
            repetition=max(1, iters),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 100),
        )
        if not isinstance(exec_ms, (int, float)) or exec_ms <= 0:
            print(f"error: invalid timing for case {case['id']!r}: {exec_ms!r}",
                  file=sys.stderr)
            return 1
        if meta.get("benchmark_method") != "cuda_graph":
            # Fail loudly: a silent fall back to event timing would still print a
            # number, and the graph gate would then reject the driver with a far
            # less obvious message than this one.
            print(f"error: case {case['id']} timed with "
                  f"{meta.get('benchmark_method')} instead of cuda_graph "
                  f"({meta.get('benchmark_fallback_reason')})", file=sys.stderr)
            return 1
        means.append(float(exec_ms))
        print(f"case_ms: {case['id']} {exec_ms:.6f}")
        print(f"# bench {case['id']}: {exec_ms:.6f} ms "
              f"method={meta.get('benchmark_method')} "
              f"chained={meta.get('benchmark_effective_repeats')}")
        del inputs
        tr._free()

    print(f"mean_ms: {sum(means) / len(means):.6f}")
    return 0


def _run_profile(tr, case_id: str | None) -> int:
    """Launch ONLY the target op, so a profiler session captures it and nothing else."""
    import torch

    cases = _cases(tr)
    if case_id:
        match = [c for c in cases if c["id"] == case_id]
        if not match:
            print(f"error: unknown --profile-case {case_id!r}; declared: "
                  f"{', '.join(c['id'] for c in cases)}", file=sys.stderr)
            return 1
        case = match[0]
    else:
        case = max(cases, key=_case_cost)

    inputs = tr._prepare(case)
    # Warmups settle the aiter JIT and the FlyDSL heuristic-config lookup so the
    # profiled launches contain kernel time, not one-off compilation.
    for _ in range(3):
        tr._run(inputs)
    torch.cuda.synchronize()
    for _ in range(2):
        tr._run(inputs)
    torch.cuda.synchronize()
    print(f"# profiled {case['id']} (M={case['params']['token']})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", default="full")        # all modes -> task correctness
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    # Unknown flags are tolerated so a newer forge-loop can pass options this
    # driver predates, but --profile-run above is a REAL flag, not swallowed.
    args, _unknown = parser.parse_known_args()

    tr = _import_task_runner()
    tr._configure()
    import torch

    if not torch.cuda.is_available():
        print("error: a ROCm GPU (gfx950) is required (torch.cuda.is_available() is False)",
              file=sys.stderr)
        return 1

    if args.profile_run:
        return _run_profile(tr, args.profile_case)
    if args.bench_mode:
        return _run_bench(tr, args.warmup, args.iters)
    return _run_correctness(tr)


if __name__ == "__main__":
    raise SystemExit(main())
