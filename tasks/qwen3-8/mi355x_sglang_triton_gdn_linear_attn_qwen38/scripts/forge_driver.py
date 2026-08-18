#!/usr/bin/env python3
"""forge-loop measurement driver for the Qwen3.8 Gated-DeltaNet linear-attention task.

Target (optimized by forge-loop): the Triton decode chain --
``fused_recurrent_gated_delta_rule_packed_decode_kernel``,
``_causal_conv1d_update_kernel``, ``_layer_norm_fwd_1pass_kernel`` -- plus the
torch split/reshape/concat around them, which is where 2.35 of the chain's 5.76
E2E points actually go.

Why this file exists
--------------------
``agents/forge/launch_agent.py:822`` prefers a task-shipped
``scripts/forge_driver.py`` and copies it VERBATIM to the workspace root;
without one it generates a shim around ``agents/forge/drivers/arena_task_adapter``.
That shim parses ``--profile-run`` with ``parse_known_args``, i.e. it silently
swallows the flag and runs correctness instead. KernelForge's
``_check_profile_contract`` only asserts ``returncode == 0``, so the shim passes
the gate without owning a profiling path. Campaigns longer than 2 hours switch
Analysis to ``profiled`` (``kernel_agents/cli.py:2088``) and actually use that
path. This driver implements the flag for real.

Contract implemented (forge-loop reads only stdout; parsers are
``mcp_server/tools/test.py:74-83`` and ``mcp_server/tools/bench.py:341-347``):

  * Correctness  ``--mode <smoke|stability|determinism|full>``
        prints ``SNR: <db> dB`` (worst case) against the harness's independent
        vectorized torch implementation, plus ``max_diff:``. The pristine chain
        measures rel_norm_err ~0.0034, i.e. ~49 dB, so it clears forge's 30 dB
        gate with ~19 dB of margin.

  * Benchmark    ``--warmup <n> --iters <n> --bench-mode``
        prints ``case_ms: <case_id> <ms>`` for every declared case plus one
        ``mean_ms:`` aggregate. Timing goes through the harness's
        ``_benchmark_cuda_graph_or_events``, so
        ``task_preparer._count_graph_replays`` observes real CUDA-graph replays.

  * Profiling    ``--profile-run [--profile-case <case_id>]``
        restores the caches, warms up, then launches ONLY the chain a few times
        and synchronizes. Prints no timing.

**This operator is stateful.** ``causal_conv1d_update`` and the recurrent kernel
update their caches in place (``gdn_triton.py`` passes ``ht=initial_state``), so
every mode restores the pristine cache snapshot before measuring; otherwise a
second measurement starts from a state the first one advanced and the numbers
drift for reasons that have nothing to do with the kernel.

All measurement logic is REUSED from ``scripts/task_runner.py``. forge-loop
never edits this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
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


def _cases(tr) -> list[dict]:
    cases = list(tr.CASES)
    if not cases:
        raise SystemExit("error: session_cases.json declares no cases")
    return cases


def _snr_db(rel_norm_err: float) -> float:
    """SNR in dB from a relative norm error: 20*log10(||ref|| / ||err||)."""
    if rel_norm_err <= 0:
        return 200.0
    return -20.0 * math.log10(rel_norm_err)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _run_correctness(tr) -> int:
    import torch

    worst_err, worst_cos, all_ok = 0.0, 1.0, True
    for case in _cases(tr):
        inputs = tr._prepare(case)
        tr._reset_state(inputs)
        got = tr._run(inputs)
        torch.cuda.synchronize()
        expected = tr._reference(inputs, steps=1)
        finite = bool(torch.isfinite(got).all())
        cos, err = tr._deviation(got, expected)

        # The caches are the point of this operator: an implementation that
        # returns the right activations but corrupts conv_state / ssm_state
        # breaks the next token, so check them too.
        ref_conv = inputs["conv_state0"].clone()
        ref_ssm = inputs["ssm_state0"].clone()
        tr._reference_step(inputs, ref_conv, ref_ssm)
        c_cos, c_err = tr._deviation(inputs["conv_state"], ref_conv)
        s_cos, s_err = tr._deviation(inputs["ssm_state"], ref_ssm)
        state_ok = c_cos > 0.999 and c_err < 0.02 and s_cos > 0.999 and s_err < 0.02

        tol = case["params"]
        ok = (
            finite
            and state_ok
            and cos > tol.get("min_cosine", 0.999)
            and err < tol.get("max_rel_norm_err", 0.03)
        )
        all_ok = all_ok and ok
        worst_err, worst_cos = max(worst_err, err), min(worst_cos, cos)
        print(f"# case {case['id']}: cos={cos:.6f} rel_norm_err={err:.5f} "
              f"conv_state_cos={c_cos:.6f} ssm_state_cos={s_cos:.6f} "
              f"finite={finite} ok={ok}")
        del inputs, expected, got, ref_conv, ref_ssm
        tr._free()

    print(f"max_diff: {worst_err:.6e}")
    print(f"SNR: {_snr_db(worst_err):.2f} dB")
    print(f"allclose: {all_ok}")
    print(f"# worst cosine {worst_cos:.6f} across {len(_cases(tr))} case(s)")
    return 0 if all_ok else 1


def _run_bench(tr, warmup: int, iters: int) -> int:
    import torch

    means: list[float] = []
    for case in _cases(tr):
        inputs = tr._prepare(case)
        tr._reset_state(inputs)
        tr._run(inputs)
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        tr._reset_state(inputs)
        exec_ms, meta = tr._benchmark_cuda_graph_or_events(
            lambda: tr._run(inputs),
            warmup=warmup,
            # One graph replay per timed iteration: this is what makes
            # _count_graph_replays see >= iters replays.
            repetition=max(1, iters),
            target_ms=bench.get("target_ms", 1.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
        )
        if not isinstance(exec_ms, (int, float)) or exec_ms <= 0:
            print(f"error: invalid timing for case {case['id']!r}: {exec_ms!r}",
                  file=sys.stderr)
            return 1
        if meta.get("benchmark_method") != "cuda_graph":
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
    """Launch ONLY the target chain, so a profiler session captures it and nothing else."""
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
        case = cases[0]

    inputs = tr._prepare(case)
    tr._reset_state(inputs)
    # Warmups settle the Triton JIT so the profiled launches contain kernel time.
    for _ in range(5):
        tr._run(inputs)
    torch.cuda.synchronize()
    tr._reset_state(inputs)
    for _ in range(3):
        tr._run(inputs)
    torch.cuda.synchronize()
    print(f"# profiled {case['id']} (bs={case['params']['batch']})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", default="full")        # all modes -> task correctness
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
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
