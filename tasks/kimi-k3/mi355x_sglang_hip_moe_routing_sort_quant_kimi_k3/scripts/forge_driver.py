#!/usr/bin/env python3
"""forge-loop measurement driver for the Kimi-K3 MoE routing / sorting / MX-quant task.

Targets (PROTECTED, optimized by forge-loop):
    ``aiter::grouped_topk_kernel``            (csrc/kernels/topk_softmax_kernels_group.cu)
    ``aiter::opus_moe_sorting_entry``         (csrc/include/moe_sorting_opus.h)
    ``aiter::fused_mx_quant_moe_sort_kernel`` (csrc/kernels/quant_kernels.cu)

Why this file exists
--------------------
The Arena forge launcher prefers a task-shipped ``scripts/forge_driver.py`` and
copies it VERBATIM to the workspace root. Without it the launcher generates a
generic shim that delegates to ``arena_task_adapter``, which does NOT implement
``--profile-run``; forge-loop then burns an LLM agent authoring a profiling-capable
driver before every run. Shipping this file makes the driver preflight pass on the
first check.

Contract implemented (forge-loop runs ``python forge_driver.py <args>`` and reads
only stdout):

  * Correctness  ``--mode <smoke|stability|determinism|full>``
        Runs the harness's full per-stage validation (top-k set + weights, the
        ordering-agnostic sort semantics, and MX dequant error) and additionally
        prints ``SNR: <db> dB`` computed on the top-k weights against the
        vectorized torch reference. A failed invariant raises and exits non-zero,
        which forge reads as a correctness regression.

  * Benchmark    ``--warmup <n> --iters <n> --bench-mode``
        prints ``case_ms: <case_id> <ms>`` per case plus one ``mean_ms: <ms>``.
        Timing goes through ``task_runner._benchmark_cuda_graph_or_events``: the
        chain is captured once into a HIP graph and REPLAYED per timed iteration.

  * Profiling    ``--profile-run``
        builds the largest case and launches only the target chain: warmups, a few
        profiled launches, one synchronize, exit 0.

All measurement logic is REUSED from ``scripts/task_runner.py``. forge-loop never
edits this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path


def _import_task_runner():
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
    p = case["params"]
    return int(p["num_tokens"]) * int(p["num_experts"])


def _run_correctness(tr) -> int:
    """Full per-stage validation plus a top-k-weight SNR for forge's gate."""
    import torch

    worst_db = math.inf
    for case in tr.CASES:
        inp = tr._prepare(case)
        tr._run(inp)
        torch.cuda.synchronize()

        notes = []
        if "topk" in inp["stages"]:
            notes.append(tr._check_topk(inp, case))
        if "sorting" in inp["stages"]:
            notes.append(tr._check_sorting(inp, case))
        if "quant" in inp["stages"]:
            notes.append(tr._check_quant(inp, case))

        # SNR on the routing weights, computed against the kernel's OWN chosen ids.
        # Aligning two independent top-k selections by expert id is not valid here:
        # with 896 experts the 16th/17th biased scores tie within fp32 noise, so the
        # two selections legitimately differ at the boundary and an id-aligned diff
        # would compare unrelated experts. _check_topk() above already validated the
        # selection itself (min(biased[chosen]) >= max(biased[rest]) - eps).
        score = torch.sigmoid(inp["gating"].float())
        ids = inp["topk_ids"].long()
        w_ref = torch.gather(score, 1, ids)
        if inp["need_renorm"]:
            w_ref = w_ref / w_ref.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        w_ref = w_ref * inp["routed_scaling_factor"]
        got = inp["topk_weights"].double().flatten()
        gold = w_ref.double().flatten()
        noise = (got - gold).norm().item()
        signal = gold.norm().item()
        db = 200.0 if noise <= 0 else 20.0 * math.log10(signal / noise)
        worst_db = min(worst_db, db)
        print(f"# case {case['id']}: SNR={db:.2f} dB {' '.join(notes)}")
        del inp
        torch.cuda.empty_cache()
    print(f"SNR: {worst_db:.2f} dB")
    return 0


def _run_bench(tr, warmup: int, iters: int) -> int:
    import torch

    results = []
    for case in tr.CASES:
        inp = tr._prepare(case)
        tr._run(inp)
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
    import torch

    case = max(tr.CASES, key=_case_cost)
    inp = tr._prepare(case)
    for _ in range(3):
        tr._run(inp)
    torch.cuda.synchronize()
    for _ in range(3):
        tr._run(inp)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimi-K3 MoE routing/sort/quant forge driver")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
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
