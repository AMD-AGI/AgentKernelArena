#!/usr/bin/env python3
"""KernelForge measurement driver for the Arena image_kernel MI355X tasks.

forge-loop treats this file as a black box invoked as ``python forge_driver.py
<args>`` and talks to it only through stdout. It implements the three modes of
that contract on top of the task's own harness (``scripts/task_runner.py``), so
the shapes, dtypes and reference implementations stay the single source of truth:

  * Correctness  ``--shape <s> --mode <smoke|stability|determinism|full>``
    -> prints ``SNR: <db> dB`` and ``allclose: True/False``. ``smoke`` checks
       only the first case (fast enough for forge's short preflight window);
       every other mode checks all cases in ``session_cases.json``.

  * Benchmark    ``--shape <s> --warmup <n> --iters <n> --bench-mode``
    -> prints ``wall_ms:`` samples plus one ``case_ms: <case-id> <ms>`` per
       selected case, timed under CUDA/HIP graph replay via ``graph_harness.py``.
       ``--shape default`` benches only the first case (forge preflight contract);
       a concrete case id benches that case alone.

  * Profiling    ``--profile-run [--profile-case <id>]``
    -> runs only the target kernel for the selected case; no reference, no
       timing, so a profiler sees just the dispatches under optimization.

Before any mode runs, the driver performs a one-time JIT compile warmup
(``task_runner.run_compile()``) so heavy aiter module builds finish outside
forge's short preflight timeouts (120s correctness / 300s bench).

The driver is the correctness ORACLE and the perf MEASURER, so forge never edits
it. Benchmarking reuses ``task_runner``'s ``_make``/``_run`` dispatch rather than
its ``run_performance`` reporting path, because forge needs per-case wall times
under a real graph replay rather than an aggregated JSON report.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# The launcher copies this file from ``scripts/`` to the workspace root, so
# resolve the workspace from either location and expose the task harness.
_HERE = Path(__file__).resolve().parent
_WORKSPACE = _HERE.parent if _HERE.name == "scripts" else _HERE
sys.path.insert(0, str(_WORKSPACE))
sys.path.insert(0, str(_WORKSPACE / "scripts"))

import task_runner as tr  # noqa: E402

# Sets GPU_ARCHS / AITER_* and chdirs into the workspace. Must run before torch
# and aiter are imported, exactly as task_runner.main() does.
tr._configure()

import torch  # noqa: E402  -- must follow _configure()

from graph_harness import cuda_graph_bench  # noqa: E402

_COMPILE_WARMED = False


def _ensure_compile_warmup() -> None:
    """Run task_runner compile smoke once per process before timed modes."""
    global _COMPILE_WARMED
    if _COMPILE_WARMED:
        return
    tr.run_compile()
    torch.cuda.synchronize()
    _COMPILE_WARMED = True

# Per-operator tolerances, kept identical to task_runner.run_correctness() so
# this driver and Arena's own scoring agree on what "correct" means.
_TOLERANCES = {
    "unified_attention": (0.08, 0.08),
    "a8w8_blockscale_gemm": (0.15, 0.12),
    "dynamic_per_tensor_quant": (0.25, 0.15),
    "mhc_fused_post_pre": (0.08, 0.08),
    "mla_decode": (0.08, 0.08),
    "ck_moe_2stage": (0.06, 0.06),
    "cktile_moe_2stage": (0.06, 0.06),
    "mxfp8_linear": (0.06, 0.06),
    "mxfp8_grouped_gemm": (0.15, 0.12),
}


def _tolerance() -> tuple[float, float]:
    return _TOLERANCES.get(tr.OPERATOR, (0.08, 0.08))


def _cases_for(mode: str) -> list[dict]:
    """Smoke checks one case; every other mode checks the full set."""
    return tr.CASES[:1] if mode == "smoke" else list(tr.CASES)


def _cases_for_bench(shape: str) -> list[dict]:
    """Forge preflight/iteration passes ``--shape default`` for one case only."""
    if not shape or shape == "default":
        return tr.CASES[:1]
    for case in tr.CASES:
        if str(case["id"]) == shape:
            return [case]
    return tr.CASES[:1]


def _find_case(case_id: str) -> dict:
    if not case_id:
        return tr.CASES[0]
    for case in tr.CASES:
        if str(case["id"]) == case_id:
            return case
    raise ValueError(f"unknown profile case: {case_id}")


def _flatten(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return []


def _snr_db(reference: torch.Tensor, test: torch.Tensor) -> float:
    reference = reference.float()
    test = test.float()
    signal = torch.mean(reference * reference).item()
    noise = torch.mean((test - reference) ** 2).item()
    if noise <= 0.0:
        return 100.0
    if signal <= 0.0:
        return 0.0
    return 10.0 * math.log10(signal / noise)


def _correctness_pair(inputs: dict, ret) -> tuple[list, list]:
    """Mirror task_runner.run_correctness(): (actual, expected) tensor lists.

    Each operator exposes its result differently — some return it, MLA writes
    into a preallocated output, and the quantizer is scored on the dequantized
    value — so the comparison target is selected per operator instead of assuming
    the return value is the thing to check.
    """
    op = tr.OPERATOR
    if op == "unified_attention":
        return [ret], [tr._attention_reference(inputs)]
    if op == "a8w8_blockscale_gemm":
        return [ret], [tr._gemm_reference(inputs)]
    if op == "dynamic_per_tensor_quant":
        return [ret.float() * inputs["scale"]], [inputs["input"].float()]
    if op == "mhc_fused_post_pre":
        return _flatten(ret), _flatten(tr._mhc_reference(inputs))
    if op == "mla_decode":
        return [inputs["output"]], [tr._mla_reference(inputs)]
    if op in ("mxfp8_linear", "mxfp8_grouped_gemm"):
        return [ret], [tr._reference(inputs)]
    return [ret], [tr._moe_reference(inputs)]


def _bench_outputs(inputs: dict, ret) -> list[torch.Tensor]:
    """Every tensor a replay must rewrite, for the graph-capture guard.

    Preallocated outputs (MLA's ``output``, the quantizer's ``output``/``scale``)
    live in ``inputs``; operators that allocate their result do so inside the
    captured graph, where the returned tensor is the graph's own static buffer.
    """
    tensors: list[torch.Tensor] = []
    for key in ("output", "scale"):
        value = inputs.get(key)
        if isinstance(value, torch.Tensor):
            tensors.append(value)
    for tensor in _flatten(ret):
        if not any(tensor is seen for seen in tensors):
            tensors.append(tensor)
    return tensors


def _matches(got: torch.Tensor, ref: torch.Tensor) -> bool:
    """Loose equality used only to prove a replay recomputed the output."""
    if got.shape != ref.shape:
        return False
    a = got.float()
    b = ref.float()
    if torch.equal(a, b):
        return True
    scale = b.abs().max().clamp_min(1e-12)
    return bool(((a - b).abs().max() / scale).item() < 2e-2)


def _run_correctness(mode: str) -> int:
    torch_mod = tr._torch()
    atol, rtol = _tolerance()
    snrs: list[float] = []
    all_close = True

    for case in _cases_for(mode):
        inputs = tr._make(case, correctness=True)
        ret = tr._run(inputs)
        torch_mod.cuda.synchronize()
        actual, expected = _correctness_pair(inputs, ret)
        for got, ref in zip(actual, expected):
            snrs.append(_snr_db(ref, got))
            if not torch_mod.allclose(
                got.float(), ref.float(), atol=atol, rtol=rtol
            ):
                all_close = False

    # Report the weakest case: forge gates on the worst observed accuracy, so
    # averaging would let one bad shape hide behind several good ones.
    print(f"SNR: {min(snrs) if snrs else 0.0:.2f} dB")
    print(f"allclose: {all_close}")
    return 0


def _run_bench(shape: str, warmup: int, iters: int) -> int:
    for case in _cases_for_bench(shape):
        inputs = tr._make(case, correctness=False)
        captured: dict = {}

        def step(inp=inputs, sink=captured) -> None:
            sink["ret"] = tr._run(inp)

        # Eager reference captured BEFORE the graph so dirty/verify can prove the
        # replay recomputed the result instead of returning stale memory.
        step()
        torch.cuda.synchronize()
        reference = [t.detach().clone() for t in _bench_outputs(inputs, captured.get("ret"))]
        provable = any(bool(t.float().abs().max().item() > 0.0) for t in reference)

        def dirty(sink=captured, inp=inputs) -> None:
            for tensor in _bench_outputs(inp, sink.get("ret")):
                tensor.zero_()

        def verify(sink=captured, inp=inputs, ref=reference) -> bool:
            produced = _bench_outputs(inp, sink.get("ret"))
            if len(produced) != len(ref):
                return False
            return all(_matches(got, exp) for got, exp in zip(produced, ref))

        result = cuda_graph_bench(
            step,
            warmup=warmup,
            iters=iters,
            # An all-zero reference cannot distinguish a real replay from a
            # no-op, so the guard is only armed when it is meaningful.
            dirty=dirty if provable else None,
            verify=verify if provable else None,
        )

        # Informational: not part of the wall_ms/case_ms contract.
        print(f"# bench mode: {result['mode']} ({case['id']})")
        for sample in result["times_ms"]:
            print(f"wall_ms: {sample:.6f}")
        times = sorted(result["times_ms"])
        median = times[len(times) // 2] if times else 0.0
        print(f"case_ms: {case['id']} {median:.6f}")
    return 0


def _run_profile(case_id: str) -> int:
    """Warm the target kernel, then expose only its dispatches to the profiler."""
    inputs = tr._make(_find_case(case_id), correctness=False)
    for _ in range(3):
        tr._run(inputs)
    torch.cuda.synchronize()
    for _ in range(3):
        tr._run(inputs)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{tr.OPERATOR} task driver")
    parser.add_argument("--shape", default="default", help="default=first case; or a case id")
    parser.add_argument("--mode", default="full", help="smoke|stability|determinism|full")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args, _unknown = parser.parse_known_args()

    if not torch.cuda.is_available():
        print("error: no GPU available (torch.cuda.is_available() is False)")
        return 1

    _ensure_compile_warmup()

    if args.profile_run:
        try:
            return _run_profile(args.profile_case)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.bench_mode:
        return _run_bench(args.shape, args.warmup, args.iters)
    return _run_correctness(args.mode)


if __name__ == "__main__":
    sys.exit(main())
