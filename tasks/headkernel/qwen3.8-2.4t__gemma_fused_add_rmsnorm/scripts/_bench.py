#!/usr/bin/env python3
"""Graph timing for the two frozen RMSNorm live shapes.

The workspace-materialized canonical helper owns capture, events and sampling.
This adapter supplies new CPU-generated inputs and poisons both graph outputs
outside timing, then checks each completed invocation against CPU-owned truth.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

from _rmsnorm_inputs import CHALLENGES, CPUInputs


WARMUP = 10
SAMPLES = 100
INPUTS = ("x", "residual", "weight")
PARAM_FIELDS = ("regime", "m", "n", "dtype", "eps", "x_shape", "x_stride",
                "residual_shape", "residual_stride", "weight_shape", "weight_stride")


def _load(name, path):
    # Explicit loader also supports the frozen baseline's .py.orig suffix.
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def verify_frozen_source(ut, meta):
    provenance = meta["source_provenance"]
    path = ut / provenance["baseline_ref"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != provenance["source_sha256"]:
        raise RuntimeError("frozen native baseline source hash mismatch")
    return digest


class ReplayState:
    """Preparation/validation state; all timing belongs to the shared helper."""

    def __init__(self, torch, cases, harness, args, tol, seed):
        self.torch, self.cases, self.harness = torch, cases, harness
        self.args = dict(args, verify_inputs=False)
        self.input_source = CPUInputs(torch, args, seed)
        self.cpu_inputs = None
        self.expected = None
        self.next_kind = "ordinary"
        self.needs_validation = False
        self.tol = tol
        self.outputs = None
        self.capture_pending = False
        self.checked_invocations = 0
        self.worst_error = 0.0

    def validate(self):
        if self.outputs is None or self.capture_pending or not self.needs_validation:
            return
        self.cases._validate_outputs(self.outputs, *(self.args[name] for name in INPUTS))
        for name in INPUTS:
            if not self.args[name].detach().cpu().equal(self.cpu_inputs[name]):
                raise RuntimeError(f"timed callable mutated {name}")
        for name, value, expected in zip(("normed", "pre_norm_sum"), self.outputs, self.expected):
            if expected.device.type != "cpu":
                raise RuntimeError("expected output escaped CPU ownership")
            ok, error = self.harness.correct(value.detach().cpu(), expected, self.tol)
            if not ok or not math.isfinite(error):
                raise RuntimeError(f"timed graph {name} failed tolerance {self.tol}: {error}")
            self.worst_error = max(self.worst_error, error)
        self.checked_invocations += 1
        self.needs_validation = False

    def prepare(self):
        # The preceding replay has completed (or is ordered on this same stream).
        # Captured allocations are not initialized until their first replay.
        self.validate()
        self.capture_pending = False
        self.cpu_inputs, self.expected = self.input_source.next(self.next_kind)
        for name in INPUTS:
            self.args[name].copy_(self.cpu_inputs[name])
        if self.outputs is not None:
            for value in self.outputs:
                value.fill_(float("nan"))
        self.needs_validation = True

    def run(self):
        self.outputs = self.cases.call(self.args)
        self.capture_pending = self.torch.cuda.is_current_stream_capturing()
        return self.outputs


class TimedGraph:
    """Receive the canonical helper's exact measured graph and output tuple."""

    def _bind(self, rerun, outputs=None):
        self._rerun, self.outputs = rerun, outputs

    def rerun(self):
        return self._rerun()


def case_params(case, spec):
    actual = {"regime": case["regime"], "m": int(case["args"]["x"].shape[0]),
              "n": int(case["args"]["x"].shape[1]),
              "dtype": str(case["args"]["x"].dtype), "eps": float(case["args"]["eps"])}
    for name in INPUTS:
        actual[name + "_shape"] = list(case["args"][name].shape)
        actual[name + "_stride"] = list(case["args"][name].stride())
    if actual != {key: spec[key] for key in PARAM_FIELDS}:
        raise RuntimeError("timed inputs differ from the frozen shape/dtype/stride/eps contract")
    return actual


def correctness_case(torch, cases, harness, case, spec, tol, seed):
    actual = case_params(case, spec)
    state = ReplayState(torch, cases, harness, case["args"], tol, seed)
    results = []
    for kind in ("ordinary", *CHALLENGES):
        state.next_kind = kind
        state.prepare()
        state.run()
        state.validate()
        results.append({"kind": kind, "passed": True})
    return {"sig": f"{case['sig']}|{case['regime']}", "params": actual,
            "expected_device": "cpu", "tolerance": tol, "checked_invocations": len(results),
            "checks": results, "max_rel_err": state.worst_error}


def measure_case(torch, cases, harness, case, spec, tol, seed):
    from _aka_benchmark import benchmark_cuda_graph_or_events_samples

    actual = case_params(case, spec)
    state = ReplayState(torch, cases, harness, case["args"], tol, seed)
    timed = TimedGraph()
    samples, metadata = benchmark_cuda_graph_or_events_samples(
        state.run, warmup=WARMUP, repetition=SAMPLES,
        prepare_fn=state.prepare, timed_run=timed,
    )
    if (metadata.get("benchmark_method") != "cuda_graph"
            or metadata.get("benchmark_warmup") != WARMUP
            or metadata.get("benchmark_samples") != SAMPLES
            or metadata.get("benchmark_effective_repeats") != 1
            or "benchmark_fallback_reason" in metadata
            or len(samples) != SAMPLES
            or any(not math.isfinite(v) or v <= 0 for v in samples)):
        raise RuntimeError("canonical helper did not return the required graph measurements")
    # prepare() checks the last measured replay before resetting it. rerun()
    # then validates the exact measured graph once more, outside the sample set.
    result = timed.rerun()
    if result is not state.outputs or result is not timed.outputs:
        raise RuntimeError("validation is not bound to the measured graph outputs")
    state.validate()
    # 10 warmups, 2 estimate replays, 1 final prime, 100 samples and 1 rerun.
    if state.checked_invocations != WARMUP + SAMPLES + 4:
        raise RuntimeError("not every completed warmup/replay was validated")
    timing_checked = state.checked_invocations
    challenge_results = []
    for kind in CHALLENGES:
        state.next_kind = kind
        if timed.rerun() is not state.outputs:
            raise RuntimeError("challenge is not bound to the measured graph")
        state.validate()
        challenge_results.append({"kind": kind, "passed": True})
    if state.checked_invocations != timing_checked + len(CHALLENGES):
        raise RuntimeError("not every post-timing correctness challenge was validated")
    return {
        "sig": f"{case['sig']}|{case['regime']}",
        "params": actual,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "samples_ms": samples,
        **metadata,
        "validation": {"checked_invocations": state.checked_invocations,
                       "timing_checked_invocations": timing_checked,
                       "fresh_input_sets": state.input_source.generated,
                       "input_seed": seed,
                       "expected_device": "cpu",
                       "fresh_inputs_per_replay": True,
                       "correctness_challenges": challenge_results,
                       "max_rel_err": state.worst_error, "tolerance": tol,
                       "outputs": ["normed", "pre_norm_sum"],
                       "inputs_varied": list(INPUTS),
                       "outputs_poisoned_before_replay": True,
                       "measured_graph_validated": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=SAMPLES)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    args.out.unlink(missing_ok=True)
    if (args.warmup, args.iters) != (WARMUP, SAMPLES):
        raise ValueError("RMSNorm requires exactly 10 warmups and 100 reported samples")
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    meta = json.loads((args.ut / "meta.json").read_text())
    harness = _load("_rmsnorm_timing_harness", args.ut / "harness_lib.py")
    cases = _load("_rmsnorm_timing_cases", args.ut / "cases.py")
    verify_frozen_source(args.ut, meta)
    specs = meta["workload"]["cases"]
    live = cases.timing_cases(harness, meta)
    if len(live) != 2 or len(specs) != 2:
        raise RuntimeError("RMSNorm timing requires both frozen live cases")
    rows = []
    for offset, (case, spec) in enumerate(zip(live, specs)):
        if (case["sig"], case["regime"], case["m"]) != (spec["sig"], spec["regime"], spec["m"]):
            raise RuntimeError("timing case identity differs from the frozen live contract")
        if args.correctness_only:
            row = correctness_case(torch, cases, harness, case, spec, float(meta["tol"]), 41000 + offset)
            print(f"[CPU-truth:{row['sig']}] PASS ordinary and four numerical challenges", flush=True)
        else:
            row = measure_case(torch, cases, harness, case, spec, float(meta["tol"]), 31000 + offset)
            print(f"{row['sig']}: {row['mean_ms']:.6f} ms (cuda_graph, {SAMPLES} samples)", flush=True)
        rows.append(row)
    # Publish only after both complete; failed runs cannot leave a partial report.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + f".tmp-{os.getpid()}")
    payload = ({"status": "ok", "mode": "cpu_truth_correctness", "cases": rows}
               if args.correctness_only else
               {"timer": "cuda_graph", "warmup": WARMUP, "iters": SAMPLES, "cases": rows})
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(args.out)


if __name__ == "__main__":
    main()
