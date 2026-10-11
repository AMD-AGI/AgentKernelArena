#!/usr/bin/env python3
"""Graph timing for the exact captured native paged-attention ABI.

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
import secrets
import statistics
import sys

from _attention_inputs import CHALLENGES, CPUInputs, IMMUTABLE, TENSORS, case_params as describe_case
from _graph_contract import contract, provenance, validate_request, mixed_error_ratio


WARMUP = 10
SAMPLES = 100


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


def verify_candidate_binding(cases, ut):
    source = hashlib.sha256((ut.parent/'source/pa_kernels.cuh').read_bytes()).hexdigest()
    launcher_digest = hashlib.sha256((ut.parent/'source/attention_candidate.py').read_bytes()).hexdigest()
    launcher = cases.current_callable()
    path = Path(launcher.__code__.co_filename)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=launcher_digest:
        raise RuntimeError('Attention launcher is not the frozen task-local wrapper')
    core = launcher.__globals__['paged_attention_ragged_core']
    if Path(core.__code__.co_filename).resolve()!=(ut/'native/pa_ragged.py').resolve():
        raise RuntimeError('Attention wrapper bypassed the task-local native build')
    core.__globals__['compile'](8,256,1,'__hip_bfloat16','__hip_bfloat16','auto','__hip_bfloat16',1)
    report=json.loads((ut.parent/'build/native_compile_report.json').read_text())
    if report['compiled_header_sha256']!=source:
        raise RuntimeError('Attention executable is not bound to the submitted device header')
    return {'launcher_source_sha256':launcher_digest,'device_source_sha256':source,
            'device_kernel':'paged_attention_ll4mi_QKV_mfma16_kernel'}


class ReplayState:
    """Preparation/validation state; all timing belongs to the shared helper."""

    def __init__(self, torch, cases, harness, args, cpu_args, tol, seed):
        self.torch, self.cases, self.harness = torch, cases, harness
        self.args = tuple(args)
        self.fixed_cpu_args = tuple(cpu_args)
        self.input_source = CPUInputs(torch, cpu_args, seed)
        self.cpu_inputs = None
        self.expected = None
        self.next_kind = "ordinary"
        self.needs_validation = False
        self.tol = tol
        self.outputs = None
        self.capture_pending = False
        self.checked_invocations = 0
        self.worst_error = 0.0
        self.worst_mixed_ratio = 0.0

    def validate(self):
        if self.outputs is None or self.capture_pending or not self.needs_validation:
            return
        if self.outputs is not self.args[0]:
            raise RuntimeError('Attention must return the supplied output tensor')
        for name,index in IMMUTABLE.items():
            expected=self.cpu_inputs if name=='query' else self.fixed_cpu_args[index]
            if not self.args[index].detach().cpu().equal(expected):
                raise RuntimeError(f"timed callable mutated {name}")
        if self.expected.device.type!='cpu':
            raise RuntimeError('Attention expected output escaped CPU ownership')
        actual=self.outputs.detach().cpu()
        if (actual.shape!=self.expected.shape or actual.dtype!=self.expected.dtype
                or not self.torch.isfinite(actual).all()):
            raise RuntimeError('Attention result changed ABI or contains nonfinite data')
        ok,error=self.harness.correct(actual,self.expected,self.tol)
        mixed=mixed_error_ratio(actual,self.expected,self.tol)
        if not ok or not math.isfinite(error) or not math.isfinite(mixed) or mixed>1:
            raise RuntimeError(f'timed graph out failed tolerance {self.tol}: {error}')
        self.worst_error=max(self.worst_error,error)
        self.worst_mixed_ratio=max(self.worst_mixed_ratio,mixed)
        self.checked_invocations += 1
        self.needs_validation = False

    def prepare(self):
        # The preceding replay has completed (or is ordered on this same stream).
        # Captured allocations are not initialized until their first replay.
        self.validate()
        self.capture_pending = False
        self.cpu_inputs, self.expected = self.input_source.next(self.next_kind)
        self.args[2].copy_(self.cpu_inputs)
        self.args[0].fill_(float('nan'))
        # The native scratch buffer is uint8 storage holding FP32 statistics
        # and BF16 partition outputs. All-ones bytes poison both as NaNs.
        self.args[1].fill_(255)
        self.needs_validation = True

    def run(self):
        self.outputs = self.cases.call({'positional':self.args,'fresh':False})
        self.capture_pending = self.torch.cuda.is_current_stream_capturing()
        return self.outputs


class TimedGraph:
    """Receive the canonical helper's exact measured graph and output tuple."""

    def _bind(self, rerun, outputs=None):
        self._rerun, self.outputs = rerun, outputs

    def rerun(self):
        return self._rerun()


def case_params(case, expected, meta):
    actual=describe_case(case['args']['positional'],meta['reference_io_sha256'])
    if actual!=expected:
        raise RuntimeError('Timed inputs differ from the complete recorded attention ABI')
    return actual


def correctness_case(torch, cases, harness, case, expected, meta, cpu_args, tol, seed):
    actual=case_params(case,expected,meta)
    state=ReplayState(torch,cases,harness,case['args']['positional'],cpu_args,tol,seed)
    results = []
    for kind in ("ordinary", *CHALLENGES):
        state.next_kind = kind
        state.prepare()
        state.run()
        state.validate()
        results.append({"kind": kind, "passed": True})
    return {"sig": f"{case['sig']}|{case['regime']}", "params": actual,
            "expected_device": "cpu", "tolerance": tol, "checked_invocations": len(results),
            "checks": results, "max_rel_err": state.worst_error,
            "max_mixed_error_ratio":state.worst_mixed_ratio}


def measure_case(torch, cases, harness, case, expected, meta, cpu_args, tol, seed):
    from _aka_benchmark import benchmark_cuda_graph_or_events_samples

    actual=case_params(case,expected,meta)
    state=ReplayState(torch,cases,harness,case['args']['positional'],cpu_args,tol,seed)
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
                       "max_mixed_error_ratio":state.worst_mixed_ratio,
                       "outputs": ['out'],
                       "inputs_varied": ['query'],
                       "restored_buffers": [],
                       "poisoned_buffers": ['out','workspace_buffer'],
                       "immutable_inputs_unchanged": True,
                       "returned_aliases_checked": True,
                       "measured_graph_validated": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=SAMPLES)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    args.out.unlink(missing_ok=True)
    if (args.warmup, args.iters) != (WARMUP, SAMPLES):
        raise ValueError("Attention requires exactly 10 warmups and 100 reported samples")
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    meta = json.loads((args.ut / "meta.json").read_text())
    harness = _load("_attention_timing_harness", args.ut / "harness_lib.py")
    cases = _load("_attention_timing_cases", args.ut / "cases.py")
    verify_frozen_source(args.ut, meta)
    policy=contract(args.ut.parent)
    pins=provenance(args.ut.parent,policy)
    if not args.correctness_only:
        if args.request is None:
            raise ValueError('Performance must use the protected task-runner request')
        pins['request']=validate_request(json.loads(args.request.read_text()),args.ut.parent,policy)
    pins['source_binding']=verify_candidate_binding(cases,args.ut)
    digest=hashlib.sha256()
    with (args.ut/meta['reference_io']).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    if digest.hexdigest()!=meta['reference_io_sha256']:
        raise RuntimeError('Original attention fixture checksum differs')
    captured=cases.load_live_case(device='cpu')
    cpu_args=tuple(captured['args']['positional'])
    values=[]
    for index,value in enumerate(cpu_args):
        if isinstance(value,torch.Tensor):
            tensor=torch.empty_strided(value.shape,value.stride(),dtype=value.dtype,device='cuda')
            if index not in (0,1):
                tensor.copy_(value)
            values.append(tensor)
        else:values.append(value)
    live=[{**{k:captured[k] for k in ('sig','regime','m')},'args':{'positional':tuple(values),'fresh':False}}]
    if len(policy['cases'])!=1:
        raise RuntimeError('Attention timing requires the one complete recorded live case')
    rows = []
    for offset,case in enumerate(live):
        expected=policy['cases'][offset]['params']
        if f"{case['sig']}|{case['regime']}"!=policy['cases'][offset]['test_case_id']:
            raise RuntimeError("timing case identity differs from the frozen live contract")
        if args.correctness_only:
            row=correctness_case(torch,cases,harness,case,expected,meta,cpu_args,float(meta['tol']),41000+offset)
            print(f"[CPU-truth:{row['sig']}] PASS ordinary and four numerical challenges", flush=True)
        else:
            row=measure_case(torch,cases,harness,case,expected,meta,cpu_args,float(meta['tol']),secrets.randbits(63))
            print(f"{row['sig']}: {row['mean_ms']:.6f} ms (cuda_graph, {SAMPLES} samples)", flush=True)
        rows.append(row)
    # Publish only after the complete case; failed runs leave no partial report.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + f".tmp-{os.getpid()}")
    payload = ({"status": "ok", "mode": "cpu_truth_correctness", "cases": rows}
               if args.correctness_only else
               {"timer": "cuda_graph", "warmup": WARMUP, "iters": SAMPLES, "cases": rows})
    payload.update(pins)
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(args.out)


if __name__ == "__main__":
    main()
