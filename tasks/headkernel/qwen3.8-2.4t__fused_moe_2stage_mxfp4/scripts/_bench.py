#!/usr/bin/env python3
"""Checked graph timing for the complete captured native MoE workload."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import secrets
import statistics
import sys

# The controller uses -I; add only this protected scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _routing_contract import (CHECKED, PREPARED, PROTOCOL, SAMPLES, WARMUP,
                               effective_execution_identity, fingerprint, require, specs, strict_json,
                               validate_payload, validate_request)
from _moe_runtime import NativeRuntime, ReplayState, cpu_snapshot


class TimedGraph:
    def _bind(self, rerun, outputs=None):
        self._rerun, self.outputs = rerun, outputs

    def rerun(self):
        return self._rerun()


def expect_rejection(function, description):
    try:
        function()
    except ValueError:
        return
    raise ValueError(description + ' was accepted')


def graph_correctness(state):
    torch = state.torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Warm this explicit stream before capture for stream-local native workspaces.
    with torch.cuda.stream(stream):
        state.prepare()
        state.run()
        state.validate()
        state.prepare()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            state.run()
        # A skipped replay must leave poisoned outputs and fail the same check.
        state.prepare()
        expect_rejection(state.validate, 'no-op graph replay')
        state.needs_validation = False
        state.prepare()
        graph.replay()
        state.outputs.reshape(-1)[0] += 1024
        expect_rejection(state.validate, 'corrupted graph output')
        state.needs_validation = False
        for _ in range(2):
            state.prepare()
            graph.replay()
            state.validate()
    torch.cuda.synchronize()


def engagement_and_smoke(state):
    torch, runtime = state.torch, state.runtime
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        state.prepare()
        state.run()
        torch.cuda.synchronize()
    state.validate()
    names = sorted({str(event.key) for event in prof.key_averages()
                    if float(getattr(event, 'self_device_time_total', 0.0)
                             or getattr(event, 'device_time_total', 0.0)) > 0})
    matched = {stage: [name for name in names if stage in name.lower()] for stage in ('moe1', 'moe2')}
    require(all(matched.values()), 'both native MoE device stages must engage')
    timings = {}
    # Retain the original two-repeat eager timing smoke, excluded from scoring.
    for leg, route in (('baseline', 'captured'), ('candidate', state.route)):
        runtime.call(state.args, route)
        torch.cuda.synchronize()
        values = []
        for _ in range(2):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            runtime.call(state.args, route)
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        require(all(math.isfinite(value) and value > 0 for value in values), 'invalid native timing smoke')
        timings[leg + '_ms'] = statistics.mean(values)
    return {'required_fragments': ['moe1', 'moe2'], 'matched': matched,
            'smoke_timing': dict(timings, purpose='smoke_only')}


def correctness_case(runtime, spec, route, seed):
    state = ReplayState(runtime, spec, route, seed)
    state.use_captured = state.check_captured = True
    state.prepare()
    state.run()
    state.validate()
    state.use_captured = state.check_captured = False
    for _ in range(int(runtime.meta['random_draws'])):
        state.prepare()
        state.run()
        state.validate()
    first = runtime.call(state.args, route)
    first_copy = cpu_snapshot(first)
    second = runtime.call(state.args, route)
    require(first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr(),
            'consecutive calls share output storage')
    require(state.torch.equal(cpu_snapshot(first), first_copy), 'a later call mutated an earlier output')
    del first, second
    # The fixture corruption gate, plus no-op output poison, use the same checker.
    gold = state.captured_ref
    poisoned = state.torch.full_like(gold, float('nan'))
    expect_rejection(lambda: state.compare(poisoned, gold), 'no-op output')
    corrupted = gold.clone()
    corrupted.reshape(-1)[0] += 1024
    expect_rejection(lambda: state.compare(corrupted, gold), 'corrupted output')
    graph_correctness(state)
    engagement = engagement_and_smoke(state)
    return {'sig': spec['sig'] + '|' + spec['regime'], 'params': state.params,
            'selected_route': route, 'native_dispatch': state.dispatch,
            'resolved_execution': state.resolved_execution,
            'resolved_execution_observations': state.execution_observations,
            'resolved_capture_observations': state.capture_observations,
            'validation': {**validation(state), 'negative_controls': {'no_op': 'rejected', 'wrong_output': 'rejected'},
                           'random_draws': int(runtime.meta['random_draws']), 'graph_replay_values': 2,
                           'fresh_outputs': True, 'prior_output_unchanged': True,
                           'device_engagement': engagement}}


def validation(state):
    return {'expected_device': 'cpu', 'inputs_unchanged': True,
            'reference_after_candidate_snapshot': True,
            'captured_reference_checked': True, 'routes_weights_scales_preserved': True,
            'tolerance': state.tol, 'max_rel_err': state.worst_error,
            'checked_invocations': state.checked_invocations, 'fresh_input_sets': state.generated}


def measure_case(runtime, spec, route, seed, preflight):
    from _aka_benchmark import benchmark_cuda_graph_or_events_samples

    state = ReplayState(runtime, spec, route, seed)
    timed = TimedGraph()
    samples, metadata = benchmark_cuda_graph_or_events_samples(
        state.run, warmup=WARMUP, repetition=SAMPLES,
        prepare_fn=state.prepare, timed_run=timed)
    require(metadata.get('benchmark_method') == 'cuda_graph'
            and metadata.get('benchmark_effective_repeats') == 1
            and 'benchmark_fallback_reason' not in metadata, 'checked graph timing is mandatory')
    result = timed.rerun()
    require(result is state.outputs and result is timed.outputs, 'wrong graph output was validated')
    state.validate()
    require(state.checked_invocations == CHECKED and state.generated == PREPARED,
            'not every measured graph invocation was refreshed and checked')
    require(state.execution_observations == WARMUP + 2 and state.capture_observations == 2,
            'native arguments must be observed in every warmup and both captured graphs')
    return {'sig': spec['sig'] + '|' + spec['regime'], 'params': state.params,
            'selected_route': route, 'native_dispatch': state.dispatch,
            'resolved_execution': state.resolved_execution,
            'resolved_execution_observations': state.execution_observations,
            'resolved_capture_observations': state.capture_observations,
            'mean_ms': math.fsum(samples) / len(samples), 'median_ms': statistics.median(samples),
            'min_ms': min(samples), 'max_ms': max(samples), 'samples_ms': samples, **metadata,
            'validation': {**validation(state), 'outputs_poisoned_before_replay': True,
                           'sort_scratch_poisoned_before_replay': True, 'measured_graph_validated': True},
            'preflight': preflight['validation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.unlink(missing_ok=True)
    task = Path(__file__).resolve().parents[1]
    request = strict_json(args.request.read_text())
    validate_request(request, task, request['mode'])
    require(request['mode'] in ('correctness', 'performance'), 'worker requires a GPU evaluation phase')
    meta = strict_json((task / 'ut/meta.json').read_text())
    import torch
    runtime = NativeRuntime(task, torch, meta)
    rows = []
    for spec in specs(meta):
        route = request['source']['routes'][spec['sig']]
        preflight = correctness_case(runtime, spec, route, secrets.randbits(63))
        row = (measure_case(runtime, spec, route, secrets.randbits(63), preflight)
               if request['mode'] == 'performance' else preflight)
        rows.append(row)
        print(f"[checked] {row['sig']} route={route} max_rel_err={row['validation']['max_rel_err']:.6g}", flush=True)
    payload = {'status': 'ok', 'protocol': PROTOCOL, 'request': request,
               'request_sha256': fingerprint(request), 'cases': rows}
    payload['effective_execution_sha256'] = effective_execution_identity(payload, meta, request)
    if request['mode'] == 'performance':
        payload.update(timer='cuda_graph', warmup=WARMUP, iters=SAMPLES)
    validate_request(request, task, request['mode'])
    validate_payload(payload, meta, request)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp-' + secrets.token_hex(8))
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
