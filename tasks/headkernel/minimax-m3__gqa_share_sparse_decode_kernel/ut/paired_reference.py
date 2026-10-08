"""One candidate replay and one deferred, timed protected reference per input."""
import hashlib
import json
import math
from pathlib import Path

from evaluation_contract import canonical, fingerprint


CORE_FIELDS = ('case', 'correct', 'samples_ms', 'warmup_iterations', 'fresh_input_resets',
               'output_initializations', 'oracle_checks', 'benchmark_method')


def require(condition, message):
    if not condition:
        raise ValueError('Paired reference: ' + message)


def clear_outputs(api, output):
    raw_storage = api['raw_storage']
    seen = set()
    for value in api['leaves'](output):
        pointer = value.untyped_storage().data_ptr()
        if pointer not in seen:
            raw_storage(value).fill_(0xAA)
            seen.add(pointer)


def initialize_outputs(api, inputs, output, initial_out):
    if 'initialize_output' in api:
        api['initialize_output'](output)
        return
    raw_storage = api['raw_storage']
    if initial_out is not None:
        inputs['out'].copy_(initial_out)
    mutable = inputs['out'].untyped_storage().data_ptr() if initial_out is not None else None
    for value in api['leaves'](output):
        if value.untyped_storage().data_ptr() != mutable:
            raw_storage(value).fill_(0xAA)


def device_time(call):
    import torch
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    start.record(); call(); stop.record(); stop.synchronize()
    elapsed = float(start.elapsed_time(stop))
    require(math.isfinite(elapsed) and elapsed > 0, 'invalid device sample')
    return elapsed


def capture_native_graph(api, case, fn, inputs, truth):
    """Warm and capture each leg on the same dedicated stream and own its output."""
    import torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    output = None
    reference_output_owned = False

    def own_output(value):
        nonlocal output, reference_output_owned
        output = value
        reference_output_owned = True

    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                api['restore_storages'](inputs, truth)
                output = api['invoke'](fn, inputs, own_output)
                stream.synchronize()
                # Freeze setup outputs too. Reference setup begins only after
                # candidate setup has produced these CPU-owned observations.
                api['cpu_clone'](output)
                api['assert_immutable_inputs'](inputs, truth)
                clear_outputs(api, output)
            api['restore_storages'](inputs, truth)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = api['invoke'](fn, inputs, own_output)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        api['assert_immutable_inputs'](inputs, truth)
        tensors, scalars = api['runtime_abi'](inputs, output)
        api['observe_case'](case, tensors, scalars)
        clear_outputs(api, output)
        torch.cuda.synchronize()
        return graph, output, stream
    except BaseException as error:
        if output is not None:
            if reference_output_owned:
                try:
                    # A failed attestation may precede the setup stream wait.
                    # Finish wrapper writes before clearing on the cleanup stream.
                    torch.cuda.synchronize()
                    clear_outputs(api, output)
                    torch.cuda.synchronize()
                except BaseException as cleanup_error:
                    error.add_note("Reference output cleanup also failed: " + repr(cleanup_error))
            else:
                clear_outputs(api, output)
                torch.cuda.synchronize()
        raise


def assert_disjoint_legs(api, inputs, output, reference_inputs, reference_output):
    def storages(values):
        return {value.untyped_storage().data_ptr() for value in api['leaves'](values)
                if value.untyped_storage().nbytes()}
    require(not (storages([inputs, output]) & storages([reference_inputs, reference_output])),
            'candidate/reference storage aliases')


class PairedOracle:
    def __init__(self, api, case, policy, inputs, output, reference_inputs, reference_output,
                 reference_graph, initial_out, tolerance):
        self.api, self.case, self.policy = api, case, policy
        self.inputs, self.output = inputs, output
        self.reference_inputs, self.reference_output = reference_inputs, reference_output
        self.reference_graph, self.initial_out, self.tolerance = reference_graph, initial_out, tolerance
        self.index = 0; self.samples = []; self.clears = 0

    def verify(self, truth):
        import torch
        api = self.api
        # No reference replay for this draw can occur before these owned CPU
        # observations. The comparison functions below are the original ones.
        try:
            torch.cuda.synchronize()
            actual_cpu = api['cpu_clone'](self.output)
            api['assert_immutable_inputs'](self.inputs, truth)
            api['restore_storages'](self.reference_inputs, truth)
            # Match the candidate's pre-timing CPU input observation, also
            # checking the byte-exact restoration before reference execution.
            api['assert_immutable_inputs'](self.reference_inputs, truth)
            initialize_outputs(api, self.reference_inputs, self.reference_output, self.initial_out)
            tensors, scalars = api['runtime_abi'](self.reference_inputs, self.reference_output)
            api['observe_case'](self.case, tensors, scalars)
            if self.index < self.policy['warmup_iterations']:
                self.reference_graph.replay()
            else:
                self.samples.append(device_time(self.reference_graph.replay))
            torch.cuda.synchronize()
            expected_cpu = api['cpu_clone'](self.reference_output)
            api['assert_immutable_inputs'](self.reference_inputs, truth)
        finally:
            clear_outputs(api, self.reference_output)
            torch.cuda.synchronize()
            self.clears += 1
        if 'compare_native_outputs' in api:
            api['compare_native_outputs'](actual_cpu, expected_cpu, self.inputs, self.tolerance,
                                          expected_inputs=truth)
        else:
            api['compare'](actual_cpu, expected_cpu, self.tolerance)
        self.index += 1

    def reference_row(self):
        count = self.policy['benchmark_iterations']; warmups = self.policy['warmup_iterations']
        require(self.index == count + warmups and self.clears == self.index and len(self.samples) == count,
                'incomplete checked reference replays or output clearing')
        return {'case': self.case, 'correct': True, 'samples_ms': self.samples,
                'warmup_iterations': warmups, 'fresh_input_resets': count,
                'output_initializations': count, 'oracle_checks': count,
                'benchmark_method': self.policy['method']}


def paired_performance(api, case, manifest, request, inputs, reference_inputs, fn, reference_fn,
                       initial_truth, initial_out, identity, reference_identity,
                       reset_inputs, current_input_signature=None):
    """Reuse the deferred oracle; preserve canonical checked_replays and counts."""
    import torch
    graph, output, stream = capture_native_graph(api, case, fn, inputs, initial_truth)
    reference_graph, reference_output, reference_stream = capture_native_graph(
        api, case, reference_fn, reference_inputs, initial_truth)
    assert_disjoint_legs(api, inputs, output, reference_inputs, reference_output)
    policy = manifest['measurement']
    oracle = PairedOracle(api, case, policy, inputs, output, reference_inputs, reference_output,
                          reference_graph, initial_out, api['tolerance'])
    trace = []

    def reset(seed):
        truth = reset_inputs(seed)
        signature = current_input_signature() if current_input_signature is not None else {
            'input_seed': seed, 'fixture': case['fixture']}
        require(signature['input_seed'] == seed, 'prepared input seed differs')
        trace.append(json.loads(canonical(signature)))
        return truth

    def initialize():
        initialize_outputs(api, inputs, output, initial_out)

    def observe():
        tensors, scalars = api['runtime_abi'](inputs, output)
        return api['observe_case'](case, tensors, scalars)

    try:
        row = api['checked_replays'](case, policy, reset_inputs=reset, initialize_outputs=initialize,
            replay=graph.replay, verify=oracle.verify, measure=device_time, observe=observe,
            seed=request['challenge_seed'])
        reference_row = oracle.reference_row()
        warmups = policy['warmup_iterations']; count = policy['benchmark_iterations']
        require(len(trace) == warmups + count, 'input schedule is incomplete')
        schedule = {'schema_version': 1, 'case_id': case['case_id'],
            'manifest_sha256': fingerprint(manifest), 'warmup_inputs': trace[:warmups],
            'measured_inputs': trace[warmups:]}
        schedule_sha = fingerprint(schedule)
        row['paired_schedule_sha256'] = schedule_sha
        reference_row['paired_schedule_sha256'] = schedule_sha
        row['paired_reference'] = {'case_id': case['case_id'],
            'candidate_binding': identity, 'reference_binding': reference_identity,
            'candidate_snapshot_before_reference': True, 'reference_reused_as_oracle': True,
            'output_conformance': True, 'input_schedule': schedule,
            'reference_output_clear_calls': oracle.clears,
            'checked_pair_count': warmups + count,
            'graph_setup': {'warmup_invocations_per_leg': 3, 'capture_invocations_per_leg': 1,
                'capture_on_warmed_stream': True, 'separate_graphs_and_outputs': True},
            'legs': {'candidate_port': {**{name: row[name] for name in CORE_FIELDS},
                                        'paired_schedule_sha256': schedule_sha},
                     'protected_reference': reference_row}}
        return row
    finally:
        clear_outputs(api, reference_output)
        torch.cuda.synchronize()
        del graph, reference_graph, output, reference_output, stream, reference_stream, oracle


def attach_comparison(root, report, manifest, request):
    """Emit data for the generic paired-reference primary scorer, without reruns."""
    root = Path(root); path = root / 'provenance/PAIRED-REFERENCE.json'
    provenance = json.loads(path.read_text())
    require(request['phase'] == 'performance' and request['manifest_sha256'] == fingerprint(manifest),
            'comparison request changed')
    rows = [row['paired_reference'] for row in report['cases']]
    require(len(rows) == len(manifest['cases']), 'paired comparison case coverage incomplete')
    report['paired_reference_comparison'] = {'schema_version': 1,
        'schema': 'paired-reference-comparison-v1', 'status': 'ok',
        'baseline_kind': provenance['baseline_kind'], 'score_input': True, 'diagnostic_only': False,
        'request': request, 'source_hashes': request['source_sha256'],
        'manifest_sha256': fingerprint(manifest), 'runtime_image': manifest['runtime_image'],
        'native_source_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'challenge_seed': request['challenge_seed'], 'cases': rows,
        'anti_cheat_attestation': False, 'fresh_trusted_host_retest_required': True}
    return report
