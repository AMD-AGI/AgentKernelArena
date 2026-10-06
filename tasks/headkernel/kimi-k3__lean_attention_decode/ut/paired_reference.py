"""Time each protected native oracle on the candidate's exact private inputs."""
import hashlib
import json
from pathlib import Path

from evaluation_contract import canonical, checked_replays, fingerprint
from fresh_runner import clear_device_reference, cpu_copy, require_cpu
from paired_input_validation import input_signature, sampling_receipt

CORE = ('case', 'correct', 'samples_ms', 'warmup_iterations', 'fresh_input_resets',
        'output_initializations', 'oracle_checks', 'benchmark_method')


def require(value, message):
    if not value:
        raise ValueError('Kimi paired reference: ' + message)


def raw_storage(value, torch):
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))


class NativePair:
    """Use the original adapter's bindings, metadata checks and comparison math."""
    def __init__(self, prepared):
        self.prepared = prepared
        self.torch = prepared.torch
        self.stage1 = hasattr(prepared, 'reference_inputs')
        self.reference_inputs = prepared.reference_inputs if self.stage1 else prepared.ref_inputs
        self.reference_output = None

    def reference_storages(self):
        if not self.stage1:
            return self.prepared.ref_groups
        return {name: raw_storage(value, self.torch) for name, value in self.reference_inputs.items()
                if value is not None}

    def restore_reference(self, truth):
        # Stage 1 returns separate buffers: match its original candidate
        # output poison before timing. Stage 2/Lean output and scratch aliases
        # are initialized by the exact post-initialization input truth below.
        if self.stage1 and self.reference_output is not None:
            for value in self.reference_output.values():
                if value is not None:
                    raw_storage(value, self.torch).fill_(0xff)
        storages = self.reference_storages()
        require(set(storages) == set(truth), 'reference storage scope differs')
        for name, value in truth.items():
            storages[name].copy_(value)

    def invoke_reference(self):
        p = self.prepared
        self.reference_output = (p._invoke('reference', self.reference_inputs) if self.stage1
                                 else p.invoke('reference', self.reference_inputs))
        return self.reference_output

    def check_reference_metadata(self):
        p = self.prepared
        inputs, output = p.inputs, p.output
        try:
            p.inputs, p.output = self.reference_inputs, self.reference_output
            require(p.validate_metadata() is not False, 'reference metadata differs')
        finally:
            p.inputs, p.output = inputs, output

    def check_reference_inputs(self, truth, *, exact=False):
        after = cpu_copy(self.reference_storages(), self.torch)
        if exact:
            require(set(after) == set(truth), 'reference input scope differs')
            for name in truth:
                require(self.torch.equal(after[name], truth[name]), 'reference did not restore exact input bytes: ' + name)
        self.prepared.assert_immutable(after, truth)

    def scrub_reference(self):
        # Stage 2 and Lean can write these existing input/output allocations
        # before their first invocation returns. Their cleanup must not depend
        # on assigning reference_output successfully. The recursive clear also
        # covers Stage 1's separate returned buffers and deduplicates aliases.
        known_outputs = {name: self.reference_inputs[name]
                         for name in ('out', 'o', 'Mp', 'Lp', 'Op', 'locks')
                         if name in self.reference_inputs}
        if clear_device_reference([known_outputs, self.reference_output], self.torch):
            self.torch.cuda.synchronize()

    def assert_disjoint(self):
        def pointers(values):
            if self.torch.is_tensor(values):
                return {values.untyped_storage().data_ptr()} if values.untyped_storage().nbytes() else set()
            if isinstance(values, dict):
                return set().union(*(pointers(v) for v in values.values()))
            if isinstance(values, (list, tuple)):
                return set().union(*(pointers(v) for v in values))
            return set()
        p = self.prepared
        require(not (pointers([p.inputs, p.output]) & pointers([self.reference_inputs, self.reference_output])),
                'candidate and reference storage aliases')


def bindings(prepared, provenance):
    engagement = prepared.native_engagement()
    result = {}
    for leg in ('candidate', 'reference'):
        result[leg] = {
            'leg': leg, 'module': engagement[leg + '_callable'].split(':')[0],
            'callable': engagement[leg + '_callable'],
            'source_sha256': (prepared.runtime.request['source_sha256'] if leg == 'candidate'
                              else provenance['reference_source_sha256']),
            'gpu_binding': provenance['gpu_binding'],
            'native_binding': engagement.get(leg + '_binding'),
            'native_source_sha256': engagement['native_source_sha256'],
        }
    require(result['candidate']['module'] != result['reference']['module'], 'native module isolation lost')
    return result


def capture_graphs(pair, manifest, request):
    """Capture both legs on their own warmed streams outside all policy draws."""
    p = pair.prepared; callbacks = p.callbacks; torch = pair.torch
    policy = manifest['measurement']
    setup_seed = request['challenge_seed'] + policy['warmup_iterations'] + policy['benchmark_iterations']
    truth = callbacks.reset_inputs(setup_seed)
    callbacks.initialize_outputs()
    graphs = []; streams = []
    # Candidate setup observations precede reference setup. No policy draw
    # uses this setup seed. The original correctness phase is unchanged.
    for leg in ('candidate', 'reference'):
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        streams.append(stream)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    if leg == 'candidate':
                        truth = callbacks.reset_inputs(setup_seed)
                        callbacks.initialize_outputs()
                        p.invoke_candidate(); stream.synchronize()
                        p.validate_metadata()
                        cpu_copy(p.output, torch)
                        p.assert_immutable(cpu_copy(callbacks._inputs(), torch), truth.value)
                        clear_device_reference(p.output, torch)
                    else:
                        pair.scrub_reference(); pair.restore_reference(truth.value)
                        pair.check_reference_inputs(truth.value, exact=True)
                        pair.invoke_reference(); stream.synchronize()
                        pair.check_reference_metadata()
                        cpu_copy(pair.reference_output, torch)
                        pair.check_reference_inputs(truth.value)
                        pair.scrub_reference()
                if leg == 'candidate':
                    truth = callbacks.reset_inputs(setup_seed); callbacks.initialize_outputs()
                else:
                    pair.restore_reference(truth.value)
                    pair.check_reference_inputs(truth.value, exact=True)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                p.invoke_candidate() if leg == 'candidate' else pair.invoke_reference()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            # Capture invocation may only record launches. Observe storage and
            # metadata here; numerical correctness is checked on every replay.
            if leg == 'candidate':
                p.validate_metadata(); cpu_copy(p.output, torch)
                p.assert_immutable(cpu_copy(callbacks._inputs(), torch), truth.value)
                clear_device_reference(p.output, torch)
            else:
                pair.check_reference_metadata(); cpu_copy(pair.reference_output, torch)
                pair.check_reference_inputs(truth.value); pair.scrub_reference()
            torch.cuda.synchronize(); graphs.append(graph)
        except BaseException:
            pair.scrub_reference()
            if p.output is not None:
                clear_device_reference(p.output, torch); torch.cuda.synchronize()
            raise
    pair.assert_disjoint()
    return graphs, streams


class PairedOracle:
    def __init__(self, pair, reference_graph, policy):
        self.pair = pair; self.graph = reference_graph; self.policy = policy
        self.samples = []; self.index = 0; self.clears = 0

    def verify(self, truth):
        pair = self.pair; p = pair.prepared; cb = p.callbacks; torch = pair.torch
        require(truth is cb._truth, 'stale private input truth')
        require_cpu(truth.value, torch)
        try:
            torch.cuda.synchronize(); p.validate_metadata()
            actual = cpu_copy(p.output, torch)
            after = cpu_copy(cb._inputs(), torch)
            p.assert_immutable(after, truth.value)
            # Reference restoration, initialization and timing occur only
            # after both immutable candidate observations reside on CPU.
            pair.restore_reference(truth.value)
            pair.check_reference_inputs(truth.value, exact=True)
            pair.check_reference_metadata()
            if self.index < self.policy['warmup_iterations']:
                self.graph.replay()
            else:
                self.samples.append(cb.measure(self.graph.replay))
            torch.cuda.synchronize()
            pair.check_reference_metadata()
            expected = cpu_copy(pair.reference_output, torch)
            pair.check_reference_inputs(truth.value)
        finally:
            pair.scrub_reference()
            self.clears += 1
        require_cpu(actual, torch); require_cpu(expected, torch)
        require(p.compare(actual, expected) is not False, 'original oracle rejected paired output')
        cb._truth = None; self.index += 1
        return True


def paired_performance(prepared, manifest, request):
    pair = NativePair(prepared); callbacks = prepared.callbacks
    provenance = json.loads((prepared.root/'provenance/PAIRED-REFERENCE.json').read_text())
    trace = []; actual_receipts = []
    try:
        graphs, streams = capture_graphs(pair, manifest, request)
        oracle = PairedOracle(pair, graphs[1], manifest['measurement'])

        def reset(seed):
            truth = callbacks.reset_inputs(seed)
            signature = input_signature(prepared, seed, truth.value)
            trace.append(json.loads(canonical(signature)))
            actual_receipts.append(json.loads(canonical(signature)))
            return truth

        row = checked_replays(prepared.case, manifest['measurement'], reset_inputs=reset,
            initialize_outputs=callbacks.initialize_outputs, replay=graphs[0].replay,
            verify=oracle.verify, measure=callbacks.measure, observe=prepared.observe,
            seed=request['challenge_seed'])
        policy = manifest['measurement']; warm = policy['warmup_iterations']; count = policy['benchmark_iterations']
        require(oracle.index == oracle.clears == len(trace) == warm + count and len(oracle.samples) == count,
                'incomplete paired replays')
        schedule = {'schema_version': 1, 'case_id': prepared.case['case_id'],
                    'manifest_sha256': fingerprint(manifest), 'warmup_inputs': trace[:warm],
                    'measured_inputs': trace[warm:]}
        schedule_sha = fingerprint(schedule); binding = bindings(prepared, provenance)
        row['paired_schedule_sha256'] = schedule_sha
        row['realized_input_signatures'] = actual_receipts
        row['workload_control_sampling'] = sampling_receipt(prepared.case, request, trace, row['samples_ms'], policy)
        candidate = {key: row[key] for key in CORE}; candidate['paired_schedule_sha256'] = schedule_sha
        reference = dict(candidate, samples_ms=oracle.samples)
        row['paired_reference'] = {'case_id': prepared.case['case_id'], 'candidate_binding': binding['candidate'],
            'reference_binding': binding['reference'], 'candidate_snapshot_before_reference': True,
            'reference_reused_as_oracle': True, 'output_conformance': True,
            'input_schedule': schedule, 'reference_output_clear_calls': oracle.clears,
            'checked_pair_count': warm + count,
            'graph_setup': {'warmup_invocations_per_leg': 3, 'capture_invocations_per_leg': 1,
                            'capture_on_warmed_stream': True, 'separate_graphs_and_outputs': True},
            'legs': {'candidate_port': candidate, 'protected_reference': reference}}
        return row
    finally:
        pair.scrub_reference()
        if prepared.output is not None:
            clear_device_reference(prepared.output, pair.torch); pair.torch.cuda.synchronize()


def attach_comparison(root, report, manifest, request):
    path = Path(root)/'provenance/PAIRED-REFERENCE.json'
    provenance = json.loads(path.read_text())
    report['paired_reference_comparison'] = {'schema_version': 1, 'schema': 'paired-reference-comparison-v1',
        'status': 'ok', 'baseline_kind': provenance['baseline_kind'], 'score_input': True, 'diagnostic_only': False,
        'request': request, 'source_hashes': request['source_sha256'], 'manifest_sha256': fingerprint(manifest),
        'runtime_image': manifest['runtime_image'], 'native_source_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'challenge_seed': request['challenge_seed'], 'cases': [row['paired_reference'] for row in report['cases']],
        'anti_cheat_attestation': False, 'fresh_trusted_host_retest_required': True}
    return report
