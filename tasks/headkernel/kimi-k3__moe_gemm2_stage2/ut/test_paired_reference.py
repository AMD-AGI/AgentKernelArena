"""CPU doubles exercise the actual paired graph/oracle control flow."""
from collections import Counter
from contextlib import contextmanager
from array import array
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from evaluation_contract import checked_replays
from fresh_runner import FreshCallbacks
from paired_reference import NativePair, PairedOracle, capture_graphs
import paired_input_validation as receipts


class Tensor:
    def __init__(self, value, torch, name, device='cuda'):
        self.value, self.torch, self.name = value, torch, name
        self.device = SimpleNamespace(type=device)

    def detach(self): return self

    def copy_(self, source): self.value = source.value

    def to(self, *, device, copy):
        assert copy
        self.torch.log.append(('snapshot', self.name, device))
        return Tensor(self.value, self.torch, self.name, device)

    def untyped_storage(self):
        return SimpleNamespace(tensor=self, data_ptr=lambda: id(self), nbytes=lambda: 1)


class Torch:
    uint8 = 'uint8'

    def __init__(self):
        self.log = []; self.active = None; self.captures = []
        self.cuda = SimpleNamespace(synchronize=lambda: self.log.append('sync'),
            Stream=self.stream_object, current_stream=self.stream_object,
            stream=self.stream_context, CUDAGraph=lambda: object(), graph=self.graph_context)

    def is_tensor(self, value): return isinstance(value, Tensor)

    def equal(self, left, right): return left.value == right.value

    def stream_object(self):
        return SimpleNamespace(wait_stream=lambda stream: None, synchronize=lambda: self.log.append('stream_sync'))

    @contextmanager
    def stream_context(self, stream):
        self.active = stream
        try: yield
        finally: self.active = None

    @contextmanager
    def graph_context(self, graph, stream):
        self.captures.append(stream)
        yield

    def empty(self, *args, **kwargs):
        torch = self
        class Raw:
            def set_(self, storage, *args): self.storage = storage; return self
            def copy_(self, source): self.storage.tensor.value = source.value
            def fill_(self, value): self.storage.tensor.value = value
            def zero_(self):
                torch.log.append(('scrub', self.storage.tensor.name))
                self.storage.tensor.value = 0
        return Raw()


def fixture(test):
    torch = Torch(); counts = Counter(); seeds = []
    source = Tensor(0, torch, 'candidate_input'); output = Tensor(0, torch, 'candidate_output')
    ref_source = Tensor(0, torch, 'reference_input'); expected = Tensor(0, torch, 'reference_output')
    prepared = SimpleNamespace(torch=torch, inputs={'a': source}, output=output)

    def refresh(seed):
        counts['refresh'] += 1; seeds.append(seed); source.value = seed * 7 + 13

    def initialize(): counts['initialize'] += 1; output.value = -999

    def candidate():
        counts['candidate'] += 1; torch.log.append('candidate'); output.value = source.value * 2

    def immutable(after, before):
        test.assertEqual(after['a'].value, before['a'].value)

    def metadata(): torch.log.append('candidate_metadata'); return True

    def compare(actual, reference):
        torch.log.append('compare'); test.assertEqual(actual.value, reference.value)

    prepared.invoke_candidate = candidate; prepared.validate_metadata = metadata
    prepared.assert_immutable = immutable; prepared.compare = compare
    callbacks = FreshCallbacks(refresh_inputs=refresh, initialize_outputs=initialize,
        snapshot_inputs=lambda: {'a': source}, snapshot_outputs=lambda: output,
        validate_metadata=metadata, assert_immutable=immutable, reference=lambda truth: test.fail('duplicate eager oracle'),
        compare=compare, replay=candidate, torch_module=torch)
    prepared.callbacks = callbacks
    def measure(call):
        counts['measure'] += 1; call(); return 0.25
    callbacks.measure = measure

    class Pair:
        def __init__(self): self.prepared = prepared; self.torch = torch; self.reference_output = expected
        def restore_reference(self, truth):
            torch.log.append('restore_reference'); ref_source.value = truth['a'].value
        def check_reference_inputs(self, truth, *, exact=False):
            test.assertEqual(ref_source.value, truth['a'].value)
            torch.log.append('reference_input_check')
        def check_reference_metadata(self): torch.log.append('reference_metadata')
        def invoke_reference(self):
            counts['reference'] += 1; torch.log.append('reference'); expected.value = ref_source.value * 2
        def scrub_reference(self):
            counts['scrub'] += 1; torch.log.append('scrub_reference'); expected.value = 0
        def assert_disjoint(self): counts['disjoint'] += 1
    pair = Pair()
    return torch, counts, seeds, prepared, pair, source, output, expected


class PairingTests(unittest.TestCase):
    def first_reference_setup_failure(self, output_names):
        torch, counts, seeds, prepared, _, source, output, _ = fixture(self)
        candidate_outputs = {name: Tensor(0, torch, 'candidate_' + name) for name in output_names}
        reference_outputs = {name: Tensor(0, torch, 'reference_' + name) for name in output_names}
        prepared.inputs = {'a': source, 'weight': Tensor(314, torch, 'candidate_weight'), **candidate_outputs}
        prepared.ref_inputs = {'a': Tensor(0, torch, 'reference_input'),
                               'weight': Tensor(314, torch, 'reference_weight'), **reference_outputs}
        prepared.ref_groups = dict(prepared.ref_inputs)
        prepared.callbacks._inputs = lambda: dict(prepared.inputs)
        invocations = []
        def fail(leg, inputs):
            self.assertEqual(leg, 'reference')
            self.assertIs(inputs, prepared.ref_inputs)
            invocations.append(leg)
            for name in output_names:
                inputs[name].value = 123456
            raise RuntimeError('first reference invocation failed after writes')
        prepared.invoke = fail
        pair = NativePair(prepared)
        with self.assertRaisesRegex(RuntimeError, 'first reference invocation failed'):
            capture_graphs(pair, {'measurement': {'warmup_iterations': 10, 'benchmark_iterations': 100}},
                           {'challenge_seed': 71})
        self.assertEqual(invocations, ['reference'])
        self.assertIsNone(pair.reference_output)
        self.assertEqual({name: value.value for name, value in reference_outputs.items()},
                         {name: 0 for name in output_names})
        self.assertEqual(prepared.ref_inputs['weight'].value, 314)
        self.assertEqual(prepared.ref_inputs['a'].value, source.value)
        self.assertEqual(output.value, 0)
        self.assertEqual(counts['candidate'], 4)

    def test_first_reference_stage2_setup_failure_scrubs_known_output(self):
        self.first_reference_setup_failure(('out',))

    def test_first_reference_lean_setup_failure_scrubs_all_known_scratch(self):
        self.first_reference_setup_failure(('o', 'Mp', 'Lp', 'Op', 'locks'))

    def test_known_and_returned_output_aliases_are_scrubbed_once(self):
        torch = Torch(); output = Tensor(123456, torch, 'shared_output')
        weight = Tensor(314, torch, 'readonly_weight')
        prepared = SimpleNamespace(torch=torch, ref_inputs={'out': output, 'weight': weight})
        pair = NativePair(prepared)
        pair.reference_output = {'out': output, 'return': output, 'return_scale': None}
        pair.scrub_reference()
        self.assertEqual(output.value, 0); self.assertEqual(weight.value, 314)
        self.assertEqual(torch.log.count(('scrub', 'shared_output')), 1)

    def test_stage1_reference_return_buffers_match_original_candidate_poison(self):
        torch = Torch(); prepared = SimpleNamespace(torch=torch, reference_inputs={})
        pair = NativePair(prepared)
        payload = Tensor(0, torch, 'reference_payload'); scales = Tensor(0, torch, 'reference_scales')
        pair.reference_output = {'out': None, 'return': payload, 'return_scale': scales}
        pair.restore_reference({})
        self.assertEqual(payload.value, 0xff); self.assertEqual(scales.value, 0xff)
        pair.scrub_reference()
        self.assertEqual(payload.value, 0); self.assertEqual(scales.value, 0)

    def test_real_adapter_bridge_restores_both_storage_styles_and_checks_aliases(self):
        for stage1 in (False, True):
            torch = Torch(); candidate = Tensor(1, torch, 'candidate')
            reference = Tensor(2, torch, 'reference'); original_output = Tensor(3, torch, 'output')
            reference_output = Tensor(4, torch, 'reference_output')
            inputs = {'a': candidate}; reference_inputs = {'a': reference}
            prepared = SimpleNamespace(torch=torch, inputs=inputs, output=original_output)
            def invoke(leg, bound):
                self.assertEqual(leg, 'reference'); self.assertIs(bound, reference_inputs)
                return reference_output
            if stage1:
                prepared.reference_inputs = reference_inputs; prepared._invoke = invoke
                truth = {'a': Tensor(99, torch, 'truth', 'cpu')}
            else:
                prepared.ref_inputs = reference_inputs
                # Give this storage a distinct alias to exercise the group path.
                reference.copy_ = lambda value: setattr(reference, 'value', value.value)
                prepared.ref_groups = {'storage_alias': reference}; prepared.invoke = invoke
                truth = {'storage_alias': Tensor(99, torch, 'truth', 'cpu')}
            pair = NativePair(prepared); pair.restore_reference(truth)
            self.assertEqual(reference.value, 99); self.assertIs(pair.invoke_reference(), reference_output)
            pair.assert_disjoint()
            def metadata():
                self.assertIs(prepared.inputs, reference_inputs)
                self.assertIs(prepared.output, reference_output)
                raise AssertionError('metadata failure')
            prepared.validate_metadata = metadata
            with self.assertRaisesRegex(AssertionError, 'metadata failure'): pair.check_reference_metadata()
            self.assertIs(prepared.inputs, inputs); self.assertIs(prepared.output, original_output)
            pair.reference_inputs = inputs
            with self.assertRaisesRegex(ValueError, 'storage aliases'): pair.assert_disjoint()

    def test_actual_cpu_control_observations_match_reconstructed_schedule(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root/'cases.json').read_text())
        registry = json.loads((root/'provenance/PAIRED-WORK-REGISTRY.json').read_text())
        stage1 = 'stage1' in root.name
        for case in manifest['cases']:
            geometry = registry['cases'][case['case_id']]; seed = 1843
            histogram = case['work_distribution'][geometry['histogram_field']]
            rng = random.Random(seed); work = receipts.draw(histogram, rng)
            if geometry['kind'] == 'moe':
                from routing import make_routes
                values = make_routes(geometry['tokens'], geometry['tile_m'], work, geometry['capacity_rows'],
                                     geometry['expert_slots'], seed, padding_token=geometry['padding_token'])
            else:
                # Independently express the actual native index selection as
                # scalar gathers, rather than the validator's array slices.
                valid = [row for start, size in geometry['valid_kv_spans'] for row in range(start, start+size)]
                starts = [rng.randrange(len(valid)) for _ in range(64)]
                ids = array('q', (valid[(start+j) % len(valid)] for start in starts for j in range(work)))
                ids.extend([valid[0]] * (geometry['kv_capacity'] - len(ids)))
                values = {'kv_indices': ids, 'kv_indptr': array('i', (i*work for i in range(65)))}
            metadata = {}; truth = {}
            for name, meta in geometry['control_metadata'].items():
                metadata[name] = {**meta, 'dtype': 'torch.'+meta['dtype'], 'storage_offset': 0,
                                  'element_size': 4 if meta['dtype'] == 'int32' else 8, 'alias': 'alias_'+name}
                truth[name if stage1 else 'alias_'+name] = values[name].tobytes()
            if geometry['kind'] == 'moe' and not stage1:
                metadata['topk_ids'] = None
            remaining = geometry['private_cpu_storage_bytes'] - sum(map(len, truth.values()))
            truth['other_storage'] = SimpleNamespace(numel=lambda: remaining, element_size=lambda: 1)
            prepared = SimpleNamespace(record={'inputs': metadata})
            if stage1: prepared.reference_inputs = {}
            actual = receipts.input_signature(prepared, seed, truth)
            expected = receipts.expected_signature(root, case, geometry, seed)
            self.assertEqual(actual, expected)
            key = next(name for name in truth if name != 'other_storage')
            truth[key] = bytes([truth[key][0] ^ 1]) + truth[key][1:]
            self.assertNotEqual(receipts.input_signature(prepared, seed, truth)['actual_control_hashes'], expected['actual_control_hashes'])

    def test_110_exact_pairs_with_100_timed_samples_per_leg_and_one_oracle(self):
        torch, counts, seeds, prepared, pair, source, output, expected = fixture(self)
        policy = {'method': 'cuda_graph', 'warmup_iterations': 10, 'benchmark_iterations': 100}
        oracle = PairedOracle(pair, SimpleNamespace(replay=pair.invoke_reference), policy)
        callbacks = prepared.callbacks
        row = checked_replays({'case_id': 'cpu'}, policy, reset_inputs=callbacks.reset_inputs,
            initialize_outputs=callbacks.initialize_outputs, replay=prepared.invoke_candidate,
            verify=oracle.verify, measure=callbacks.measure, observe=lambda: {'case_id': 'cpu'}, seed=1001)
        self.assertEqual(seeds, list(range(1001, 1111)))
        self.assertEqual(counts, Counter(refresh=110, initialize=110, candidate=110, reference=110, scrub=110, measure=200))
        self.assertEqual(len(row['samples_ms']), 100); self.assertEqual(len(oracle.samples), 100)
        self.assertEqual(oracle.index, 110); self.assertEqual(oracle.clears, 110)
        self.assertEqual(expected.value, 0)
        for index, item in enumerate(torch.log):
            if item == 'restore_reference':
                prefix = torch.log[:index]
                candidate = max(i for i, value in enumerate(prefix) if value == 'candidate')
                self.assertIn(('snapshot', 'candidate_output', 'cpu'), prefix[candidate:])
                self.assertIn(('snapshot', 'candidate_input', 'cpu'), prefix[candidate:])
            if item == 'compare': self.assertEqual(torch.log[index - 1], 'scrub_reference')

    def test_candidate_input_mutation_rejected_before_reference_and_scrubbed(self):
        torch, counts, seeds, prepared, pair, source, output, expected = fixture(self)
        cb = prepared.callbacks; truth = cb.reset_inputs(2); cb.initialize_outputs(); prepared.invoke_candidate()
        source.value += 1; expected.value = 123456
        oracle = PairedOracle(pair, SimpleNamespace(replay=pair.invoke_reference), {'warmup_iterations': 10})
        with self.assertRaises(AssertionError): oracle.verify(truth)
        self.assertEqual(counts['reference'], 0); self.assertEqual(expected.value, 0)

    def test_reference_failure_scrubs_partial_outputs(self):
        torch, counts, seeds, prepared, pair, source, output, expected = fixture(self)
        cb = prepared.callbacks; truth = cb.reset_inputs(3); cb.initialize_outputs(); prepared.invoke_candidate()
        def fail(): expected.value = 123456; raise RuntimeError('kernel failure')
        oracle = PairedOracle(pair, SimpleNamespace(replay=fail), {'warmup_iterations': 10})
        with self.assertRaisesRegex(RuntimeError, 'kernel failure'): oracle.verify(truth)
        self.assertEqual(expected.value, 0)

    def test_wrong_output_rejected_after_reference_scrub(self):
        torch, counts, seeds, prepared, pair, source, output, expected = fixture(self)
        cb = prepared.callbacks; truth = cb.reset_inputs(5); cb.initialize_outputs(); prepared.invoke_candidate()
        output.value = 0
        oracle = PairedOracle(pair, SimpleNamespace(replay=pair.invoke_reference), {'warmup_iterations': 10})
        with self.assertRaises(AssertionError): oracle.verify(truth)
        self.assertEqual(counts['reference'], 1); self.assertEqual(expected.value, 0)

    def test_matched_setup_counts_streams_and_nonpolicy_seed(self):
        torch, counts, seeds, prepared, pair, source, output, expected = fixture(self)
        manifest = {'measurement': {'warmup_iterations': 10, 'benchmark_iterations': 100}}
        graphs, streams = capture_graphs(pair, manifest, {'challenge_seed': 71})
        self.assertEqual(counts['candidate'], 4); self.assertEqual(counts['reference'], 4)
        self.assertEqual(counts['disjoint'], 1)
        self.assertEqual(set(seeds), {181}); self.assertEqual(torch.captures, streams)
        self.assertEqual(len(graphs), 2); self.assertIsNot(streams[0], streams[1])
        first_ref = torch.log.index('reference')
        self.assertIn(('snapshot', 'candidate_output', 'cpu'), torch.log[:first_ref])
        self.assertEqual(expected.value, 0)

    def test_reference_setup_failure_scrubs_both_outputs(self):
        torch, counts, seeds, prepared, pair, source, output, expected = fixture(self)
        def fail(): expected.value = 123456; raise RuntimeError('setup failure')
        pair.invoke_reference = fail
        with self.assertRaisesRegex(RuntimeError, 'setup failure'):
            capture_graphs(pair, {'measurement': {'warmup_iterations': 10, 'benchmark_iterations': 100}}, {'challenge_seed': 71})
        self.assertEqual(expected.value, 0); self.assertEqual(output.value, 0)


if __name__ == '__main__': unittest.main()
