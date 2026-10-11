"""CPU regressions for routing admission and the actual canonical graph adapter."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import dataclasses
import functools
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import _routing_contract as contract
import _moe_runtime as runtime
import _bench as bench

TASK = Path(__file__).resolve().parents[1]
META = contract.strict_json((TASK / 'ut/meta.json').read_text())


class Tensor:
    serial = 0

    def __init__(self, values=(1., 2., 3., 4.), device='cuda', dtype='torch.bfloat16'):
        Tensor.serial += 1
        self.values, self.device, self.dtype = list(values), types.SimpleNamespace(type=device), dtype
        self.shape, self.pointer = (2, 2), Tensor.serial

    def stride(self):
        return (2, 1)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return self

    def data_ptr(self):
        return self.pointer

    def clone(self):
        return Tensor(self.values, self.device.type, self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return Tensor(self.values, 'cpu', self.dtype)

    def copy_(self, other):
        self.values[:] = other.values
        return self

    def uniform_(self, low, high, generator):
        generator.value += 1
        self.values[:] = [low + ((generator.value * 7 + i * 3) % 97) / 97 * (high - low) for i in range(4)]
        return self

    def reshape(self, *_shape):
        return self

    def __setitem__(self, index, value):
        self.values[index] = value

    def fill_(self, value):
        self.values[:] = [value] * 4
        return self

    def is_floating_point(self):
        return True


class FakeCuda:
    def __init__(self):
        self.capturing, self.active_graph = False, None

    def is_available(self):
        return True

    def is_current_stream_capturing(self):
        return self.capturing

    def synchronize(self):
        pass

    def Stream(self):
        return types.SimpleNamespace(wait_stream=lambda _s: None, synchronize=lambda: None)

    def current_stream(self):
        return self.Stream()

    def stream(self, _stream):
        return nullcontext()

    def Event(self, **_kwargs):
        return types.SimpleNamespace(record=lambda *_a: None, synchronize=lambda: None,
                                     elapsed_time=lambda _end: 0.02)

    def CUDAGraph(self):
        return types.SimpleNamespace(replay=lambda: None)

    @contextmanager
    def graph(self, graph, **_kwargs):
        self.capturing, self.active_graph = True, graph
        try:
            yield
        finally:
            self.capturing, self.active_graph = False, None


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__('torch')
        self.cuda = FakeCuda()

    def is_tensor(self, value):
        return isinstance(value, Tensor)

    def empty_strided(self, *_args, **kwargs):
        return Tensor(device=kwargs['device'])

    def Generator(self, **_kwargs):
        class Generator:
            value = 0

            def manual_seed(self, seed):
                self.value = seed
                return self
        return Generator()

    def equal(self, left, right):
        return left.values == right.values


def example_execution(block=32, stage='moe2'):
    metadata = {
        'stage1': {'callable': 'aiter.fused_moe:moe1'},
        'stage2': {'callable': 'aiter.fused_moe:' + stage},
        'block_m': block, 'ksplit': 0, 'run_1stage': False,
        'has_bias': False, 'use_non_temporal_load': True, 'fuse_quant': '',
        'stage2_has_bias': False, 'flat': False, 'output_aux': False,
        'prequant': True, 'skip_inter_quant': False, 'route_bucket': '',
        'expected_sorted_blocks': None, 'min_sorted_blocks': None, 'max_sorted_blocks': None,
    }
    return {'schema': 'qwen-moe-resolved-native-v1', 'effective_block_size_M': block,
            'metadata': [copy.deepcopy(metadata), copy.deepcopy(metadata)],
            'sorting': [{'args': ['ids', 'weights', 512, 8192, 'torch.bfloat16', block, None, None, 0],
                         'kwargs': {'accumulate': True, 'flat': False, 'return_local_topk_ids': False}}],
            'stages': [{'args': ['hidden', 'w1', 'w2', 10, 'sortedids', 'sortedweights', 'experts', 'valid',
                                'out', True, block], 'kwargs': {}}]}


class FakeRuntime:
    def __init__(self, torch):
        self.torch, self.meta = torch, META
        self.cpu = {'hidden_states': Tensor(device='cpu'), 'weights': Tensor(device='cpu')}
        self.args = {key: Tensor(value.values) for key, value in self.cpu.items()}
        self.reference_calls = 0
        self.fail_replay = None
        self.harness = types.SimpleNamespace(correct=self.correct)

    def correct(self, observed, expected, tol):
        rms = max(1e-6, math.sqrt(sum(value * value for value in expected.values) / 4))
        diff = [abs(a - b) for a, b in zip(observed.values, expected.values)]
        ok = all(delta <= tol * (rms + abs(ref)) for delta, ref in zip(diff, expected.values))
        return ok, max(diff) if ok else float('inf')

    def case(self, spec):
        return self.cpu, self.args, Tensor(device='cpu')

    def call(self, kwargs, route='captured', collector=None):
        output = Tensor()
        captured = self.torch.cuda.capturing
        if collector is not None:
            block = 32 if route == 'captured' else int(route[5:])
            collector.observe_execution(example_execution(block), None if route == 'captured' else block)

        def run():
            if collector is not None and (self.fail_replay == 'no_op'
                                          or (captured and self.fail_replay == 'graph_no_op')):
                return
            output.values[:] = [value * 2 for value in kwargs['hidden_states'].values]
            if collector is not None and self.fail_replay == 'mutate_input':
                kwargs['weights'].values[0] += 1
        if self.torch.cuda.capturing:
            self.torch.cuda.active_graph.replay = run
        else:
            run()
        if collector is None:
            self.reference_calls += 1
        return output


class ContractTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_fifo_policy_is_rejected_without_opening_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / 'task'
            source = task / contract.POLICY_FILE
            source.parent.mkdir(parents=True)
            os.mkfifo(source)
            # This trap makes the regression safe even if the guard regresses:
            # opening the FIFO is forbidden rather than allowed to hang the test.
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('FIFO was opened')) as read:
                with self.assertRaisesRegex(ValueError, 'task-local regular file'):
                    contract.source_context(task, META)
                read.assert_not_called()

    def test_policy_is_literal_data_only(self):
        source = (TASK / contract.POLICY_FILE).read_bytes()
        self.assertEqual(set(contract.parse_policy(source, META).values()), {'captured'})
        for poisoned in (
            b'import os\n' + source,
            source.replace(b'"captured"', b'__import__("os").getcwd()', 1),
            source.replace(b'def select_routes():', b'@print\ndef select_routes():'),
            source.replace(b'"captured"', b'"block256"', 1),
            source.replace(b'"prefill_m8192_live"', b'"decode_m64_live"'),
            source + b'\nprint("candidate host side effect")\n',
        ):
            with self.subTest(poisoned=poisoned), self.assertRaises((ValueError, SyntaxError)):
                contract.parse_policy(poisoned, META)

    def test_declared_policy_normalizes_comments_but_execution_requires_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / 'task'
            shutil.copytree(TASK, task, symlinks=True, ignore=shutil.ignore_patterns('build', '__pycache__', '_aka_benchmark.py', 'reference_io.pt'))
            before = contract.source_context(task)
            source = task / contract.POLICY_FILE
            source.write_text('# comment-only change\n' + source.read_text())
            after = contract.source_context(task)
            self.assertIsNone(before['effective_source_sha256'])
            self.assertIsNone(after['effective_source_sha256'])
            self.assertEqual(before['declared_policy_sha256'], after['declared_policy_sha256'])
            self.assertNotEqual(before['source_file_sha256'], after['source_file_sha256'])
            source.write_text(source.read_text().replace('"captured"', '"block32"', 1))
            changed = contract.source_context(task)
            self.assertIsNone(changed['effective_source_sha256'])
            self.assertNotEqual(before['declared_policy_sha256'], changed['declared_policy_sha256'])
            native = task / 'ut' / META['source_provenance']['baseline_ref']
            native.write_text(native.read_text() + '\n# tampered baseline\n')
            with self.assertRaisesRegex(ValueError, 'native source changed'):
                contract.source_context(task)

    def payload(self):
        request = contract.make_request(TASK, 'performance', 'a' * 48)
        samples = [0.1 + index / 10000 for index in range(100)]
        rows = []
        for spec in contract.specs(META):
            rows.append({'sig': spec['sig'] + '|' + spec['regime'], 'params': contract.case_params(META, spec),
                         'selected_route': 'captured', 'benchmark_method': 'cuda_graph',
                         'native_dispatch': {'declared_block_size_M': None, 'native_block_m': 32, 'effective_block_size_M': 32},
                         'resolved_execution': example_execution(),
                         'resolved_execution_observations': 12, 'resolved_capture_observations': 2,
                         'benchmark_warmup': 10, 'benchmark_samples': 100, 'benchmark_effective_repeats': 1,
                         'samples_ms': samples[:], 'mean_ms': math.fsum(samples) / 100,
                         'median_ms': (samples[49] + samples[50]) / 2, 'min_ms': min(samples), 'max_ms': max(samples),
                         'validation': {'expected_device': 'cpu', 'inputs_unchanged': True,
                                        'reference_after_candidate_snapshot': True, 'tolerance': 0.02,
                                        'max_rel_err': 0.0, 'captured_reference_checked': True,
                                        'routes_weights_scales_preserved': True, 'checked_invocations': 114,
                                        'fresh_input_sets': 116, 'outputs_poisoned_before_replay': True,
                                        'measured_graph_validated': True}})
        payload = {'status': 'ok', 'protocol': contract.PROTOCOL, 'request': request,
                   'request_sha256': contract.fingerprint(request), 'cases': rows,
                   'timer': 'cuda_graph', 'warmup': 10, 'iters': 100}
        payload['effective_execution_sha256'] = contract.effective_execution_identity(payload, META, request)
        return payload, request

    def test_report_rejects_missing_cases_samples_abi_and_stale_source(self):
        payload, request = self.payload()
        self.assertEqual(len(contract.validate_payload(payload, META, request)), 2)
        mutations = [lambda p: p['cases'].pop(),
                     lambda p: p['cases'].reverse(),
                     lambda p: p['cases'][0]['samples_ms'].pop(),
                     lambda p: p['cases'][0]['params'].update(topk=8),
                     lambda p: p['cases'][0].update(mean_ms=0.0001),
                     lambda p: p['cases'][0]['validation'].update(checked_invocations=113),
                     lambda p: p.update(request_sha256='0' * 64),
                     lambda p: p['cases'][0].update(benchmark_fallback_reason='event'),
                     lambda p: p['cases'][0].pop('resolved_execution'),
                     lambda p: p['cases'][0].update(resolved_capture_observations=1),
                     lambda p: p['cases'][0]['resolved_execution'].update(effective_block_size_M=64),
                     lambda p: p['cases'][0]['resolved_execution']['metadata'][0]['stage2'].update(callable='other'),
                     lambda p: p.update(effective_execution_sha256='0' * 64)]
        for mutate in mutations:
            invalid = copy.deepcopy(payload)
            mutate(invalid)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                contract.validate_payload(invalid, META, request)

    def test_failed_compile_clears_stale_scoreable_report(self):
        import task_runner
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / 'task'
            shutil.copytree(TASK, task, symlinks=True,
                            ignore=shutil.ignore_patterns('build', '__pycache__', 'reference_io.pt'))
            (task / 'build').mkdir()
            report = task / 'build/compile_report.json'
            report.write_text('{"status":"ok","test_cases":[{"execution_time_ms":0.001}]}')
            (task / contract.POLICY_FILE).write_text('import os\n')
            with patch.object(task_runner, 'TASK', task), self.assertRaises(ValueError):
                task_runner.evaluate('compile', 10)
            failed = json.loads(report.read_text())
            self.assertEqual(failed['status'], 'fail')
            self.assertEqual(failed['test_cases'], [])

    def test_routing_changes_only_native_block_hint(self):
        native = object.__new__(runtime.NativeRuntime)
        calls = []
        native.fn = lambda **kwargs: calls.append(kwargs)
        original = {'hidden_states': object(), 'expert_mask': object(), 'activation': object(), 'splitk': 0}
        native.call(original, 'captured')
        native.call(original, 'block64')
        self.assertEqual(calls[0], original)
        self.assertEqual(calls[1], dict(original, block_size_M=64))
        self.assertNotIn('block_size_M', original)
        with self.assertRaises(ValueError):
            native.call(original, 'arbitrary_callback')

    def test_mutated_input_is_rejected_before_reference(self):
        native = FakeRuntime(FakeTorch())
        native.fail_replay = 'mutate_input'
        state = runtime.ReplayState(native, META['workload']['cases'][0], 'captured', 17)
        state.prepare()
        state.run()
        with self.assertRaisesRegex(ValueError, 'mutated weights'):
            state.validate()
        self.assertEqual(native.reference_calls, 0)

    def test_relocated_native_enums_keep_the_same_execution_identity(self):
        def exports(module):
            values = {}
            for name, member, number in (('ActivationType', 'Silu', 0), ('QuantType', 'per_1x32', 3)):
                kind = type(name, (), {'__module__': module})
                instance = kind()
                instance.name, instance.value = member, number
                values[name] = kind
                values[name + '_value'] = instance
            return types.SimpleNamespace(**values)
        ordinary = exports('aiter.jit.module_aiter_core')
        relocated = exports('module_aiter_core')
        records = []
        for native in (ordinary, relocated):
            registry = runtime.native_enum_identities(native)
            record = runtime.native_value({'activation': native.ActivationType_value,
                                           'quant_type': native.QuantType_value}, native_enums=registry)
            records.append(record)
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[0]['activation'], {'enum': 'aiter.jit.module_aiter_core.ActivationType',
                                                   'name': 'Silu', 'value': 0})
        self.assertEqual(contract.fingerprint(records[0]), contract.fingerprint(records[1]))
        with self.assertRaisesRegex(ValueError, 'unsupported native launch metadata'):
            runtime.native_value(relocated.ActivationType_value)
        # A lookalike type with the ordinary module spelling is not an image export.
        spoof = exports('aiter.jit.module_aiter_core')
        with self.assertRaisesRegex(ValueError, 'unsupported native launch metadata'):
            runtime.native_value(spoof.ActivationType_value,
                                 native_enums=runtime.native_enum_identities(ordinary))

    def test_native_default_and_explicit_alias_have_identical_observed_arguments(self):
        @dataclasses.dataclass
        class Metadata:
            stage1: object
            stage2: object
            block_m: int
            ksplit: int = 0
            run_1stage: bool = False

        def stage1():
            pass

        def stage2():
            pass

        for block in (32, 64, 128):
            native = object.__new__(runtime.NativeRuntime)
            native.torch = FakeTorch()
            native.native_enums = {}
            native.dispatcher = lambda: Metadata(functools.partial(stage1, kernelName='moe1'),
                                                 functools.partial(stage2, kernelName='moe2'), block)
            native.sorting = lambda *args, **kwargs: [Tensor() for _ in range(5)]

            def two_stages(*args, **kwargs):
                native._dispatch()
                return Tensor()

            native.two_stages = two_stages

            def production(**kwargs):
                metadata = native._dispatch()
                actual = kwargs.get('block_size_M')
                actual = metadata.block_m if actual is None else actual
                native._sorting('ids', 'weights', 512, 8192, 'torch.bfloat16', actual,
                                None, None, 0, accumulate=True, flat=False)
                return native._two_stages('hidden', 'w1', 'w2', 10, 'ids', 'weights', 'experts',
                                          'valid', 'output', True, actual)

            native.fn = production
            observations = []
            collector = types.SimpleNamespace(observe_execution=lambda value, declared:
                                              observations.append((copy.deepcopy(value), declared)))
            native.call({}, 'captured', collector)
            native.call({}, 'block' + str(block), collector)
            self.assertIsNone(observations[0][1])
            self.assertEqual(observations[1][1], block)
            self.assertEqual(observations[0][0], observations[1][0])
            self.assertEqual(contract.fingerprint(observations[0][0]), contract.fingerprint(observations[1][0]))
            self.assertNotIn('0x', json.dumps(observations[0][0]))

    def test_native_execution_changes_between_capture_invocations_are_rejected(self):
        state = runtime.ReplayState(FakeRuntime(FakeTorch()), META['workload']['cases'][0], 'captured', 1)
        state.observe_execution(example_execution(32), None)
        with self.assertRaisesRegex(ValueError, 'native execution changed'):
            state.observe_execution(example_execution(64), None)
        with self.assertRaisesRegex(ValueError, 'native execution changed'):
            state.observe_execution(example_execution(32, stage='different_moe2'), None)

    def test_canonical_graph_adapter_checks_every_completed_replay(self):
        helper_path = TASK / 'scripts/_aka_benchmark.py'
        self.assertTrue(helper_path.is_file(), 'materialize canonical performance helper before this CPU test')
        torch = FakeTorch()
        with patch.dict(sys.modules, {'torch': torch}):
            spec = importlib.util.spec_from_file_location('_aka_benchmark', helper_path)
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            with patch.dict(sys.modules, {'_aka_benchmark': helper}):
                native = FakeRuntime(torch)
                row = bench.measure_case(native, META['workload']['cases'][0], 'captured', 1, {'validation': {}})
                self.assertEqual(row['validation']['checked_invocations'], 114)
                self.assertEqual(row['validation']['fresh_input_sets'], 116)
                self.assertEqual(len(row['samples_ms']), 100)
                self.assertEqual(native.reference_calls, 114)
                for fault in ('no_op', 'graph_no_op'):
                    native = FakeRuntime(torch)
                    native.fail_replay = fault
                    with self.subTest(fault=fault), self.assertRaises((ValueError, RuntimeError)):
                        bench.measure_case(native, META['workload']['cases'][0], 'captured', 1, {'validation': {}})
                    if fault == 'graph_no_op':
                        self.assertGreaterEqual(native.reference_calls, 10)


if __name__ == '__main__':
    unittest.main()
