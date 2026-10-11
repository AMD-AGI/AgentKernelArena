"""CPU regressions for routing admission and the actual canonical graph adapter."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import importlib.util
import json
import math
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

    def test_effective_source_normalizes_comments_but_binds_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / 'task'
            shutil.copytree(TASK, task, symlinks=True, ignore=shutil.ignore_patterns('build', '__pycache__', '_aka_benchmark.py', 'reference_io.pt'))
            before = contract.source_context(task)
            source = task / contract.POLICY_FILE
            source.write_text('# comment-only change\n' + source.read_text())
            after = contract.source_context(task)
            self.assertEqual(before['effective_source_sha256'], after['effective_source_sha256'])
            self.assertNotEqual(before['source_file_sha256'], after['source_file_sha256'])
            source.write_text(source.read_text().replace('"captured"', '"block32"', 1))
            self.assertNotEqual(before['effective_source_sha256'], contract.source_context(task)['effective_source_sha256'])
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
                         'benchmark_warmup': 10, 'benchmark_samples': 100, 'benchmark_effective_repeats': 1,
                         'samples_ms': samples[:], 'mean_ms': math.fsum(samples) / 100,
                         'median_ms': (samples[49] + samples[50]) / 2, 'min_ms': min(samples), 'max_ms': max(samples),
                         'validation': {'expected_device': 'cpu', 'inputs_unchanged': True,
                                        'reference_after_candidate_snapshot': True, 'tolerance': 0.02,
                                        'max_rel_err': 0.0, 'captured_reference_checked': True,
                                        'routes_weights_scales_preserved': True, 'checked_invocations': 114,
                                        'fresh_input_sets': 116, 'outputs_poisoned_before_replay': True,
                                        'measured_graph_validated': True}})
        return {'status': 'ok', 'protocol': contract.PROTOCOL, 'request': request,
                'request_sha256': contract.fingerprint(request), 'cases': rows,
                'timer': 'cuda_graph', 'warmup': 10, 'iters': 100}, request

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
                     lambda p: p['cases'][0].update(benchmark_fallback_reason='event')]
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
