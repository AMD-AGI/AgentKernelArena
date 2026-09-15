"""Behavioral controls for protected GEAK task outputs and captured replays."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/geak_eval'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TIMER = load_file('_geak_canonical_timer', ROOT / 'src/tools/perf/aka_benchmark.py')
with patch.dict(sys.modules, {'_aka_benchmark': TIMER}):
    CONTRACT = load_file('_geak_contract', TASKS / 'L2/topk/_timed_contract.py')


def function(path, name, **namespace):
    # Read the actual protected comparison without importing GPU-only kernels.
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    env = {'torch': torch, **namespace}
    exec(compile(module, str(path), 'exec'), env)
    return env[name]


class CapturedReplayTests(unittest.TestCase):
    def invoke(self, *, bad_original=False, bad_replay=False, mutate=False):
        x = torch.tensor([1.0, 2.0])
        history = []
        def kernel():
            if mutate:
                x.zero_()
            return x * 3
        def reference(saved):
            history.append(('reference', saved['x'].clone()))
            return saved['x'] * 3
        def benchmark(fn, *, timed_run, **options):
            self.assertEqual(options, {'warmup': 50, 'repetition': 200})
            history.append(('timing', x.clone()))
            actual = fn()
            if bad_original:
                actual.zero_()
            def replay():
                history.append(('replay', x.clone()))
                actual.copy_(torch.zeros_like(x) if bad_replay else fn())
                return actual
            timed_run._bind(replay, actual)
            return 0.25, {'benchmark_method': 'cuda_graph'}
        try:
            result = CONTRACT.checked_benchmark(
                benchmark, kernel, inputs={'x': x}, reference=reference,
                check=lambda actual, expected: torch.testing.assert_close(actual, expected),
                perturb=lambda saved: {'x': -saved['x']},
                warmup=50, repetition=200,
            )
            self.assertEqual(result[0], 0.25)
            self.assertTrue(result[1]['benchmark_replay_checked'])
            return history
        finally:
            torch.testing.assert_close(x, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
            self.assertEqual(history[0][0], 'reference')

    def test_actual_graph_and_changed_input_replay(self):
        h = self.invoke()
        self.assertEqual([x[0] for x in h], ['reference', 'timing', 'reference', 'replay'])
        torch.testing.assert_close(h[-1][1], torch.tensor([-1., -2.]))

    def test_original_wrong_output_cannot_be_hidden_by_correct_replay(self):
        with self.assertRaises(AssertionError):
            self.invoke(bad_original=True)

    def test_stale_replay_rejected_and_inputs_restored(self):
        with self.assertRaises(AssertionError):
            self.invoke(bad_replay=True)

    def test_candidate_cannot_contaminate_reference_inputs(self):
        with self.assertRaisesRegex(AssertionError, 'Read-only input changed'):
            self.invoke(mutate=True)

    def test_all_nested_public_outputs_checked(self):
        good = (torch.ones(2), (torch.ones(1), None))
        bad = (torch.ones(2), (torch.tensor([float('inf')]), None))
        with self.assertRaisesRegex(AssertionError, 'Nonfinite output'):
            CONTRACT.assert_output_contract(bad, good)

    def test_output_dtype_is_not_hidden_by_comparison_cast(self):
        with self.assertRaisesRegex(AssertionError, 'Wrong output dtype'):
            CONTRACT.assert_output_contract(torch.ones(2, dtype=torch.float64), torch.ones(2))

    def test_optional_outputs_cannot_be_dropped(self):
        with self.assertRaisesRegex(AssertionError, 'Missing tensor output'):
            CONTRACT.assert_output_contract((torch.ones(2), None), (torch.ones(2), torch.ones(1)))

    def test_eager_correctness_uses_pristine_reference(self):
        x = torch.tensor([2.0])
        def kernel():
            x.zero_()
            return x.clone()
        with self.assertRaisesRegex(AssertionError, 'Read-only input changed'):
            CONTRACT.checked_call(kernel, inputs={'x': x}, reference=lambda s: s['x'] * 2,
                                  check=torch.testing.assert_close)
        self.assertEqual(x.item(), 2.0)


class TopKContractTests(unittest.TestCase):
    def setUp(self):
        self.check = function(TASKS / 'L2/topk/test_kernel_harness.py', 'check_topk')

    def test_large_legacy_tolerance_cannot_accept_zero_answer(self):
        x = torch.zeros(1, 128256)
        x[0, -2:] = torch.tensor([1., 2.])
        expected = torch.topk(x, 2)
        wrong = (torch.zeros(1, 2), torch.tensor([[0, 1]]))
        # Confirm this negative control passes the retained numerical tolerance.
        torch.testing.assert_close(wrong[0], expected.values, atol=1e-4*x.shape[1], rtol=1.3e-6)
        with self.assertRaisesRegex(AssertionError, 'exact top-k'):
            self.check(wrong, expected, x)

    def test_indices_must_point_to_returned_values(self):
        x = torch.tensor([[0., 3., 2., 1.]])
        with self.assertRaises(AssertionError):
            self.check((torch.tensor([[3., 2.]]), torch.tensor([[1, 3]])), torch.topk(x, 2), x)

    def test_tied_values_allow_valid_distinct_alternate_indices(self):
        x = torch.tensor([[2., 2., 2., 0.]])
        self.check((torch.tensor([[2., 2.]]), torch.tensor([[0, 2]])), torch.topk(x, 2), x)

    def test_tied_values_do_not_allow_duplicate_indices(self):
        x = torch.tensor([[2., 2., 2., 0.]])
        with self.assertRaisesRegex(AssertionError, 'Duplicate'):
            self.check((torch.tensor([[2., 2.]]), torch.tensor([[0, 0]])), torch.topk(x, 2), x)


if __name__ == '__main__':
    unittest.main()
