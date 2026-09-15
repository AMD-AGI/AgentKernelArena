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


class AdditionalOperatorContractTests(unittest.TestCase):
    def test_mla_nonfinite_tail_cannot_use_five_percent_allowance(self):
        check = function(TASKS / 'L1/mla_decode/test_kernel_harness.py', 'check_correctness_val',
                         assert_output_contract=CONTRACT.assert_output_contract)
        ref = torch.ones(1, 16, 512)
        bad = ref.clone(); bad.flatten()[0] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'Nonfinite'):
            check(ref, bad)

    def test_mla_original_finite_error_ratio_policy_retained(self):
        check = function(TASKS / 'L1/mla_decode/test_kernel_harness.py', 'check_correctness_val',
                         assert_output_contract=CONTRACT.assert_output_contract)
        ref = torch.ones(100)
        bad = ref.clone(); bad[:4] += .1
        self.assertTrue(check(ref, bad)[0])
        bad[4:6] += .1
        self.assertFalse(check(ref, bad)[0])

    def test_mla_reference_zero_query_known_average(self):
        ref = function(TASKS / 'L1/mla_decode/test_kernel_harness.py', 'run_ref')
        inputs = dict(q=torch.zeros(1, 2, 3), k_input=torch.randn(3, 1, 3),
                      v_input=torch.tensor([[[1., 3.]], [[2., 6.]], [[3., 9.]]]),
                      kv_indices=torch.arange(3), output=torch.empty(1, 2, 2), sm_scale=.5)
        torch.testing.assert_close(ref(inputs), torch.tensor([[[2., 6.], [2., 6.]]]))

    def test_rms_correct_gradient_does_not_hide_wrong_forward(self):
        check = function(TASKS / 'L2/fast_rms_layernorm/test_kernel_harness.py', '_check_rms_outputs',
                         assert_output_contract=CONTRACT.assert_output_contract)
        with self.assertRaises(AssertionError):
            check((torch.zeros(3), torch.ones(3)), (torch.ones(3), torch.ones(3)))

    def test_routing_independent_known_selection_and_leftmost_tie(self):
        ref = function(TASKS / 'L1/moe_routing_sigmoid_top1/test_kernel_harness.py', '_routing_reference')
        x = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.bfloat16)
        w = torch.tensor([[0., 1., 2.], [0., 0., 0.]], dtype=torch.bfloat16)
        ids, weights = ref({'x': x, 'w': w}, shared=False)
        self.assertEqual(ids.tolist(), [[2], [0]])
        torch.testing.assert_close(weights, torch.sigmoid(torch.tensor([[2.], [0.]])))

    def test_mxfp4_packed_sign_perturbation_is_true_negation(self):
        decode = function(TASKS / 'L3/fused_moe_mxfp4/test_kernel_harness.py', '_mxfp4_to_f32',
                          _MXFP4_LUT=[0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        packed = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        torch.testing.assert_close(decode(packed.bitwise_xor(0x88)), -decode(packed), atol=0, rtol=0)


class DegenerateInputLayoutTests(unittest.TestCase):
    def test_singleton_strided_scale_and_scalar_still_checked_bytewise(self):
        with patch.dict(sys.modules, {'_aka_benchmark': TIMER}):
            module = load_file('_geak_refk_contract', TASKS / 'L1/refk_fp8_blockwise_mm/_timed_contract.py')
        storage = torch.arange(8.)
        scale = storage.as_strided((1, 1), (8, 2))
        scalar = torch.tensor(3.)
        guard = module.PristineInputs({'scale': scale, 'scalar': scalar})
        guard.check()
        scale.add_(1)
        with self.assertRaisesRegex(AssertionError, 'Read-only input changed'):
            guard.check()
        guard.restore()
        guard.check()
        self.assertEqual(scale.stride(), (8, 2))


class FP8VariantOracleTests(unittest.TestCase):
    def setUp(self):
        self.oracle = load_file('_geak_fp8_oracles', TASKS / 'L3/fused_rms_fp8/_contract_oracles.py')
        self.dtype = torch.float8_e4m3fnuz

    def test_quantization_known_zero_and_unit_groups(self):
        x = torch.cat((torch.zeros(1, 128), torch.ones(1, 128)), dim=1)
        q, scale = self.oracle.quantize(x, self.dtype)
        self.assertTrue(torch.equal(q[:, :128].float(), torch.zeros(1, 128)))
        self.assertTrue(torch.equal(q[:, 128:].float(), torch.full((1, 128), torch.finfo(self.dtype).max)))
        torch.testing.assert_close(scale, torch.tensor([[1e-10, 1.]]) / torch.finfo(self.dtype).max)

    def test_compensating_wrong_scale_and_q_are_rejected(self):
        expected = self.oracle.quantize(torch.ones(1, 128), self.dtype)
        wrong = ((expected[0].float() / 2).to(self.dtype), expected[1] * 2)
        torch.testing.assert_close(wrong[0].float() * wrong[1], expected[0].float() * expected[1])
        with self.assertRaises(AssertionError):
            self.oracle.check_quant(wrong, expected)

    def test_reduce_rms_all_five_outputs_have_independent_known_answers(self):
        saved = {'x1': torch.ones(3, 2, 128), 'w1': torch.ones(128),
                 'x2': torch.full((3, 2, 256), 2.), 'w2': torch.full((256,), 2.),
                 'x3': torch.full((3, 2, 64), .5), 'res1': torch.ones(2, 128)}
        outputs = self.oracle.rms(saved, self.dtype, reduce=True)
        self.assertEqual(len(outputs), 5)
        torch.testing.assert_close(outputs[1], torch.ones(2, 128))
        torch.testing.assert_close(outputs[2], torch.full((2, 256), 2.))
        torch.testing.assert_close(outputs[3], torch.full((2, 128), 4.))
        torch.testing.assert_close(outputs[4], torch.full((2, 64), 1.5))
        wrong = (*outputs[:4], torch.zeros_like(outputs[4]))
        with self.assertRaises(AssertionError):
            self.oracle.check_fused(wrong, outputs)

    def test_activation_reduction_uses_both_halves_and_auxiliary(self):
        x = torch.cat((torch.ones(3, 2, 128), torch.full((3, 2, 128), 2.)), dim=-1)
        result, aux = self.oracle.activation_mul({'x': x, 'x2': torch.ones(3, 2, 16)}, self.dtype)
        expected_value = torch.nn.functional.silu(torch.tensor(3.)) * 6
        torch.testing.assert_close(result[0].float() * result[1], torch.full((2, 128), expected_value))
        torch.testing.assert_close(aux, torch.full((2, 16), 3.))

    def test_rms_missing_optional_outputs_stay_none(self):
        out = self.oracle.rms({'x1': torch.ones(2, 128), 'w1': torch.ones(128)}, self.dtype, show=False)
        self.assertEqual(out[1:], (None, None, None))
