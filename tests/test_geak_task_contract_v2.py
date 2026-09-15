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


class AutogradOutputStorageTests(unittest.TestCase):
    def test_timing_handle_preserves_custom_view_storage_without_autograd_replay_error(self):
        class Forward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return (x * 3).view_as(x)
        original = []
        def public(layernorm, x, gemma):
            self.assertFalse(gemma)
            original.append(Forward.apply(x))
            return original[-1]
        wrapped = function(TASKS / 'L2/fast_rms_layernorm/test_kernel_harness.py',
                           '_forward_for_timing', fast_rms_layernorm=public)
        x = torch.ones(2, requires_grad=True)
        output = wrapped(None, x)
        self.assertEqual(output.data_ptr(), original[0].data_ptr())
        self.assertFalse(output.requires_grad)
        with torch.no_grad():
            output.fill_(float('nan'))
            output.copy_(torch.full((2,), 3.))
        torch.testing.assert_close(output, torch.full((2,), 3.))
        # The original autograd view becomes invalid for arithmetic after this
        # legitimate storage reuse, reproducing the reported GPU guard error.
        with self.assertRaises(RuntimeError):
            original[0] - 3


class DiscreteRoundingContractTests(unittest.TestCase):
    def test_fp8_adjacent_codes_and_two_step_error(self):
        oracle = load_file('_geak_fp8_step_oracle', TASKS / 'L3/fused_rms_fp8/_contract_oracles.py')
        dtype = torch.float8_e4m3fn
        ref = (torch.full((1, 128), 256., dtype=dtype), torch.full((1, 1), .001))
        # Adjacent codes 256/288 straddle the 272 midpoint; reconstructed
        # values still satisfy the unchanged numerical gate.
        oracle.check_quant((torch.full((1, 128), 288., dtype=dtype), ref[1]), ref)
        with self.assertRaisesRegex(AssertionError, 'more than one'):
            oracle.check_quant((torch.full((1, 128), 320., dtype=dtype), ref[1]), ref)

    def test_only_possible_first_saturated_expert_is_admissible(self):
        check = function(TASKS / 'L1/moe_routing_sigmoid_top1/test_kernel_harness.py', '_check_timed_routing')
        lower = torch.tensor([[.9, .99999988, .95, 1., 1.]])
        upper = torch.tensor([[.9, 1., .95, 1., 1.]])
        weights = torch.ones(1, 2)
        expected = (torch.tensor([[3, 5]], dtype=torch.int32), weights)
        check((torch.tensor([[1, 5]], dtype=torch.int32), weights), expected, lower, upper)
        for wrong in (0, 2, 4, 5):
            with self.assertRaises(AssertionError):
                check((torch.tensor([[wrong, 5]], dtype=torch.int32), weights), expected, lower, upper)

    def test_unsaturated_routing_and_original_integer_ids_stay_exact(self):
        path = TASKS / 'L1/moe_routing_sigmoid_top1/test_kernel_harness.py'
        check = function(path, '_check_timed_routing')
        scores = torch.tensor([[.5, .5, .4]])
        weights = torch.tensor([[.5, 1.]])
        expected = (torch.tensor([[0, 3]], dtype=torch.int32), weights)
        wrong = (torch.tensor([[1, 3]], dtype=torch.int32), weights)
        with self.assertRaises(AssertionError):
            check(wrong, expected, scores, scores)
        with self.assertRaises(AssertionError):
            function(path, '_check_routing')(wrong, expected)


class MLACompleteContractTests(unittest.TestCase):
    def test_single_finite_corruption_cannot_use_legacy_five_percent_allowance(self):
        path = TASKS / 'L1/mla_decode/test_kernel_harness.py'
        reference = function(path, 'run_ref')
        legacy = function(path, 'check_correctness_val', assert_output_contract=CONTRACT.assert_output_contract)
        rounding = load_file('_geak_mla_rounding', TASKS / 'L1/mla_decode/_rounding_reference.py')
        build = function(path, '_mla_contract', KV_LORA_RANK=2, run_ref=reference, check_correctness_val=legacy,
                         attention_rounding_bounds=rounding.attention_rounding_bounds,
                         check_rounding_bounds=rounding.check_rounding_bounds)
        inputs = dict(q=torch.zeros(1, 50, 3), k_input=torch.zeros(3, 1, 3),
                      v_input=torch.ones(3, 1, 2), kv_indices=torch.arange(3),
                      output=torch.empty(1, 50, 2), sm_scale=.5, num_kv_splits=2,
                      attn_logits=torch.empty(1, 50, 2, 3, dtype=torch.bfloat16))
        readonly, ref, check = build(inputs)
        expected = ref(readonly)
        wrong = expected.clone(); wrong.flatten()[0] -= .1
        self.assertTrue(legacy(expected, wrong)[0])
        with self.assertRaises(AssertionError):
            check(wrong, expected)

    def test_bf16_partition_rounding_is_enclosed_without_excluding_ideal(self):
        rounding = load_file('_geak_mla_rounding_samples', TASKS / 'L1/mla_decode/_rounding_reference.py')
        generator = torch.Generator().manual_seed(317)
        means = torch.randn(100, 2, 7, generator=generator)
        absolute_means = means.abs() + .25
        log_sums = torch.randn(100, 2, generator=generator)*3
        lower, upper = rounding.split_rounding_bounds(means, absolute_means, log_sums)
        ideal = (means*torch.softmax(log_sums, dim=-1).unsqueeze(-1)).sum(-2)
        # Sample actual BF16 stores and both extremal probability-dot errors.
        for sign in (-1, 0, 1):
            partials = (means + sign*2**-8*absolute_means).bfloat16().float()
            weights = torch.softmax(log_sums.bfloat16().float(), dim=-1)
            actual = (partials*weights.unsqueeze(-1)).sum(-2).bfloat16().float()
            self.assertTrue((actual >= lower).all())
            self.assertTrue((actual <= upper).all())
        rounding.check_rounding_bounds(ideal, ideal, lower, upper)
        for index in (0, ideal.numel()-1):
            bad = ideal.clone()
            bad.flatten()[index] = upper.flatten()[index] + .1 + .01*ideal.flatten()[index].abs()
            with self.assertRaisesRegex(AssertionError, 'rounding bound'):
                rounding.check_rounding_bounds(bad, ideal, lower, upper)

    def test_dependency_identity_repeatability_and_collision(self):
        import hashlib
        import json
        import tempfile
        import types
        module = load_file('_geak_mla_dependency', TASKS / 'L1/mla_decode/_aiter_dependency.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / 'dependencies/aiter_triton'
            dependency.mkdir(parents=True)
            source = dependency / 'primitive.py'; source.write_text('VALUE = 7\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (root / 'runtime-dependencies.json').write_text(json.dumps({'source_root': 'dependencies/aiter_triton', 'files': {'primitive.py': digest}}))
            module.ROOT = root
            self.assertEqual(module.verify_dependency(), dependency)
            self.assertEqual(module.verify_dependency(), dependency)
            source.write_text('VALUE = 8\n')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                module.verify_dependency()
            source.write_text('VALUE = 7\n')
            with patch.dict(sys.modules, {'aiter': types.ModuleType('aiter')}):
                with self.assertRaisesRegex(RuntimeError, 'outside this task'):
                    module.bind_dependency()
            with patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name == 'aiter' or name.startswith('aiter.'):
                        del sys.modules[name]
                module.bind_dependency()
                before = sys.modules['aiter.ops.triton']
                module.bind_dependency()
                self.assertIs(before, sys.modules['aiter.ops.triton'])
                self.assertEqual(before.__path__, [str(dependency)])

    def test_failure_reports_only_completed_case_outcomes_and_honest_kind(self):
        from copy import deepcopy
        from dataclasses import replace
        import json
        import tempfile
        from types import SimpleNamespace
        from src.task_protocol import CaseManifest, baseline_correctness_accepted, parse_command_result
        from src.task_spec import load_task_spec
        runner = load_file('_geak_mla_runner', TASKS / 'L1/mla_decode/_arena_eval.py')
        baseline = load_task_spec(TASKS / 'L1/mla_decode/config.yaml',
                                  task_id='triton2triton/geak_eval/L1/mla_decode').baseline
        self.assertEqual(baseline.correctness_policy, 'required')
        diagnostic = replace(baseline, correctness_policy='diagnostic', diagnostic_reason='Test-only policy')
        cases = [{'test_case_id': f'case/{i}', 'checks': ['correctness', 'performance'], 'params': {}} for i in range(2)]
        outcomes = [{'test_case_id': 'case/0', 'status': 'PASS'},
                    {'test_case_id': 'case/1', 'status': 'FAIL', 'reason': 'outside bound',
                     'failure_kind': 'numerical_mismatch'}]
        actions = SimpleNamespace(inputs=lambda: {}, validate=lambda: None,
                                  correctness=lambda require: outcomes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'workloads.json').write_text(json.dumps({'cases': cases, 'input_tables': {}}))
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'load_actions', return_value=actions), \
                    patch.object(runner, 'inspect_candidate', return_value='implemented'):
                manifest = CaseManifest.from_result(parse_command_result(
                    'ARENA_EVAL_RESULT='+json.dumps(runner.evaluate('task', 'validate-task')),
                    role='task', action='validate-task', returncode=0))
                result = runner.evaluate('baseline', 'correctness')
                self.assertEqual([r['status'] for r in result['cases']], ['PASS', 'FAIL'])
                self.assertEqual(result['failure_kind'], 'numerical_mismatch')
                self.assertEqual(result['cases'][1]['failure_kind'], 'numerical_mismatch')
                def accepted(payload, policy=diagnostic, phase='task_validation'):
                    parsed = parse_command_result('ARENA_EVAL_RESULT='+json.dumps(payload), role='baseline',
                                                  action='correctness', returncode=1)
                    return baseline_correctness_accepted(parsed, baseline=policy, phase=phase, manifest=manifest)
                self.assertTrue(accepted(result))
                self.assertFalse(accepted(result, baseline))
                self.assertFalse(accepted(result, phase='candidate_evaluation'))
                nested = deepcopy(result)
                nested['cases'][1]['metadata'] = {'failure_kind': nested['cases'][1].pop('failure_kind')}
                self.assertFalse(accepted(nested))
                outcomes[0].update(status='FAIL', reason='launch failed', failure_kind='execution_failure')
                mixed = runner.evaluate('baseline', 'correctness')
                self.assertEqual(mixed['failure_kind'], 'execution_failure')
                self.assertFalse(accepted(mixed))
                mixed['failure_kind'] = 'numerical_mismatch'
                self.assertFalse(accepted(mixed))  # Per-case kinds must also be purely numerical.
                def fail_before_results(require):
                    raise ImportError('missing runtime implementation')
                actions.correctness = fail_before_results
                result = runner.evaluate('baseline', 'correctness')
                self.assertEqual(result['cases'], [])
                self.assertEqual(result['failure_kind'], 'execution_failure')
                parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result), role='baseline',
                                     action='correctness', returncode=1)

    def test_harness_and_control_failures_publish_top_level_kind(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        rounding = load_file('_geak_mla_failure_type', TASKS / 'L1/mla_decode/_rounding_reference.py')
        collect = function(
            TASKS / 'L1/mla_decode/test_kernel_harness.py', 'mode_correctness',
            ALL_CONFIGS=[(21, 1, 16)]*3, config_str=str, setup_inputs=lambda *a: {},
            _mla_contract=lambda inputs: ({}, None, None), run_ref=lambda inputs: torch.ones(1),
            check_correctness_val=lambda ref, actual: (False, .1, .2),
            NumericalMismatch=rounding.NumericalMismatch,
            checked_call=Mock(side_effect=[rounding.NumericalMismatch('outside bound'),
                                          RuntimeError('launch failed'), torch.ones(1)]))
        with patch.object(torch.cuda, 'empty_cache'):
            outcomes = collect([0, 1, 2], collect=True)
        self.assertEqual([row['failure_kind'] for row in outcomes],
                         ['numerical_mismatch', 'execution_failure', 'numerical_mismatch'])
        self.assertTrue(all('metadata' not in row for row in outcomes))
        h = SimpleNamespace(ALL_CONFIGS=[], _pick=lambda *a: [], mode_correctness=lambda *a, **k: [],
                            CONTROL_CASES=[{'test_case_id': 'control'}],
                            run_contract_controls=Mock(side_effect=RuntimeError('control setup failed')))
        control = function(TASKS / 'L1/mla_decode/_arena_actions.py', 'correctness', h=h)(lambda *a: None)
        self.assertEqual(control[0]['failure_kind'], 'execution_failure')
        self.assertNotIn('metadata', control[0])


if __name__ == '__main__':
    unittest.main()
