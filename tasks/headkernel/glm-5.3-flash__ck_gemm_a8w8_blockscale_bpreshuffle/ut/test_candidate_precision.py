import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_precision',ROOT/'ut/candidate_precision.py')
precision=importlib.util.module_from_spec(spec);sys.modules[spec.name]=precision;spec.loader.exec_module(precision)


def candidate_close(actual,expected):
    torch.testing.assert_close(actual,expected,rtol=0.01,atol=0.02)
    return precision.require_candidate_accuracy(actual,expected)


class CandidateScaleTests(unittest.TestCase):
    def test_actual_verifier_integrates_with_canonical_checked_replays(self):
        contract_spec=importlib.util.spec_from_file_location('fp8_replay_contract',ROOT/'ut/evaluation_contract.py')
        contract=importlib.util.module_from_spec(contract_spec);contract_spec.loader.exec_module(contract)
        tree=ast.parse((ROOT/'scripts/task_runner.py').read_text())
        node=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='verify')
        tensors={'A':torch.ones(4,dtype=torch.bfloat16),'C':torch.empty(4,dtype=torch.bfloat16)}
        namespace={'torch':torch,'tensors':tensors}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_verifier','exec'),namespace)
        verifier=namespace['verify'];state={};case={'case_id':'cpu-callback-contract','calls_per_sample':1}
        policy={'method':'cuda_graph','warmup_iterations':2,'benchmark_iterations':3}
        def reset(seed):
            tensors['A'].fill_(seed+1)
            state['expected']=(tensors['A']*1e-4).clone()
            return state['expected'],{'A':tensors['A'].clone()}
        def initialize():tensors['C'].fill_(float('nan'))
        def replay():tensors['C'].copy_(state['expected'])
        def measure(call):call();return 0.5  # CPU callback-contract test, not device timing evidence.
        with patch.object(torch.cuda,'synchronize',lambda:None):
            with patch.object(precision,'require_candidate_accuracy',wraps=precision.require_candidate_accuracy) as accuracy:
                result=contract.checked_replays(case,policy,reset_inputs=reset,initialize_outputs=initialize,
                    replay=replay,verify=verifier,measure=measure,observe=lambda:case,seed=0)
                self.assertEqual(result['samples_ms'],[0.5]*3)
                self.assertEqual(accuracy.call_count,5)
                self.assertIsNone(verifier((state['expected'],{'A':tensors['A'].clone()})))
            with self.assertRaisesRegex(AssertionError,'Scale-relative'):
                contract.checked_replays(case,policy,reset_inputs=reset,initialize_outputs=initialize,
                    replay=lambda:tensors['C'].zero_(),verify=verifier,measure=measure,observe=lambda:case,seed=0)

    def test_scale_disproportionate_errors_fail_without_a_case_specific_rule(self):
        for scale in (2.0**-8,1.0,2.0**8):
            expected=torch.full((4,4),1e-4*scale,dtype=torch.bfloat16)
            for actual in (torch.zeros_like(expected),expected*0.5,torch.full_like(expected,0.002*scale)):
                with self.assertRaises(AssertionError):candidate_close(actual,expected)
            self.assertTrue(candidate_close(expected,expected)['scale_relative_pass'])

    def test_pointwise_check_still_rejects_a_diluted_local_error(self):
        expected=torch.ones(10000,dtype=torch.bfloat16);actual=expected.clone();actual[0]=1.05
        self.assertTrue(precision.candidate_error_metrics(actual,expected)['scale_relative_pass'])
        with self.assertRaises(AssertionError):candidate_close(actual,expected)

    def test_zero_reference_requires_exact_zero(self):
        expected=torch.zeros(4,dtype=torch.bfloat16)
        self.assertTrue(candidate_close(expected,expected)['scale_relative_pass'])
        with self.assertRaisesRegex(AssertionError,'Scale-relative'):
            candidate_close(expected+1e-4,expected)

    def test_native_verifier_keeps_original_pointwise_behavior(self):
        tree=ast.parse((ROOT/'scripts/task_runner.py').read_text())
        node=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='verify')
        namespace={'torch':torch,'tensors':{'A':torch.ones(4,dtype=torch.bfloat16),'C':torch.zeros(4,dtype=torch.bfloat16)}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'verification_test','exec'),namespace)
        expected=torch.full((4,),1e-4,dtype=torch.bfloat16);truth=(expected,{'A':namespace['tensors']['A'].clone()})
        with patch.object(torch.cuda,'synchronize',lambda:None):
            namespace['verify'](truth,native=True)
            with self.assertRaisesRegex(AssertionError,'Scale-relative'):
                namespace['verify'](truth)

    def test_all_mandatory_cases_and_extra_seed_policy_are_preserved(self):
        import json
        manifest=json.loads((ROOT/'cases.json').read_text())
        self.assertEqual(len(manifest['cases']),14)
        self.assertEqual(manifest['measurement']['correctness_seeds'],[0,1])
        self.assertEqual(json.loads((ROOT/'provenance/CANDIDATE-PRECISION.json').read_text())['GPU_evidence']['seeds'],[0,1,3,43])
        self.assertEqual(precision.candidate_error_metrics(torch.ones(1,dtype=torch.bfloat16),torch.ones(1,dtype=torch.bfloat16))['normalized_l2_limit'],2/255)


if __name__=='__main__':unittest.main()
