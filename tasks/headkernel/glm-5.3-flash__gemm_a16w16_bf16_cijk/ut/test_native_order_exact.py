"""Regressions against113 independently enumerated legal paired outcomes."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import native_precision
import native_order_exact
from native_precision import exact_pair_orders,NativeOrderProofError

DATA=json.loads((Path(__file__).parent/'data/native_order_seed817302019.json').read_text())
STRESS=json.loads((Path(__file__).parent/'data/native_order_stress.json').read_text())
def bf16(bits):return torch.tensor(bits,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)

class Tests(unittest.TestCase):
    def test_all113_legal_outcomes_include24_missed_pointwise_failures(self):
        rows=DATA['outcomes'];self.assertEqual(len(rows),113)
        self.assertEqual(sum(not r['old_search_reachable'] for r in rows),59)
        self.assertEqual(sum(not r['old_search_reachable'] and bool(r['original_pointwise_failed_lanes']) for r in rows),24)
        count=len(rows);parts=bf16(DATA['partials']).reshape(16,1,2).repeat(1,1,count)
        actual=bf16([r['bits'] for r in rows]).reshape(1,2*count);positions=torch.tensor([[0,i] for i in range(count)])
        proof=exact_pair_orders(parts,actual,positions,max_orders=0)
        self.assertEqual(proof['pairs_proved_exactly'],113);self.assertEqual(len(proof['exact_fallback']),1)
        self.assertEqual(proof['exact_work_budget']['used'],{'groups':1,'states':4171,'transitions':39071})
        self.assertIsNone(proof['exact_work_budget']['exhausted'])
        for witness in proof['witnesses']:self.assertEqual(sorted(witness['arrival_order']),list(range(16)))

    def test_recorded113_outcomes_plus_unreachable_remain4174_states(self):
        model=native_order_exact.ExactPairOrders(DATA['partials'])
        for row in DATA['outcomes']:self.assertIsNotNone(model.solve(row['bits']))
        self.assertIsNone(model.solve(DATA['invalid_shared_pair_bits']))
        self.assertEqual(model.visited,4174)
        self.assertEqual(model.budget.report['used']['transitions'],39119)

    def test_cartesian_predecessor_stress_fails_at_protected_work_limit(self):
        parts=bf16(STRESS['partial_bits']).reshape(16,1,2)
        target=bf16(STRESS['target_bits']).reshape(1,2)
        with self.assertRaises(NativeOrderProofError) as caught:
            exact_pair_orders(parts,target,torch.tensor([[0,0]]),max_orders=0)
        evidence=caught.exception.native_precision_evidence;budget=evidence['work_budget']
        self.assertEqual(evidence['error_type'],'ExactWorkBudgetExceeded')
        self.assertEqual(evidence['failure_stage'],'solve')
        self.assertEqual(budget['exhausted'],'transitions')
        self.assertEqual(budget['used']['transitions'],262144)
        self.assertLess(budget['used']['states'],budget['limits']['states'])
        self.assertEqual(evidence['failed_pairs'][0]['observed_bf16_bits'],STRESS['target_bits'])
        self.assertEqual(evidence['failed_pairs'][0]['partial_bf16_bits'],STRESS['partial_bits'])

    def test_state_and_group_budgets_are_aggregate_across_distinct_groups(self):
        # Both groups have trivial valid orders. A per-group reset would
        # incorrectly accept the second group under either reduced budget.
        parts=torch.zeros((16,1,4),dtype=torch.bfloat16);parts[:,:,2:]=1
        target=torch.tensor([[0.,0.,16.,16.]],dtype=torch.bfloat16)
        positions=torch.tensor([[0,0],[0,1]])
        for name,limit,resource,used_groups in (
                ('MAX_EXACT_STATES',20,'states',2),
                ('MAX_EXACT_GROUPS',1,'groups',1),
                ('MAX_EXACT_TRANSITIONS',40,'transitions',2)):
            with self.subTest(resource=resource),patch.object(native_order_exact,name,limit):
                with self.assertRaises(NativeOrderProofError) as caught:
                    exact_pair_orders(parts,target,positions,max_orders=0)
                evidence=caught.exception.native_precision_evidence;budget=evidence['work_budget']
                self.assertEqual(budget['exhausted'],resource)
                self.assertEqual(budget['used'][resource],limit)
                self.assertEqual(budget['used']['groups'],used_groups)
                self.assertEqual(len(evidence['failed_pairs']),2)

    def test_cached_targets_still_consume_transition_budget(self):
        with patch.object(native_order_exact,'MAX_EXACT_TRANSITIONS',35):
            model=native_order_exact.ExactPairOrders([[0,0]]*16)
        self.assertIsNotNone(model.solve([0,0]))
        states=model.visited
        self.assertIsNotNone(model.solve([0,0]))
        self.assertIsNotNone(model.solve([0,0]))
        with self.assertRaises(native_order_exact.ExactWorkBudgetExceeded):model.solve([0,0])
        self.assertEqual(model.visited,states)
        self.assertEqual(model.budget.report['used']['transitions'],35)

    def test_jointly_unreachable_pair_retains_observed_and_partial_bits(self):
        parts=bf16(DATA['partials']).reshape(16,1,2);actual=bf16(DATA['invalid_shared_pair_bits']).reshape(1,2)
        with self.assertRaises(NativeOrderProofError) as caught:exact_pair_orders(parts,actual,torch.tensor([[0,0]]),max_orders=0)
        evidence=caught.exception.native_precision_evidence;pair=evidence['failed_pairs'][0]
        self.assertEqual(pair['observed_bf16_bits'],DATA['invalid_shared_pair_bits'])
        self.assertEqual(pair['partial_bf16_bits'],DATA['partials'])
        self.assertEqual((pair['row'],pair['pair_column']),(0,0))

    def test_production_failure_receipt_preserves_pair_evidence(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import production_comparison
        evidence={'schema':'native-ASM-exact-order-failure-v1','failed_pairs':[{'row':27,'pair_column':53,'observed_bf16_bits':DATA['invalid_shared_pair_bits'],'partial_bf16_bits':DATA['partials']}]}
        error=NativeOrderProofError(evidence)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(production_comparison.task,'ROOT',Path(directory)),patch.object(production_comparison,'compare',side_effect=error),patch.dict(production_comparison.CONTEXT,{'case_id':DATA['case_id'],'seed':DATA['seed']},clear=True):
                with self.assertRaises(NativeOrderProofError):production_comparison.main()
                saved=json.loads((Path(directory)/'build/native_production_failure.json').read_text())
            self.assertEqual(saved['native_precision_evidence'],evidence)
            self.assertEqual(saved['seed'],DATA['seed'])

    def test_resource_and_witness_failures_preserve_original_bits_through_receipt(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import production_comparison
        inputs={'A':torch.zeros((64,4096),dtype=torch.bfloat16),'B':torch.zeros((4096,128),dtype=torch.bfloat16)}
        parts=torch.zeros((16,64,128),dtype=torch.bfloat16)
        row,pair_column=DATA['pair'];column=2*pair_column
        parts[:,row,column:column+2]=bf16(DATA['partials'])
        expected=torch.zeros((64,128),dtype=torch.bfloat16)
        expected[row,column:column+2]=torch.tensor(DATA['CPU_reference_values'],dtype=torch.bfloat16)
        actual=expected.clone()
        target=next(outcome['bits'] for outcome in DATA['outcomes'] if outcome['original_pointwise_failed_lanes'])
        actual[row,column:column+2]=bf16(target)
        reference_bits=[int(x)&65535 for x in expected[row,column:column+2].view(torch.int16).tolist()]
        case={'case_id':DATA['case_id'],'live_fixture':{'capture_family':'bf16_gemm'}}
        context={'case_id':DATA['case_id'],'live_fixture':case['live_fixture'],'phase':'native_production','seed':DATA['seed'],'challenge_base':408650924}
        dispatch={'libtype':'asm','kernelName':native_precision.KERNEL,'splitK':16}
        def calibrate(request=None):
            return native_precision.calibrate(case,inputs,actual,expected,dispatch=dispatch,check_sources=False)
        faults=(
            ('construction',MemoryError,lambda:patch.object(native_order_exact.ExactPairOrders,'__init__',side_effect=MemoryError('injected construction failure'))),
            ('solve',MemoryError,lambda:patch.object(native_order_exact.ExactPairOrders,'solve',side_effect=MemoryError('injected solve failure'))),
            ('solve',native_order_exact.ExactWorkBudgetExceeded,lambda:patch.object(native_order_exact,'MAX_EXACT_TRANSITIONS',0)),
            ('witness_replay',MemoryError,lambda:patch.object(torch,'zeros',side_effect=MemoryError('injected replay failure'))),
            ('witness_replay',AssertionError,lambda:patch.object(native_order_exact.ExactPairOrders,'solve',return_value=[0]*16)),
            ('witness_replay',AssertionError,lambda:patch.object(native_order_exact.ExactPairOrders,'solve',return_value=list(range(16)))),
        )
        for stage,error_type,fault in faults:
            with self.subTest(stage=stage,error=error_type.__name__),tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
                stack.enter_context(patch.object(production_comparison.task,'ROOT',Path(directory)))
                stack.enter_context(patch.object(production_comparison,'compare',side_effect=calibrate))
                stack.enter_context(patch.dict(production_comparison.CONTEXT,context,clear=True))
                stack.enter_context(patch.object(native_precision,'asm_partials',return_value=parts))
                stack.enter_context(patch.object(native_precision,'exact_pair_orders',side_effect=lambda *args:exact_pair_orders(*args,max_orders=0)))
                stack.enter_context(fault())
                with self.assertRaises(NativeOrderProofError) as caught:production_comparison.main()
                self.assertIsInstance(caught.exception.__cause__,error_type)
                saved=json.loads((Path(directory)/'build/native_production_failure.json').read_text())
                for key,value in context.items():self.assertEqual(saved[key],value)
                evidence=saved['native_precision_evidence'];pair=evidence['failed_pairs'][0]
                self.assertEqual(evidence['failure_stage'],stage)
                self.assertEqual(evidence['error_type'],error_type.__name__)
                self.assertEqual(pair['observed_bf16_bits'],target)
                self.assertEqual(pair['partial_bf16_bits'],DATA['partials'])
                self.assertEqual(pair['CPU_reference_bf16_bits'],reference_bits)
                self.assertEqual((pair['row'],pair['pair_column']),tuple(DATA['pair']))

if __name__=='__main__':unittest.main(verbosity=2)
