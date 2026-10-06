"""Regression checks for the arithmetic distinction and candidate/native boundary."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from native_precision import candidate_close,round_asm_partial,exact_pair_orders,calibrate


class Tests(unittest.TestCase):
    def test_asm_halfway_conversion_is_not_ties_to_even(self):
        halfway=torch.tensor([1+2**-8,-1-2**-8],dtype=torch.float32)
        self.assertEqual(halfway.bfloat16().float().tolist(),[1.,-1.])
        self.assertEqual(round_asm_partial(halfway).float().tolist(),[1+2**-7,-1-2**-7])

    def test_packed_lanes_require_one_common_arrival_order(self):
        parts=torch.zeros((16,1,2),dtype=torch.bfloat16)
        parts[0]=8.;parts[1]=0.03125;parts[2]=-8.
        # Either scalar result is legal, but identical input lanes cannot differ.
        positions=torch.tensor([[0,0]])
        for value in (0.,0.03125):
            proof=exact_pair_orders(parts,torch.full((1,2),value,dtype=torch.bfloat16),positions,max_orders=128)
            self.assertEqual(proof['pairs_proved_exactly'],1)
            self.assertEqual(sorted(proof['witnesses'][0]['arrival_order']),list(range(16)))
        with self.assertRaisesRegex(AssertionError,'No exact legal ASM'):
            exact_pair_orders(parts,torch.tensor([[0.,0.03125]],dtype=torch.bfloat16),positions,max_orders=128)

    def test_fp32_candidate_never_receives_native_tolerance(self):
        expected=torch.ones((1,2));actual=expected+5e-5
        with self.assertRaises(AssertionError):candidate_close(actual,expected)
        # The unchanged native threshold is a separate calibration policy.
        result=calibrate({'case_id':'fp32','live_fixture':{'capture_family':'bf16_gemm'}},
            {'A':torch.zeros((1,1)),'B':torch.zeros((1,2))},actual,expected)
        self.assertEqual(result['strict_failure_elements'],0)

    def test_unknown_dispatch_and_nonfinite_results_fail(self):
        inputs={'A':torch.zeros((64,4096),dtype=torch.bfloat16),'B':torch.zeros((4096,128),dtype=torch.bfloat16)}
        expected=torch.zeros((64,128),dtype=torch.bfloat16);actual=expected.clone();actual[0,0]=1
        case={'case_id':'bf16','live_fixture':{'capture_family':'bf16_gemm'}}
        with self.assertRaisesRegex(AssertionError,'Unsupported native precision dispatch'):
            calibrate(case,inputs,actual,expected,dispatch={'libtype':'flydsl'},check_sources=False)
        actual[0,0]=float('nan')
        with self.assertRaisesRegex(AssertionError,'Nonfinite'):
            calibrate(case,inputs,actual,expected,check_sources=False)

if __name__=='__main__':unittest.main(verbosity=2)
