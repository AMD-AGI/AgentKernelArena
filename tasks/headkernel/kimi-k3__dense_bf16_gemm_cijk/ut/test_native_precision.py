"""CPU tests for exact packed-pair native exception proofs, not looser tolerances."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ut'))
from native_precision import exact_pair_orders,calibrate,spec_for


class Tests(unittest.TestCase):
    def test_shared_arrival_order_required_for_both_bf16_lanes(self):
        import torch
        parts=torch.tensor([[[1.,256.]],[[256.,1.]], [[-256.,-256.]]],dtype=torch.bfloat16)
        positions=torch.tensor([[0,0]])
        exact_pair_orders(parts,torch.tensor([[0.,0.]],dtype=torch.bfloat16),positions)
        exact_pair_orders(parts,torch.tensor([[1.,1.]],dtype=torch.bfloat16),positions)
        # Each lane separately admits0and1, but no common order produces(0,1).
        with self.assertRaisesRegex(AssertionError,'No exact legal'):exact_pair_orders(parts,torch.tensor([[0.,1.]],dtype=torch.bfloat16),positions)
    def test_exception_is_exact_not_tolerance_band(self):
        import torch
        parts=torch.tensor([[[1.,1.]],[[2.,2.]]],dtype=torch.bfloat16)
        with self.assertRaisesRegex(AssertionError,'No exact legal'):exact_pair_orders(parts,torch.tensor([[3.015625,3.]],dtype=torch.bfloat16),torch.tensor([[0,0]]))
    def test_whole_output_strict_check_and_unmodeled_failure(self):
        import torch
        case={'case_id':'plain','capture_controls':{}}
        expected=torch.ones(2,4,dtype=torch.bfloat16)
        result=calibrate(case,{},expected,expected,check_sources=False)
        self.assertEqual(result['elements_checked'],8);self.assertEqual(result['strict_failure_elements'],0)
        bad=expected.clone();bad[1,3]=100
        with self.assertRaisesRegex(AssertionError,'without a declared precision model'):calibrate(case,{},bad,expected,check_sources=False)
    def test_configs_are_derived_from_recorded_kernel_parameters(self):
        cases=json.loads((ROOT/'cases.json').read_text())['cases']
        case=next(c for c in cases if c['live_fixture']['source_case_key']=='bf16_gemm-84759eb3ff0b4b1d0af0cb6f')
        self.assertEqual(spec_for(case),{'tile_k':64,'split_k':7,'k_warps':1})
if __name__=='__main__':unittest.main(verbosity=2)
