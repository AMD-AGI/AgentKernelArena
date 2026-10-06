import importlib.util
from pathlib import Path
import unittest
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import native_precision as n
class Tests(unittest.TestCase):
 def test_nonzero_low_scale_zero_attack_rejected_for_both_dtypes(self):
  for dtype,scale in [(torch.bfloat16,1e-3),(torch.float32,1e-6)]:
   expected=torch.tensor([[scale,-scale]],dtype=dtype);actual=torch.zeros_like(expected)
   with self.assertRaisesRegex(AssertionError,'Scale-relative'):n.candidate_close(actual,expected)
 def test_ratio_is_invariant_under_binary_scaling(self):
  base=torch.tensor([[1.,-2.,4.]],dtype=torch.bfloat16);actual=base+torch.tensor([[0.0078125,0.,0.]],dtype=torch.bfloat16)
  ratios=[n.candidate_error_metrics(actual*2**k,base*2**k)['normalized_l2'] for k in (-10,0,10)]
  self.assertEqual(ratios,[ratios[0]]*3)
 def test_exact_zero_reference_requires_exact_output(self):
  zero=torch.zeros((2,2),dtype=torch.bfloat16);self.assertTrue(n.candidate_close(zero,zero)['scale_relative_pass'])
  with self.assertRaisesRegex(AssertionError,'Scale-relative'):n.candidate_close(zero+1e-4,zero)
 def test_existing_pointwise_check_still_rejects_sparse_corruption(self):
  expected=torch.ones((10000,),dtype=torch.bfloat16);actual=expected.clone();actual[0]=1.5
  self.assertTrue(n.candidate_error_metrics(actual,expected)['scale_relative_pass'])
  with self.assertRaises(AssertionError):n.candidate_close(actual,expected)
if __name__=='__main__':unittest.main(verbosity=2)
