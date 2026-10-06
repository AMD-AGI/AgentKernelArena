"""CPU-only typed capture adapter and original native-call binding checks."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'));sys.path.insert(0,str(ROOT/'capture'))
import dense_bindings
import test_native_binding as binding_tests
from source_guard import validate_sources


class Tests(unittest.TestCase):
    def runtime(self):
        import torch
        return types.SimpleNamespace(torch_module=lambda:torch,Role=lambda *a,**k:(a,k),Family=lambda *a,**k:(a,k))
    def test_typed_wrapper_capture_and_original_solution_invocation(self):
        import torch
        native=binding_tests.Tests().native_wrapper();native.__file__=str(ROOT/'ut/native/tuned_gemm.py')
        a=torch.ones(2,128,dtype=torch.bfloat16);b=torch.ones(8,128,dtype=torch.bfloat16)
        arguments={'A':a,'B':b,'bias':None,'otype':torch.bfloat16,'scale_a':None,'scale_b':None,'scale_c':None}
        family,inputs,controls=dense_bindings.make_bindings(self.runtime(),native,arguments)
        self.assertEqual(family[0][0],'bf16_gemm');self.assertEqual(set(inputs),{'A','B'})
        self.assertEqual(controls['otype'],{'kind':'dtype','name':'bfloat16'})
        self.assertEqual(controls['capture_native_dispatch']['libtype'],'torch')
        out=dense_bindings.checked_native_call(native,native.gemm_a16w16,(),arguments,controls['capture_native_dispatch'])
        torch.testing.assert_close(out,torch.full((2,8),128.,dtype=torch.bfloat16),rtol=0,atol=0)
    def test_optional_tensor_is_captured_instead_of_dropped(self):
        import torch
        native=binding_tests.Tests().native_wrapper();native.__file__=str(ROOT/'ut/native/tuned_gemm.py')
        arguments={'A':torch.ones(2,128),'B':torch.ones(8,128),'bias':torch.ones(8),'otype':None,'scale_a':None,'scale_b':None,'scale_c':None}
        _,inputs,controls=dense_bindings.make_bindings(self.runtime(),native,arguments)
        self.assertIn('bias',inputs);self.assertIsNone(controls['otype'])
    def test_aten_supplied_out_binding_preserves_real_buffer(self):
        import torch
        arguments={'A':torch.ones(2,128),'B':torch.ones(128,8),'out':torch.empty(2,8)}
        _,inputs,controls=dense_bindings.make_aten_bindings(self.runtime(),torch,arguments)
        result=torch.mm(arguments['A'],arguments['B'],out=arguments['out'])
        captured=dense_bindings.aten_outputs_for(arguments,result)
        self.assertIs(captured['result'],arguments['out']);self.assertEqual(controls['out'],{'kind':'output_binding','name':'result'})
        self.assertEqual(set(inputs),{'A','B'})
    def test_stock_source_guard(self):validate_sources(ROOT,ROOT)
if __name__=='__main__':unittest.main(verbosity=2)
