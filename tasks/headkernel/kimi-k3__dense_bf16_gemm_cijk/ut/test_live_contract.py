"""CPU checks for final dense/FP8 intake and meaningful numerical live refresh."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'));sys.path.insert(0,str(ROOT/'scripts'))
from import_live_operands import comparable_bindings
import live_operands


class Tests(unittest.TestCase):
    def fixture(self,dtype='float32'):
        def meta(shape,stride,alias):return {'shape':shape,'stride':stride,'storage_offset':0,'dtype':'torch.'+dtype,'alias':alias}
        return {'family':'bf16_gemm','controls':{'bias':None,'scale_a':None,'scale_b':None,'scale_c':None,'otype':{'kind':'dtype','name':dtype},'tensor_attributes':{'A':{},'B':{}}},
                'inputs':{'A':meta([64,128],[128,1],'a'),'B':meta([32,128],[128,1],'b')},'outputs':{'result':meta([64,32],[32,1],'c')}}
    def test_fp32_dense_capture_remains_fp32(self):
        f=self.fixture();b=comparable_bindings(f)
        self.assertEqual(b['A']['dtype'],'torch.float32');self.assertEqual(b['C']['dtype'],'torch.float32')
        self.assertEqual(b['B']['shape'],[128,32]);self.assertEqual(b['B']['stride'],[1,128]);self.assertEqual(b['B']['alias'],'b')
    def test_unknown_controls_and_aliasing_fail_instead_of_filtering(self):
        for mutate in [lambda f:f['controls'].update(bias=1),lambda f:f['inputs']['B'].update(alias='a'),lambda f:f['controls']['tensor_attributes']['B'].update(is_shuffled=True)]:
            f=self.fixture();mutate(f)
            with self.assertRaises(ValueError):comparable_bindings(f)
    def test_direct_aten_supplied_output_contract_is_preserved(self):
        f=self.fixture('bfloat16');f['family']='aten_bf16_mm';f['inputs']['B']['shape']=[128,32];f['inputs']['B']['stride']=[1,128]
        f['controls']={'out':{'kind':'output_binding','name':'result'},'tensor_attributes':{'A':{},'B':{}}}
        mapped=comparable_bindings(f);self.assertEqual(mapped['C']['shape'],[64,32])
        self.assertEqual(mapped['B']['stride'],[1,128])
    def test_nonzero_offset_preserved(self):
        f=self.fixture();f['inputs']['A']['storage_offset']=128
        self.assertEqual(comparable_bindings(f)['A']['storage_offset'],128)
    def test_numeric_dense_refresh_and_dtype(self):
        import torch
        for dtype in (torch.bfloat16,torch.float32):
            a=torch.arange(128).reshape(4,32).to(dtype)+1;b=torch.randn(8,32).to(dtype).t();values={'A':a,'B':b}
            with patch.object(live_operands,'load',return_value=(values,None)):
                case={'live_fixture':{'path':'unused','sha256':'unused'}}
                one,want=live_operands.generate_live(case,1);again,want_again=live_operands.generate_live(case,1);two,_=live_operands.generate_live(case,3)
            self.assertEqual(one['A'].dtype,dtype);self.assertEqual(want.dtype,dtype);self.assertEqual(want.device.type,'cpu')
            self.assertTrue(torch.equal(one['A'],again['A']));self.assertFalse(torch.equal(one['A'],two['A']))
            self.assertFalse(torch.equal(one['A'].sort(dim=0).values,a.sort(dim=0).values))
            torch.testing.assert_close(want,(one['A'].float()@b.float()).to(dtype),rtol=0,atol=0)
    def test_fp8_numeric_refresh_preserves_scale_stride_and_weight_bytes(self):
        import torch
        g=torch.Generator().manual_seed(5);a=(torch.randn(4,128,generator=g)*64).to(torch.float8_e4m3fn);w=(torch.randn(128,128,generator=g)*64).to(torch.float8_e4m3fn)
        b=w.reshape(8,16,4,2,16).permute(0,2,3,1,4).contiguous().reshape(128,128)
        values={'A':a,'B':b,'SA':torch.ones(1,4).t()*0.01,'SB':torch.ones(1,1)*0.001}
        with patch.object(live_operands,'load',return_value=(values,None)):
            case={'live_fixture':{'path':'unused','sha256':'unused'}}
            fresh,want=live_operands.generate_live(case,1);other,_=live_operands.generate_live(case,3)
        self.assertEqual(fresh['SA'].stride(),(1,4));self.assertTrue(torch.equal(fresh['B'].view(torch.uint8),b.view(torch.uint8)))
        self.assertFalse(torch.equal(fresh['A'].view(torch.uint8),other['A'].view(torch.uint8)))
        self.assertEqual(want.device.type,'cpu');self.assertTrue(torch.isfinite(want).all())
if __name__=='__main__':unittest.main(verbosity=2)
