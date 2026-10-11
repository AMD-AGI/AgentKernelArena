"""CPU checks of protected vendor-reference preparation and its independent gate.

No GPU kernel runs here. The collector test substitutes a CPU callable for the
pinned vendor to verify masking/ABI/snapshot behavior, not GPU arithmetic.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import native_precision as precision

CASE={'case_id':'native-split-test','live_fixture':{'capture_family':'bf16_gemm'}}
DISPATCH={'libtype':'asm','kernelName':precision.KERNEL,'splitK':16}


def inputs_and_partials():
    # Three exact partition sums demonstrate a legal native result that is not
    # within the candidate's unchanged mathematical tolerance.
    a=torch.zeros((64,4096),dtype=torch.bfloat16)
    b=torch.zeros((128,4096),dtype=torch.bfloat16).t()
    parts=torch.zeros((16,64,128),dtype=torch.bfloat16)
    for split,value in enumerate((8.0,0.03125,-8.0)):
        a[0,split*256]=value;b[split*256,:2]=1
        parts[split,0,:2]=value
    return {'A':a,'B':b},parts


class Tests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def prepare(self,inputs,parts):
        with patch.object(precision,'verify_native_sources') as source_check, \
             patch.object(precision,'dispatch_for',return_value=DISPATCH), \
             patch.object(precision,'_collect_native_partials',return_value=parts) as collect:
            result=precision.prepare_native_reference(CASE,inputs)
            source_check.assert_called_once();collect.assert_called_once_with(inputs)
            return result

    def test_reference_checks_all_partitions_and_never_expands_candidate_tolerance(self):
        inputs,parts=inputs_and_partials();reference=self.prepare(inputs,parts)
        self.assertEqual(len(reference['evidence']['CPU_FP64_partial_checks']),16)
        self.assertTrue(all(row['scale_relative_pass'] for row in reference['evidence']['CPU_FP64_partial_checks']))
        expected=(inputs['A'].float()@inputs['B'].float()).bfloat16()
        actual=torch.zeros_like(expected)
        with self.assertRaises(AssertionError):precision.candidate_close(actual,expected)
        result=precision.calibrate(CASE,inputs,actual,expected,dispatch=DISPATCH,
            check_sources=False,native_reference=reference)
        self.assertEqual(result['exact_exception_pairs'],1)
        witness=result['proof']['witnesses'][0]
        self.assertEqual(sorted(witness['arrival_order']),list(range(16)))
        self.assertEqual(len(witness['partial_bf16_bits']),16)
        self.assertFalse(result['native_partial_reference']['GPU_truth_retained'])

    def test_wrong_or_nonfinite_vendor_partials_fail_independent_math(self):
        inputs,parts=inputs_and_partials()
        for corrupt in (0.0,float('nan'),100.0):
            with self.subTest(corrupt=corrupt):
                wrong=parts.clone();wrong[0,0,0]=corrupt
                with self.assertRaisesRegex(AssertionError,'independent CPU mathematics: split 0'):
                    self.prepare(inputs,wrong)

    def test_reference_cannot_be_reused_for_different_input_bytes(self):
        inputs,parts=inputs_and_partials();reference=self.prepare(inputs,parts)
        expected=(inputs['A'].float()@inputs['B'].float()).bfloat16();actual=torch.zeros_like(expected)
        inputs['A'][2,5]=1
        with self.assertRaisesRegex(AssertionError,'different input bytes'):
            precision.calibrate(CASE,inputs,actual,expected,dispatch=DISPATCH,
                check_sources=False,native_reference=reference)

    def test_source_dispatch_or_ABI_failure_prevents_vendor_invocation(self):
        inputs,_=inputs_and_partials()
        with patch.object(precision,'_collect_native_partials') as collect:
            with patch.object(precision,'verify_native_sources',side_effect=ValueError('pin changed')):
                with self.assertRaisesRegex(ValueError,'pin changed'):precision.prepare_native_reference(CASE,inputs)
            with patch.object(precision,'verify_native_sources'),patch.object(precision,'dispatch_for',return_value={'libtype':'other'}):
                with self.assertRaisesRegex(AssertionError,'Unsupported native split reference dispatch'):
                    precision.prepare_native_reference(CASE,inputs)
            bad=dict(inputs,B=inputs['B'].contiguous())
            with self.assertRaisesRegex(AssertionError,'input ABI'):
                precision.prepare_native_reference(CASE,bad)
            collect.assert_not_called()

    def test_other_native_families_do_not_gain_this_exception(self):
        inputs,_=inputs_and_partials()
        with patch.object(precision,'_collect_native_partials') as collect:
            self.assertIsNone(precision.prepare_native_reference(
                {'live_fixture':{'capture_family':'aten_bf16_mm'}},inputs))
            self.assertIsNone(precision.prepare_native_reference(CASE,{k:v.float() for k,v in inputs.items()}))
            collect.assert_not_called()

    def test_collector_uses_exact_masked_inputs_and_snapshots_each_output(self):
        inputs,expected=inputs_and_partials();calls=[];outputs=[]
        def vendor(a,weights,**kwargs):
            split=len(calls);masked=torch.zeros_like(inputs['A'])
            masked[:,split*256:(split+1)*256]=inputs['A'][:,split*256:(split+1)*256]
            self.assertTrue(torch.equal(a,masked));self.assertTrue(torch.equal(weights,inputs['B'].t()))
            self.assertEqual(kwargs,{'bias':None,'otype':torch.bfloat16,'scale_a':None,'scale_b':None,'scale_c':None})
            calls.append(split);out=(a.float()@weights.float().t()).bfloat16();outputs.append(out);return out
        allocation=torch.empty_strided
        def cpu_allocation(shape,stride,**kwargs):
            self.assertEqual(kwargs.pop('device'),'cuda')
            return allocation(shape,stride,device='cpu',**kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(precision,'importlib',SimpleNamespace(import_module=lambda _:SimpleNamespace(gemm_a16w16=vendor))))
            stack.enter_context(patch.object(torch,'empty_strided',side_effect=cpu_allocation))
            stack.enter_context(patch.object(torch.version,'hip','CPU test double'))
            stack.enter_context(patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(gcnArchName='gfx950')))
            stack.enter_context(patch.object(torch.cuda,'synchronize'))
            actual=precision._collect_native_partials(inputs)
        self.assertEqual(calls,list(range(16)));self.assertTrue(torch.equal(actual,expected))
        for output in outputs:
            self.assertTrue(bool(torch.isnan(output).all()),'Reference answers survived in a reusable output')
            output.fill_(999)
        self.assertTrue(torch.equal(actual,expected),'Reference retained an alias of a vendor output')


if __name__=='__main__':unittest.main(verbosity=2)
