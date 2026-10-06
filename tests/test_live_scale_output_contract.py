"""Real CPU FP8/E8M0 tensors: ignore only unwritten padding, never live outputs."""
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest('CPU PyTorch is required for the live output contract') from exc

if not hasattr(torch, 'float8_e8m0fnu'):
    raise unittest.SkipTest('PyTorch with E8M0 support is required')

TASK=Path(__file__).resolve().parents[1]/'tasks/headkernel/deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl'
sys.path.insert(0,str(TASK/'ut'))
spec=importlib.util.spec_from_file_location('live_scale_runner',TASK/'scripts/task_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
from output_contract import live_scale_domain,compare_stage1_outputs,comparison_inputs_from_snapshot


class LiveScaleTests(unittest.TestCase):
    def values(self):
        ids=torch.full((64,),2,dtype=torch.int32)
        for row,value in [(0,0),(16,1),(32,1<<24),(48,(1<<24)|1)]:ids[row]=value
        inputs={'a':torch.empty((2,64)), 'w1':torch.empty((2,768,32)), 'topk':2,'tile_m':32,
                'sorted_token_ids':ids,'sorted_expert_ids':torch.zeros(2,dtype=torch.int32),
                'num_valid_ids':torch.tensor([64,0],dtype=torch.int32)}
        payload=torch.ones((2,2,384)).to(torch.float8_e4m3fn)
        golden_scale=torch.empty((256,16),dtype=torch.float8_e8m0fnu)
        golden_scale.view(torch.uint8).fill_(255)
        actual_scale=torch.empty_like(golden_scale);actual_scale.view(torch.uint8).fill_(0)
        offsets,pairs,proof=live_scale_domain(inputs,payload,golden_scale)
        golden_scale.view(torch.uint8).reshape(-1)[offsets]=127
        actual_scale.view(torch.uint8).reshape(-1)[offsets]=127
        return inputs,(payload.clone(),actual_scale),(payload,golden_scale),offsets,proof

    def test_unwritten_padding_differences_do_not_fail_native_outputs(self):
        inputs,actual,golden,offsets,proof=self.values()
        result=compare_stage1_outputs(actual,golden,inputs,0.02,runner.compare)
        self.assertEqual(result['meaningful_scale_bytes'],48)
        self.assertEqual(result['unwritten_scale_padding_bytes'],4096-48)
        self.assertTrue(result['comparison_passed'])
        # Independent native layout landmarks (32-row, 8-column byte tiling).
        self.assertTrue({0,1,256,257,512,513,768,769}<=set(offsets.tolist()))

    def test_one_live_scale_byte_difference_fails_exactly(self):
        inputs,actual,golden,offsets,_=self.values()
        actual[1].view(torch.uint8).reshape(-1)[offsets[1]]=126
        with self.assertRaisesRegex(AssertionError,'live E8M0 bytes differ'):
            compare_stage1_outputs(actual,golden,inputs,0.02,runner.compare)

    def test_live_nan_code_fails(self):
        inputs,actual,golden,offsets,_=self.values()
        actual[1].view(torch.uint8).reshape(-1)[offsets[0]]=255
        with self.assertRaisesRegex(AssertionError,'nonfinite live E8M0'):
            compare_stage1_outputs(actual,golden,inputs,0.02,runner.compare)

    def test_full_dense_payload_still_checked(self):
        inputs,actual,golden,_,_=self.values()
        actual[0].view(torch.uint8).zero_()
        with self.assertRaisesRegex(AssertionError,'values differ'):
            compare_stage1_outputs(actual,golden,inputs,0.02,runner.compare)

    def test_missing_or_duplicate_routes_are_contract_errors(self):
        inputs,actual,golden,_,_=self.values()
        inputs['sorted_token_ids'][16]=0
        with self.assertRaisesRegex(ValueError,'exactly one'):
            compare_stage1_outputs(actual,golden,inputs,0.02,runner.compare)

    def test_invalid_reference_is_not_a_negative_control_oracle(self):
        inputs,actual,golden,offsets,_=self.values()
        golden[1].view(torch.uint8).reshape(-1)[offsets[0]]=255
        with self.assertRaisesRegex(ValueError,'reference has nonfinite'):
            compare_stage1_outputs(actual,golden,inputs,0.02,runner.compare)

    def test_canonical_runner_dispatches_the_live_contract(self):
        inputs,actual,golden,_,_=self.values()
        truth=runner.storage_snapshots(inputs)
        runner.compare_native_outputs(actual,golden,inputs,0.02,expected_inputs=truth)
        self.assertTrue(runner.OUTPUT_CONTRACT_PROOFS)
        self.assertTrue(all(p['comparison_passed'] for p in runner.OUTPUT_CONTRACT_PROOFS.values()))

    def test_unexpected_scale_allocation_rejected(self):
        inputs,actual,golden,_,_=self.values()
        bad=(golden[0],golden[1][:128].clone())
        with self.assertRaisesRegex(ValueError,'allocation differs'):
            compare_stage1_outputs(bad,bad,inputs,0.02,runner.compare)

    def test_actual_decode_extent_checks_all_4608_live_scale_bytes(self):
        ids=torch.full((12800,),64,dtype=torch.int32)
        tokens=torch.arange(64,dtype=torch.int32).repeat_interleave(6)
        slots=torch.arange(6,dtype=torch.int32).repeat(64)
        ids[:384]=tokens|(slots<<24)
        inputs={'a':torch.empty((64,1)),'w1':torch.empty((512,768,1)),'topk':6,'tile_m':32,
                'sorted_token_ids':ids,'sorted_expert_ids':torch.zeros(400,dtype=torch.int32),
                'num_valid_ids':torch.tensor([2400,0],dtype=torch.int32)}
        payload=torch.ones((64,6,384)).to(torch.float8_e4m3fn)
        scale=torch.empty((12800,16),dtype=torch.float8_e8m0fnu);scale.view(torch.uint8).fill_(255)
        offsets,pairs,proof=live_scale_domain(inputs,payload,scale)
        self.assertEqual(offsets.numel(),4608)
        self.assertEqual(proof['unwritten_scale_padding_bytes'],200192)

    def test_dequantized_weighting_catches_raw_fp8_tolerance_false_pass(self):
        inputs,actual,golden,offsets,_=self.values()
        dense=torch.full(golden[0].shape,10.0);dense[1,0,:32]=1.0
        golden[0].copy_(dense.to(golden[0].dtype));actual[0].copy_(golden[0])
        changed=dense.clone();changed[1,0,:32]=1.125;actual[0].copy_(changed.to(actual[0].dtype))
        # Sorted live row16 maps to dense token1/slot0, not the second dense row.
        golden[1].view(torch.uint8).reshape(-1)[offsets[12]]=147
        actual[1].view(torch.uint8).reshape(-1)[offsets[12]]=147
        runner.compare(actual[0],golden[0],0.02)  # Raw-only comparison falsely accepts this.
        with self.assertRaisesRegex(AssertionError,'dequantized values differ'):
            compare_stage1_outputs(actual,golden,inputs,0.02,runner.compare)

    def test_routing_domain_uses_immutable_storage_snapshot(self):
        inputs,actual,golden,_,_=self.values()
        truth=runner.storage_snapshots(inputs);tensors,_=runner.runtime_abi(inputs,None)
        original=inputs['sorted_token_ids'].clone()
        inputs['sorted_token_ids'].zero_()
        restored=comparison_inputs_from_snapshot(inputs,truth,tensors)
        self.assertTrue(torch.equal(restored['sorted_token_ids'],original))
        compare_stage1_outputs(actual,golden,restored,0.02,runner.compare)

    def test_looser_stale_tolerance_is_rejected(self):
        inputs,actual,golden,_,_=self.values()
        with self.assertRaisesRegex(ValueError,'exactly 0.02'):
            compare_stage1_outputs(actual,golden,inputs,0.15,runner.compare)

    def test_aliased_offset_routing_restores_from_one_immutable_storage(self):
        inputs,actual,golden,_,_=self.values()
        storage=torch.zeros(160,dtype=torch.int32)
        storage[8:72]=inputs['sorted_token_ids'];storage[100:102]=inputs['num_valid_ids']
        inputs['sorted_token_ids']=storage[8:72];inputs['num_valid_ids']=storage[100:102]
        inputs['sorted_expert_ids']=storage[104:106]
        truth=runner.storage_snapshots(inputs);tensors,_=runner.runtime_abi(inputs,None)
        storage.zero_()
        restored=comparison_inputs_from_snapshot(inputs,truth,tensors)
        compare_stage1_outputs(actual,golden,restored,0.02,runner.compare)


if __name__=='__main__':unittest.main()
