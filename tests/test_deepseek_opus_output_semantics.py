"""CPU checks for physical Opus FP8/E8M0 output semantics; no fixture fabrication."""
import importlib.util
from pathlib import Path
import sys

import pytest

TASK = Path(__file__).resolve().parents[1]/'tasks/headkernel/deepseek-v4-pro__moe_stage1_grouped_gemm_silu_opus_a8w4'


@pytest.fixture(scope='module')
def codec():
    spec=importlib.util.spec_from_file_location('opus_output_contract',TASK/'ut/output_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def example(codec):
    torch=pytest.importorskip('torch')
    if not hasattr(torch,'float8_e8m0fnu'):pytest.skip('E8M0 tensor support required')
    ids=torch.full((64,),2,dtype=torch.int32)
    for row,value in [(0,0),(16,1),(32,1<<24),(48,(1<<24)|1)]:ids[row]=value
    inputs={'hidden_states':torch.empty(2,64),'w1':torch.empty(2,768,32),'topk':2,'block_m':32,
        'sorted_token_ids':ids,'sorted_expert_ids':torch.zeros(2,dtype=torch.int32),
        'num_valid_ids':torch.tensor([64,0],dtype=torch.int32)}
    payload=torch.ones(2,2,384).to(torch.float8_e4m3fn)
    scale=torch.empty((256,16),dtype=torch.float8_e8m0fnu);scale.view(torch.uint8).zero_()
    offsets,pairs,proof=codec.live_scale_domain(inputs,payload,scale)
    scale.view(torch.uint8).reshape(-1)[offsets]=127
    return torch,inputs,(payload.clone(),scale.clone()),(payload,scale),offsets


def test_semantic_identity_and_full_scale_extent(codec,example):
    torch,inputs,actual,golden,offsets=example
    proof=codec.compare_opus_outputs(actual,golden,inputs)
    assert proof['semantic_tolerance']==0.02
    assert proof['scale_comparison']=='every allocated byte, including initialized padding'
    assert proof['meaningful_scale_bytes']==48 and proof['allocated_scale_bytes']==4096
    assert {0,1,256,257,512,513,768,769} <= set(offsets.tolist())


@pytest.mark.parametrize('live',[True,False])
def test_every_scale_byte_remains_exact(codec,example,live):
    torch,inputs,actual,golden,offsets=example
    location=int(offsets[0]) if live else next(i for i in range(actual[1].numel()) if i not in set(offsets.tolist()))
    actual[1].view(torch.uint8).reshape(-1)[location]^=1
    with pytest.raises(AssertionError,match='complete E8M0'):
        codec.compare_opus_outputs(actual,golden,inputs)


def test_nonuniform_scales_cannot_hide_a_large_physical_error(codec,example):
    torch,inputs,actual,golden,offsets=example
    payload=torch.full(golden[0].shape,10.0);payload[1,0,:32]=1.0
    golden[0].copy_(payload.to(golden[0].dtype));actual[0].copy_(golden[0])
    bad=payload.clone();bad[1,0,:32]=1.125;actual[0].copy_(bad.to(actual[0].dtype))
    golden[1].view(torch.uint8).reshape(-1)[offsets[12]]=147
    actual[1].view(torch.uint8).reshape(-1)[offsets[12]]=147
    # Encoded values satisfy even a 2% raw-code RMS bound, despite the changed
    # group dominating the physical activation. Dequantization must catch it.
    a,e=actual[0].float(),golden[0].float()
    assert bool(((a-e).abs() <= .02*e.square().mean().sqrt()+.02*e.abs()).all())
    with pytest.raises(AssertionError,match='dequantized values differ'):
        codec.compare_opus_outputs(actual,golden,inputs)


def test_dense_output_corruption_and_invalid_reference_are_distinct(codec,example):
    torch,inputs,actual,golden,offsets=example
    actual[0].view(torch.uint8).zero_()
    with pytest.raises(AssertionError,match='dequantized values differ'):
        codec.compare_opus_outputs(actual,golden,inputs)
    golden[1].view(torch.uint8).reshape(-1)[offsets[0]]=255
    actual[1].view(torch.uint8).copy_(golden[1].view(torch.uint8))
    with pytest.raises(ValueError,match='reference has nonfinite'):
        codec.compare_opus_outputs(actual,golden,inputs)


def test_duplicate_or_missing_routes_are_not_a_valid_rejection_oracle(codec,example):
    _,inputs,actual,golden,_=example
    inputs['sorted_token_ids'][16]=0
    with pytest.raises(ValueError,match='exactly one'):
        codec.compare_opus_outputs(actual,golden,inputs)


def test_scale_mapping_uses_owned_routing_truth(codec,example):
    torch,inputs,actual,golden,_=example
    storage=torch.zeros(160,dtype=torch.int32)
    storage[8:72]=inputs['sorted_token_ids'];storage[100:102]=inputs['num_valid_ids']
    inputs['sorted_token_ids']=storage[8:72];inputs['num_valid_ids']=storage[100:102]
    inputs['sorted_expert_ids']=storage[104:106]
    raw=torch.empty(0,dtype=torch.uint8).set_(storage.untyped_storage(),0,(storage.untyped_storage().nbytes(),),(1,))
    before={'arg.sorted_token_ids':raw.clone()}
    bindings={'arg.'+name:inputs[name] for name in ['sorted_token_ids','sorted_expert_ids','num_valid_ids']}
    storage.zero_()
    restored=codec.comparison_inputs_from_snapshot(inputs,before,bindings)
    assert restored['num_valid_ids'][0].item()==64
    codec.compare_opus_outputs(actual,golden,restored)


def test_runner_retains_raw_gate_before_new_semantic_gate(monkeypatch):
    import ast
    from types import SimpleNamespace
    tree=ast.parse((TASK/'scripts/task_runner.py').read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='compare_native_outputs')
    log=[];truth=object();proof={'live_scale_offsets_sha256':'test-proof'}
    fake=SimpleNamespace(comparison_inputs_from_snapshot=lambda inputs,snapshots,tensors: log.append(('truth',snapshots)) or 'CPU routing',
                         compare_opus_outputs=lambda a,g,inputs,path: log.append(('semantic',inputs)) or proof)
    monkeypatch.setitem(sys.modules,'output_contract',fake)
    namespace={'compare':lambda *args:log.append(('raw',args[-2])),
               'runtime_abi':lambda inputs,output:({},{}),'OUTPUT_CONTRACT_PROOFS':{}}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<semantic-output-gate>','exec'),namespace)
    namespace['compare_native_outputs']('a','g',{},0.15,expected_inputs=truth)
    assert log==[('raw',0.15),('truth',truth),('semantic','CPU routing')]
    assert namespace['OUTPUT_CONTRACT_PROOFS']=={'test-proof':proof}
