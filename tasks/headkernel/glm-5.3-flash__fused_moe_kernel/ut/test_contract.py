"""Focused CPU guard/control/fixture tests. GPU arithmetic is not exercised."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'));sys.path.insert(0,str(ROOT/'scripts'))
from source_guard import validate_sources
from fixture_codec import restore_phase,fresh_numeric_fixture
from import_fixtures import validate_controls,validate_fixture,DTYPES,SHAPES,IMAGE

class Tests(unittest.TestCase):
    def valid_fixture(self,m=64,active=64,stage='decode'):
        controls={'expert_mask':None,'activation':0,'quant_type':5,'doweight_stage1':False,'a1_scale':None,'a2_scale':None,
            'block_size_M':None,'num_local_tokens':None,'moe_sorting_dispatch_policy':0,'dtype':None,'hidden_pad':0,'intermediate_pad':0,
            'bias1':None,'bias2':None,'splitk':0,'swiglu_limit':10.0,'beta':None,'linear_beta':None,'gate_mode':'separated',
            'shared_w1':None,'shared_w2':None,'shared_w1_scale':None,'shared_w2_scale':None,'shared_expert_id':-1,'stage2_scatter':None,
            'tensor_attributes':{'w1':{'is_shuffled':True},'w2':{'is_shuffled':True}}}
        tensors={}
        for name,dtype in DTYPES.items():
            shape=SHAPES.get(name,[m,8 if name in ('topk_ids','topk_weight') else 4096]);stride=1;strides=[]
            for size in reversed(shape):strides.insert(0,stride);stride*=size
            tensors[name]={'shape':shape,'stride':strides,'dtype':dtype,'storage_offset':0}
        f={'schema':'served-tensor-fixture-v1','family':'fmoe','source_sha256':'1'*64,'startup_values':False,'origin':'served_graph' if stage=='decode' else 'served_eager',
            'graph_bucket':'decode-bs'+str(m) if stage=='decode' else None,'provenance':{'image':IMAGE,'run_id':'actual-test'},
            'served':{'stage':stage,'active_requests':active if stage=='decode' else 1,'active_tokens':active if stage=='decode' else m,'tp_rank':0},
            'inputs':tensors,'outputs':{'result':{'shape':[m,4096],'stride':[4096,1],'dtype':'bfloat16','storage_offset':0}},'controls':controls}
        return f,{'provenance':{'run_id':'actual-test'}},{'files':{'aiter/fused_moe.py':'1'*64}}
    def test_tail_capture_variants_and_real_offsets_are_preserved(self):
        for m,active in [(64,64),(64,63),(4,3),(1,1)]:
            f,rank,src=self.valid_fixture(m,active);f['inputs']['hidden_states']['storage_offset']=128
            self.assertEqual(validate_fixture(f,rank,src),('decode',m))
    def test_rejects_other_packed_scale_layout(self):
        f,rank,src=self.valid_fixture();f['controls']['tensor_attributes']['w1_scale']={'is_shuffled':True}
        with self.assertRaisesRegex(ValueError,'block scales'):validate_fixture(f,rank,src)
    def test_upstream_shuffle_roundtrip_and_blockscale_codec(self):
        import torch
        from native_shuffle_reference import shuffle_weight
        from layout_contract import unpack_weight,dequantize_weight
        raw=torch.randint(-10,11,(2,256,256),generator=torch.Generator().manual_seed(7)).float().to(torch.float8_e4m3fn)
        packed=shuffle_weight(raw,(16,16))
        self.assertTrue(torch.equal(unpack_weight(packed).view(torch.uint8),raw.view(torch.uint8)))
        scales=torch.tensor([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]])
        actual=dequantize_weight(packed,scales)
        for expert in range(2):
            for n in range(2):
                for k in range(2):torch.testing.assert_close(actual[expert,n*128:(n+1)*128,k*128:(k+1)*128],raw[expert,n*128:(n+1)*128,k*128:(k+1)*128].float()*scales[expert,n,k])
    def test_numeric_refresh_changes_values_and_keeps_real_routes(self):
        import torch
        inputs={'hidden_states':(torch.arange(32).reshape(8,4).float()+1).to(torch.bfloat16),
                'topk_ids':torch.arange(16).reshape(8,2).int(),'topk_weight':torch.arange(16).reshape(8,2).float(),'w1':torch.ones(2)}
        one=fresh_numeric_fixture(inputs,123);two=fresh_numeric_fixture(inputs,124);again=fresh_numeric_fixture(inputs,123)
        self.assertTrue(torch.equal(one['inputs']['hidden_states'],again['inputs']['hidden_states']))
        self.assertFalse(torch.equal(one['inputs']['hidden_states'],two['inputs']['hidden_states']))
        order=one['inputs']['topk_ids'][:,0].long()//2
        self.assertFalse(torch.equal(one['inputs']['hidden_states'],inputs['hidden_states'].index_select(0,order)))
        torch.testing.assert_close(one['inputs']['topk_weight'],inputs['topk_weight'].index_select(0,order))
        self.assertIs(one['inputs']['w1'],inputs['w1']);self.assertEqual(one['inputs']['hidden_states'].device.type,'cpu')
        self.assertNotIn('expected',one);self.assertEqual(one['reference_policy'],'native_after_candidate_CPU_snapshot')
    def test_native_reference_follows_candidate_CPU_snapshots(self):
        import types
        events=[]
        class Tensor:
            def __init__(self,name):self.name=name
            def detach(self):return self
            def cpu(self):events.append('cpu:'+self.name);return self
            def clone(self):events.append('snapshot:'+self.name);return self
            def float(self):return self
            def contiguous(self):return self
            def view(self,dtype):return self
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:events.append('synchronize')),
            isfinite=lambda value:types.SimpleNamespace(all=lambda:True),equal=lambda a,b:True,uint8=object(),
            testing=types.SimpleNamespace(assert_close=lambda a,b,**kwargs:events.append('compare')))
        def native(**arguments):
            self.assertIn('snapshot:result',events);self.assertIn('snapshot:hidden_states',events)
            events.append('native_reference');return Tensor('native_output')
        tree=ast.parse((ROOT/'scripts/task_runner.py').read_text());build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_state');verify=next(n for n in build.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
        namespace={'torch':torch,'tensors':{'result':Tensor('result'),'hidden_states':Tensor('hidden_states')},'native_module':types.SimpleNamespace(fused_moe=native),'native_controls':{},'diagnostic':None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[verify],type_ignores=[])),'actual_protected_verify','exec'),namespace)
        namespace['verify']({'inputs':{'hidden_states':Tensor('truth')}})
        self.assertLess(events.index('snapshot:result'),events.index('native_reference'))
        self.assertLess(events.index('snapshot:hidden_states'),events.index('native_reference'))
        self.assertEqual(events[-1],'compare')
    def test_stock_guard(self):validate_sources(ROOT,ROOT)
    def test_intermediate_fp8_rounding_boundary_from_failed_prefill(self):
        import torch
        # Exact CPU observations from seed 20261067, token 5262, expert 232.
        value=torch.tensor(-4.125622272491455,dtype=torch.float32)
        maximum=torch.tensor(17.113691329956055,dtype=torch.float32)
        old_scale=maximum/448.0
        scale=maximum*torch.tensor(1/448.0,dtype=torch.float32)
        self.assertEqual(int(old_scale.view(torch.int32)),1025275857)
        self.assertEqual(int(scale.view(torch.int32)),1025275858)
        self.assertEqual(float((value/old_scale).to(torch.float8_e4m3fn)),-112.0)
        self.assertEqual(float((value*(1.0/scale)).to(torch.float8_e4m3fn)),-104.0)
        # The current frozen source has a later native reciprocal-order repair.
        evidence=json.loads((ROOT/'provenance/NATIVE-ARITHMETIC-REPAIR.json').read_text())
        self.assertEqual(evidence['source_sha256'],hashlib.sha256((ROOT/'ut/reference/kernels.py').read_bytes()).hexdigest())
    def test_native_reciprocal_guard_rejects_other_assembly_forms(self):
        source=(ROOT/'source/kernels.py').read_text()
        attacks=[('v_rcp_f32 $0, $1;','s_endpgm;'),('constraints="=v,v"','constraints="=s,s"'),
                 ('dtype=tl.float32','dtype=tl.int32'),('is_pure=True','is_pure=False'),('pack=1','pack=2')]
        for before,after in attacks:
            with self.subTest(replacement=after),tempfile.TemporaryDirectory() as temp:
                candidate=Path(temp);(candidate/'source').mkdir()
                self.assertIn(before,source)
                (candidate/'source/kernels.py').write_text(source.replace(before,after))
                with self.assertRaises(ValueError):validate_sources(candidate,ROOT)
    def test_replay_receipts_preserve_inputs_failures_and_context(self):
        from replay_receipts import ReplayReceipts
        failure=AssertionError('native-output-mismatch');truth={'seed':123};seen=[]
        def reset(seed):
            truth['seed']=seed;seen.append(('reset',seed));return truth
        def verify(value):
            self.assertIs(value,truth);seen.append(('verify',value['seed']))
            if value['seed']==124:raise failure
        with tempfile.TemporaryDirectory() as temp:
            receipts=ReplayReceipts(Path(temp),123,'source')
            checked_reset,checked_verify=receipts.leg('prefill','candidate_port',reset,verify)
            self.assertIsNone(checked_verify(truth))
            self.assertIs(checked_reset(123),truth);checked_verify(truth)
            checked_reset(124)
            with self.assertRaises(AssertionError) as caught:checked_verify(truth)
            self.assertIs(caught.exception,failure)
            rows=[json.loads(line) for line in receipts.path.read_text().splitlines()]
            self.assertEqual(rows[1]['iteration'],-1)
            self.assertEqual(rows[-1]['event'],'verify_failure')
            self.assertEqual({key:rows[-1][key] for key in ('case_id','leg','seed','iteration')},
                {'case_id':'prefill','leg':'candidate_port','seed':124,'iteration':1})
        self.assertEqual(seen,[('verify',123),('reset',123),('verify',123),('reset',124),('verify',124)])
    def test_every_entrypoint_rejects_host_rebinding(self):
        source=(ROOT/'source/kernels.py').read_text();functions=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef)]
        for function in functions:
            with self.subTest(function=function.name),tempfile.TemporaryDirectory() as temp:
                stage=Path(temp);(stage/'source').mkdir()
                lines=source.splitlines(True);lines[function.body[0].lineno-1:function.end_lineno]=['    tl = triton.runtime\n']
                (stage/'source/kernels.py').write_text(''.join(lines))
                with self.assertRaises(ValueError):validate_sources(stage,ROOT)
    def test_whitespace_does_not_select_native(self):
        source=(ROOT/'source/kernels.py').read_text()
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);(stage/'source').mkdir();(stage/'source/kernels.py').write_text(source+'\n# no semantic change\n')
            validate_sources(stage,ROOT)
        runner=(ROOT/'scripts/task_runner.py').read_text()
        self.assertNotIn('use_native',runner)
        self.assertNotIn('ut/reference/kernels.py',runner)
    def test_opaque_controls_rejected(self):
        with self.assertRaises(ValueError):validate_controls({'quant_type':5})
    def test_raw_loader_preserves_alias_offsets_and_rejects_corruption(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'blobs').mkdir();data=struct.pack('ffff',1.,2.,3.,4.);digest=hashlib.sha256(data).hexdigest();blob=root/'blobs/data.bin';blob.write_bytes(data)
            fixture={'controls':{},'inputs':{'a':{'dtype':'float32','alias':'s0','storage_offset':0,'shape':[4],'stride':[1]},
                'b':{'dtype':'float32','alias':'s0','storage_offset':1,'shape':[2],'stride':[1]}},'payload':{'inputs':{'s0':{'storage_nbytes':16,'segments':[{'blob':'blobs/data.bin','sha256':digest,'bytes':16,'offset_bytes':0}]}}}}
            result=restore_phase(root,fixture,'inputs')
            self.assertEqual(result['a'].untyped_storage().data_ptr(),result['b'].untyped_storage().data_ptr())
            self.assertEqual(result['b'].storage_offset(),1);torch.testing.assert_close(result['b'],torch.tensor([2.,3.]))
            fixture['payload']['inputs']['s0']['segments'][0]['bytes']=8
            blob.write_bytes(data[:8]);fixture['payload']['inputs']['s0']['segments'][0]['sha256']=hashlib.sha256(data[:8]).hexdigest()
            with self.assertRaisesRegex(ValueError,'uncaptured'):restore_phase(root,fixture,'inputs')
            fixture['payload']['inputs']['s0']['segments'][0]['bytes']=16
            fixture['payload']['inputs']['s0']['segments'][0]['sha256']=digest
            blob.write_bytes(b'0'*16)
            with self.assertRaisesRegex(ValueError,'hash'):restore_phase(root,fixture,'inputs')
if __name__=='__main__':unittest.main(verbosity=2)
