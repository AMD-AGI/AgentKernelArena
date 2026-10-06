"""CPU binding proof against the vendored exact native wrapper body."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from native_dispatch import bind_solutions,describe_dispatch,typed_config,require_dispatch


class Tests(unittest.TestCase):
    def native_wrapper(self):
        import torch
        source=ROOT/'ut/native/tuned_gemm.py'
        pin=json.loads((ROOT/'provenance/NATIVE-BASELINE.json').read_text())['source_hashes_by_family']['bf16_gemm']
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),pin)
        fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gemm_a16w16');fn.decorator_list=[]
        # Execute the original Python dispatch body with CPU tensors and a spy
        # solution. Only its external annotation/decorator/runtime dependencies
        # are supplied; its solMap dispatch statements are unchanged.
        namespace={'Tensor':torch.Tensor,'torch':torch,'save_shapes':lambda *a:None,
                   'get_GEMM_A16W16_config':lambda **kw:{'libtype':'torch','solidx':0,'kernelName':None}}
        def original(A,B,*args,**kwargs):return A@B.t()
        namespace['solMap']={'torch':original}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'pinned_native_dispatch','exec'),namespace)
        module=types.SimpleNamespace(solMap=namespace['solMap'],torch_gemm=original,gemm_a16w16=namespace['gemm_a16w16'],get_GEMM_A16W16_config=namespace['get_GEMM_A16W16_config'])
        return module
    def test_module_only_rebinding_is_dead_but_solmap_binding_reaches_replacement(self):
        import torch
        native=self.native_wrapper();a=torch.ones(2,4);b=torch.ones(3,4);original=native.solMap['torch']
        def edited(A,B,*args,**kwargs):return torch.full((A.shape[0],B.shape[0]),123.,dtype=A.dtype)
        native.torch_gemm=edited
        self.assertTrue(torch.equal(native.gemm_a16w16(a,b),torch.full((2,3),4.)))
        with bind_solutions(native,edited) as calls:result=native.gemm_a16w16(a,b)
        self.assertEqual(calls,['torch']);self.assertTrue(torch.equal(result,torch.full((2,3),123.)))
        self.assertIs(native.solMap['torch'],original)
    def test_binding_restores_original_entries_after_exception(self):
        native=self.native_wrapper();original=dict(native.solMap)
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with bind_solutions(native,lambda *a:None):raise RuntimeError('sentinel')
        self.assertEqual(native.solMap,original)
    def test_dispatch_config_and_callable_are_required(self):
        import torch
        native=self.native_wrapper();observed=describe_dispatch(native,torch.ones(2,4),torch.ones(3,4))
        require_dispatch({'capture_controls':{'capture_native_dispatch':observed}},observed)
        bad=json.loads(json.dumps(observed));bad['solidx']=1
        with self.assertRaises(ValueError):require_dispatch({'capture_controls':{'capture_native_dispatch':bad}},observed)
    def test_typed_config_never_emits_nonfinite_json(self):
        encoded=typed_config({'optional':float('nan'),'solidx':0,'scale':float('inf')})
        json.dumps(encoded,allow_nan=False)
        self.assertEqual(encoded['optional'],{'kind':'nonfinite_float','value':'nan'})
    def test_protected_runner_reaches_submitted_kernel_through_native_wrapper(self):
        import importlib.util
        import torch
        from unittest.mock import patch
        import native_dispatch
        import storage_guard
        native=self.native_wrapper()
        spec=importlib.util.spec_from_file_location('binding_runner',ROOT/'scripts/task_runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        m,n,k=2,8,128
        def tensor(shape,strides,role):return {'shape':shape,'strides':strides,'storage_offset':0,'dtype':'bfloat16','role':role,'device_type':'cuda'}
        case={'case_id':'CPU-binding-spy','scalars':{'M':m,'N':n,'K':k,'fp8':False,'BM':32,'BN':64,'BK':128},'tensors':{'A':tensor([m,k],[k,1],'input'),'B':tensor([k,n],[1,k],'input'),'C':tensor([m,n],[n,1],'output')},'live_fixture':{'capture_family':'bf16_gemm'},'capture_controls':{'otype':{'kind':'dtype','name':'bfloat16'}}}
        case['capture_controls']['capture_native_dispatch']=describe_dispatch(native,torch.empty(m,k,dtype=torch.bfloat16),torch.empty(n,k,dtype=torch.bfloat16))
        calls=[]
        class Kernel:
            def __getitem__(self,grid):
                def launch(*args,**kwargs):
                    calls.append((grid,args,kwargs));return types.SimpleNamespace(name='submitted_spy',hash='spy')
                return launch
        original_empty=torch.empty
        def cpu_empty(*args,**kwargs):kwargs['device']='cpu';return original_empty(*args,**kwargs)
        with patch.object(storage_guard,'extents',return_value={'A':m*k*2,'B':k*n*2,'C':m*n*2}),patch.object(torch,'empty',cpu_empty),patch.object(runner,'observe_case',lambda case,tensors,scalars:case),patch.object(native_dispatch,'load_native',return_value=native):
            state=runner.build_state(case,0,types.SimpleNamespace(gemm_kernel=Kernel()));compiled=state[1]()
        self.assertEqual(compiled.name,'submitted_spy');self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],(1,1));self.assertEqual(calls[0][1][5:8],(m,n,k))
        self.assertEqual(calls[0][1][1].stride(),(1,k))

    def test_pending_capture_is_not_a_scoreable_manifest(self):
        if (ROOT/'NOT_BUILT').is_file():self.assertFalse((ROOT/'cases.json').exists())
        requirements=json.loads((ROOT/'CASE-REQUIREMENTS.json').read_text())
        self.assertEqual(len(requirements['observed_wrapper_requirements']),18)
        self.assertEqual({r['inputs']['A']['shape'][0] for r in requirements['observed_wrapper_requirements']},{8192,16384})
        self.assertTrue(all('occurrences' not in r for r in requirements['observed_wrapper_requirements']))
if __name__=='__main__':unittest.main(verbosity=2)
