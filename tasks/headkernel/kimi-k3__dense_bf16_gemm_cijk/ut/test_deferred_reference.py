"""CPU event spy for the protected post-candidate GPU-reference ordering."""
import ast
from pathlib import Path
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_candidate_and_inputs_are_observed_before_fp32_reference(self):
        events=[];backend=types.SimpleNamespace(allow_tf32=True)
        class Tensor:
            dtype='bfloat16'
            def __init__(self,name):self.name=name
            def cpu(self):events.append('cpu:'+self.name);return Tensor(self.name+'-cpu')
            def clone(self):events.append('clone:'+self.name);return self
            def detach(self):return self
            def float(self):return self
            def contiguous(self):return self
            def view(self,*a):return self
            def to(self,*a):return self
            def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:self.name)
            def __matmul__(self,other):
                self_outer.assertFalse(backend.allow_tf32)
                events.append('GPU_FP32_reference');return Tensor('reference')
        self_outer=self
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:events.append('sync')),
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=backend)),
            isfinite=lambda value:types.SimpleNamespace(all=lambda:True),equal=lambda a,b:True,uint8='uint8',
            testing=types.SimpleNamespace(assert_close=lambda a,b,**kw:events.append('compare')))
        tree=ast.parse((ROOT/'scripts/task_runner.py').read_text());build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_state');verify=next(n for n in build.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
        guard=types.SimpleNamespace(snapshot_complete=lambda t:t.cpu().clone(),view_snapshot=lambda t,s:t,
            assert_output_guards=lambda *a:None,
            assert_inputs_unchanged=lambda tensors,inputs:[tensors[name].cpu().clone() for name in inputs])
        env={'torch':torch,'tensors':{name:Tensor(name) for name in ('A','B','C')},'storage_guard':guard,'case':{'tensors':{'C':{}}},'native_events':[]}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[verify],type_ignores=[])),'protected_verify','exec'),env)
        env['verify']((None,{'A':Tensor('truthA'),'B':Tensor('truthB')}))
        position=events.index('GPU_FP32_reference')
        for name in ('A','B','C'):self.assertLess(events.index('cpu:'+name),position)
        self.assertLess(events.index('clone:C-cpu'),position);self.assertEqual(events[-1],'compare');self.assertTrue(backend.allow_tf32)
if __name__=='__main__':unittest.main(verbosity=2)
