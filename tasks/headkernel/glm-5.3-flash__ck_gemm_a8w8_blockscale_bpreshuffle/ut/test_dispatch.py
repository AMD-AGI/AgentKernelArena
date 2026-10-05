"""CPU dispatch regression. This does not compile or execute GPU arithmetic."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest.mock

ROOT=Path(__file__).resolve().parents[1]

class Tensor:
    def __init__(self,shape,strides,dtype,device):
        self.shape=tuple(shape);self.strides=tuple(strides);self.dtype=dtype;self.device=device
    def copy_(self,value):return self

class Kernel:
    def __init__(self,function):self.function=function;self.calls=[]
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            self.calls.append((grid,tuple((x.shape,x.strides,x.dtype,x.device) if isinstance(x,Tensor) else x for x in args),kwargs))
            return self
        return launch


def main():
    torch=types.ModuleType('torch');torch.float8_e4m3fn='float8_e4m3fn';torch.bfloat16='bfloat16';torch.float32='float32'
    torch.empty_strided=lambda shape,strides,dtype,device:Tensor(shape,strides,dtype,device)
    triton=types.ModuleType('triton');tl=types.ModuleType('triton.language');tl.constexpr=object();triton.language=tl;triton.jit=Kernel
    specs=json.loads((ROOT/'cases.json').read_text())['cases'];original=(ROOT/'source/kernels.py').read_text()
    variants={'stock':original,'whitespace':original+'\n\n# Dispatch must be independent of source bytes.\n',
              'local_rename':re.sub(r'\brows\b','row_indices',original)}
    results={}
    with unittest.mock.patch.dict(sys.modules,{'torch':torch,'triton':triton,'triton.language':tl}):
        spec=importlib.util.spec_from_file_location('runner_dispatch_test',ROOT/'scripts/task_runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        validator=runner.validate_sources
        runner.validate_sources=lambda candidate,reference:validator(candidate,ROOT)
        runner.generate=lambda case,seed:({},None)
        runner.observe_case=lambda case,tensors,scalars:case
        for label,source in variants.items():
            with tempfile.TemporaryDirectory() as tmp:
                stage=Path(tmp);(stage/'source').mkdir();(stage/'source/kernels.py').write_text(source)
                runner.validate_sources(stage,ROOT)
                runner.ROOT=stage
                module=runner.load_source()
                for case in specs:
                    state=runner.build_state(case,0,module);state[1]()
                results[label]=module.gemm_kernel.calls
    assert results['stock']==results['whitespace']==results['local_rename'], 'A nonsemantic source edit changed algorithm dispatch'
    assert len(results['stock'])==len(specs)
    print(json.dumps({'status':'PASS','case_count':len(specs),'variants':list(variants),'launch_path':'submitted gemm_kernel for every variant','GPU_actions':False}))

if __name__=='__main__':main()
