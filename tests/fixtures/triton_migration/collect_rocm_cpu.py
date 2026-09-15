"""CPU-only collection inventory. Stubs permit decorators; no kernel is executed."""
import ast,importlib.util,itertools,json,sys,types
from pathlib import Path
import torch,numpy,pytest
from _pytest.mark.structures import ParameterSet
from contextlib import redirect_stdout
import io

class Attr:
 def __init__(self,name='stub'):self.name=name
 def __getattr__(self,k):return Attr(self.name+'.'+k)
 def __call__(self,*a,**k):return a[0] if a and callable(a[0]) else (lambda f:f)
 def __repr__(self):return self.name
class Const:
 def __init__(self,v):self.value=v
 def __int__(self):return int(self.value)
class Jit:
 def __call__(self,f=None,**kwargs):return f if f else lambda f:f
tri=types.ModuleType('triton');tl=types.ModuleType('triton.language');tri.__path__=[];tl.__path__=[]
tri.jit=Jit();tri.autotune=lambda *a,**k:lambda f:f;tri.heuristics=lambda *a,**k:lambda f:f
tri.Config=lambda *a,**k:None;tri.cdiv=lambda a,b:(a+b-1)//b;tri.next_power_of_2=lambda a:1<<(a-1).bit_length()
tri.language=tl;tri.runtime=Attr();tri.testing=Attr();tri.__getattr__=lambda k:Attr('triton.'+k)
tl.constexpr=Const;tl.__getattr__=lambda k:Attr('tl.'+k)
sys.modules.update({'triton':tri,'triton.language':tl})
for name in ['triton.runtime','triton.runtime.driver','triton.runtime.jit','triton.tools','triton.tools.mxfp','triton.tools.experimental_descriptor','triton._C','triton._C.libtriton','triton.compiler','triton.compiler.compiler','triton.backends','triton.backends.compiler']:
 mod=types.ModuleType(name);mod.__path__=[];mod.__getattr__=lambda k:Attr(k);sys.modules[name]=mod
perf=types.ModuleType('performance_utils_pytest');perf.PytestBenchmarker=Attr();perf.do_bench_config=Attr();perf.save_all_benchmark_results=Attr();sys.modules['performance_utils_pytest']=perf
tri.runtime=types.SimpleNamespace(driver=types.SimpleNamespace(active=types.SimpleNamespace(
 get_current_target=lambda:types.SimpleNamespace(backend='hip',arch='gfx950'),
 get_active_torch_device=lambda:torch.device('cpu'),
 get_device_properties=lambda d:{'max_shared_mem':65536,'multiprocessor_count':120})))
sys.modules['triton.runtime.driver'].active=tri.runtime.driver.active
torch.cuda.get_device_properties=lambda *a:types.SimpleNamespace(multi_processor_count=120,shared_memory_per_block=65536)
torch.cuda.get_device_name=lambda *a:'CPU collection stub'
torch.cuda.current_device=lambda:0
torch.cuda.get_device_capability=lambda *a:(9,0)
torch.cuda.is_available=lambda:False
original_empty=torch.empty
torch.empty=lambda *a,**k:original_empty(*a,**{**k,'device':'cpu'})

def serial(v):
 if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if isinstance(v,(bool,int,float,str)) or v is None:return v
 return str(v)

out={};errors={}
for p in sorted([*Path('tasks/instruction2triton/rocmbench').glob('*/*.py'),*Path('tasks/triton2triton/rocmbench').glob('*/*/*.py')]):
 if p.name != p.parent.name+'.py' or not (p.parent/'config.yaml').exists():continue
 try:
  name='inventory_'+p.parent.name
  spec=importlib.util.spec_from_file_location(name,p.resolve());module=importlib.util.module_from_spec(spec);sys.modules[name]=module
  with redirect_stdout(io.StringIO()):spec.loader.exec_module(module)
  functions=[]
  for fname in module.__dict__:
   f=getattr(module,fname)
   if not fname.startswith('test_') or not callable(f) or fname.startswith('test_save'):continue
   cases=[{}]
   for mark in reversed(getattr(f,'pytestmark',[])):
    if mark.name!='parametrize':continue
    keys=mark.args[0];keys=[x.strip() for x in keys.split(',')] if isinstance(keys,str) else keys
    vals=list(mark.args[1]); expanded=[]
    for previous in cases:
     for val in vals:
      if isinstance(val,ParameterSet):values=val.values
      else:values=val if len(keys)>1 else [val]
      row=dict(previous);row.update(zip(keys,values));expanded.append(row)
    cases=expanded
   functions.append({'function':fname,'cases':[serial(v) for v in cases]})
  out[p.parent.as_posix()]=functions
 except BaseException as e:errors[p.parent.as_posix()]=f'{type(e).__name__}: {e}'
if errors:raise RuntimeError(errors)
print(json.dumps(out,sort_keys=True))
