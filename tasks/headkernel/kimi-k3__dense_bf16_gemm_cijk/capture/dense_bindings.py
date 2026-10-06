"""Owner adapter for a new shared-runtime Kimi dense capture; no installation."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from native_dispatch import describe_dispatch
NATIVE_SHA='1fafc9782b43f8c6e198e1d39ce83e01ea3b5892f8b5e1a26c4a5eeb74252b51'


def make_bindings(runtime,module,arguments):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=NATIVE_SHA:raise ValueError('Capture wrapper source is not pinned')
    inputs={};roles={};controls={}
    for name in ('A','B','bias','scale_a','scale_b','scale_c'):
        value=arguments[name]
        if value is None:controls[name]=None
        else:
            if not runtime.torch_module().is_tensor(value):raise ValueError('Expected tensor: '+name)
            inputs[name]=value;roles[name]=runtime.Role('mutable' if name=='A' else 'readonly')
    otype=arguments['otype'];controls['otype']=None if otype is None else {'kind':'dtype','name':str(otype).removeprefix('torch.')}
    controls['tensor_attributes']={name:({'is_shuffled':bool(value.is_shuffled)} if hasattr(value,'is_shuffled') else {}) for name,value in inputs.items()}
    controls['capture_native_dispatch']=describe_dispatch(module,arguments['A'],arguments['B'],bias=arguments['bias'],otype=otype,scale_a=arguments['scale_a'],scale_b=arguments['scale_b'])
    return runtime.Family('bf16_gemm',NATIVE_SHA,roles,{'result':runtime.Role('mutable')}),inputs,controls


def outputs_for(result):return {'result':result}


def make_aten_bindings(runtime,torch,arguments):
    """Call only outside a parent bf16_gemm wrapper to avoid double counting."""
    supplied=arguments.get('out') is not None
    schema=torch.ops.aten.mm.out._schema if supplied else torch.ops.aten.mm.default._schema
    source=hashlib.sha256(str(schema).encode()).hexdigest()
    inputs={name:arguments[name] for name in ('A','B')}
    roles={'A':runtime.Role('mutable'),'B':runtime.Role('readonly')}
    controls={'out':{'kind':'output_binding','name':'result'} if supplied else None,
              'tensor_attributes':{name:({'is_shuffled':bool(value.is_shuffled)} if hasattr(value,'is_shuffled') else {}) for name,value in inputs.items()}}
    return runtime.Family('aten_bf16_mm',source,roles,{'result':runtime.Role('mutable')}),inputs,controls


def aten_outputs_for(arguments,result):
    out=arguments.get('out')
    if out is not None and (result.data_ptr()!=out.data_ptr() or result.shape!=out.shape or result.stride()!=out.stride()):raise ValueError('ATen out argument differs from returned storage/view')
    return {'result':result}


def checked_native_call(module,function,args,kwargs,expected_dispatch):
    """Record the actual cached solution call while delegating unchanged work."""
    from native_dispatch import bind_solutions,typed_config
    originals=dict(module.solMap)
    def original_solution(*values,**named):
        config=named.get('config')
        if typed_config(config)!=expected_dispatch['config']:raise ValueError('Actual solution config differs from pre-call capture binding')
        return originals[config['libtype']](*values,**named)
    with bind_solutions(module,original_solution) as calls:
        result=function(*args,**kwargs)
    if calls!=[expected_dispatch['libtype']]:raise ValueError('Native wrapper did not reach the captured solMap entry exactly once')
    return result
