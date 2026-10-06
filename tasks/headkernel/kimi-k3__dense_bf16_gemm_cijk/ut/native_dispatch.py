"""Protected pinned-native dispatch inspection and effective solMap binding."""
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def typed_config(value):
    if value is None or type(value) in (str,bool,int):return value
    if type(value) is float:
        return value if math.isfinite(value) else {'kind':'nonfinite_float','value':'nan' if math.isnan(value) else 'inf' if value>0 else '-inf'}
    if isinstance(value,dict):
        if not all(type(k) is str for k in value):raise ValueError('Non-string native config key')
        return {k:typed_config(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [typed_config(v) for v in value]
    if type(value).__module__.split('.')[0]=='numpy' and hasattr(value,'item'):return typed_config(value.item())
    raise ValueError('Unencoded native config value: '+type(value).__qualname__)


def load_native():
    module=importlib.import_module('aiter.tuned_gemm')
    pinned=json.loads((ROOT/'provenance/NATIVE-BASELINE.json').read_text())['source_hashes_by_family']['bf16_gemm']
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=pinned:raise ValueError('Native tuned GEMM source changed')
    return module


def describe_dispatch(module,A,B,*,bias=None,otype=None,scale_a=None,scale_b=None):
    if A.dim()!=2 or B.dim()!=2:raise ValueError('Dense task requires a captured 2D contract')
    config=module.get_GEMM_A16W16_config(M=A.shape[0],N=B.shape[0],K=A.shape[1],bias=bias is not None,dtype=str(A.dtype),otype=str(A.dtype if otype is None else otype),scaleAB=scale_a is not None or scale_b is not None,bpreshuffle=getattr(B,'is_shuffled',False) is True)
    function=module.solMap[config['libtype']]
    return {'schema':'aiter-solmap-dispatch-v1','config':typed_config(config),'libtype':config['libtype'],'solidx':typed_config(config['solidx']),'callable':{'module':function.__module__,'qualname':function.__qualname__}}


@contextmanager
def bind_solutions(module,replacement):
    """Update cached dictionary entries; restore identities even on failure."""
    table=module.solMap;original=dict(table);calls=[]
    if not original or not all(callable(fn) for fn in original.values()):raise ValueError('Invalid native solution table')
    def bind(name):
        def call(*args,**kwargs):
            calls.append(name)
            return replacement(*args,**kwargs)
        return call
    wrappers={name:bind(name) for name in original}
    table.update(wrappers)
    try:
        yield calls
    finally:
        changed=module.solMap is not table or set(table)!=set(original) or any(table.get(name) is not wrapped for name,wrapped in wrappers.items())
        table.clear();table.update(original)
        if changed:raise ValueError('Native solution table changed during protected binding')


def require_dispatch(case,observed):
    expected=case['capture_controls'].get('capture_native_dispatch')
    if expected is None or expected!=observed:raise ValueError('Actual native dispatch differs from captured config/callable')
