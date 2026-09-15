"""Independent known-output checks and controls of the actual task comparator."""
from pathlib import Path
import ast
import importlib.util
import sys
from reference_support import references


def load_model():
    path=Path(__file__).resolve().parents[1]/"model.py"
    spec=importlib.util.spec_from_file_location("arena_control_model",path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def verify(actual,expected,label,atol=1e-5,rtol=1e-5):
    import torch
    def check(a,b):
        if isinstance(a,(tuple,list)):
            return isinstance(b,(tuple,list)) and len(a)==len(b) and all(check(x,y) for x,y in zip(a,b))
        return a.shape==b.shape and a.dtype==b.dtype and bool(torch.isfinite(a.float()).all()) and bool(torch.allclose(a.float(),b.float(),atol=atol,rtol=rtol))
    if not check(actual,expected): raise AssertionError(f"Independent reference known answer failed: {label}")
    # Exercise the actual protected comparator where it is separately defined.
    path=Path(__file__).resolve().parents[1]/"test_kernel_harness.py"
    names={n.name for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    def corrupt(a):
        if isinstance(a,(tuple,list)):return type(a)([corrupt(a[0]),*a[1:]])
        if str(a.dtype).startswith(('torch.float8','torch.float4')):
            bad=a.clone(); bad.view(torch.uint8).reshape(-1)[0]^=32
            return bad
        return (a.float()+1000).to(a.dtype)
    wrong=corrupt(actual)
    if "_compare" in names:
        h=references(["_compare"])
        if not h._compare(actual,expected)[0]: raise AssertionError("Task comparator rejected independent known answer")
        if h._compare(wrong,expected)[0]: raise AssertionError("Task comparator accepted wrong output")
        comparator="task _compare"
    elif "_norm_max_err" in names:
        h=references(["_norm_max_err"])
        a=actual[0] if isinstance(actual,(tuple,list)) else actual
        b=expected[0] if isinstance(expected,(tuple,list)) else expected
        w=wrong[0] if isinstance(wrong,(tuple,list)) else wrong
        if h._norm_max_err(b,a)[0]>h.REL_TOL or h._norm_max_err(b,w)[0]<=h.REL_TOL:
            raise AssertionError("Task normalized-error comparator control failed")
        comparator="task _norm_max_err"
    else:
        if check(wrong,expected):raise AssertionError("Known-answer comparator accepted wrong output")
        comparator="known-answer output check; full task gate additionally runs on GPU"
    return {"control":label,"known_answer":"PASS","negative_output":"rejected","comparator":comparator}
