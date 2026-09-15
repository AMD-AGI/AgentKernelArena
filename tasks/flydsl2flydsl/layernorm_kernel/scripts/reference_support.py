"""Load only protected reference functions for CPU known-answer controls.

No candidate imports, GPU initialization, or harness main function executes here.
The controls call the same function bodies the GPU harness uses.
"""
import ast
import math
from pathlib import Path
import types


def references(names, constants=None):
    import torch
    ns = {"torch":torch,"math":math,"UNIT_SIZE":32}
    ns.update(constants or {})
    path = Path(__file__).resolve().parents[1] / "test_kernel_harness.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try: value=ast.literal_eval(node.value)
            except (ValueError,TypeError): continue
            for target in node.targets:
                if isinstance(target, ast.Name): ns[target.id]=value
                elif isinstance(target,ast.Tuple) and isinstance(value,(tuple,list)):
                    ns.update({n.id:v for n,v in zip(target.elts,value) if isinstance(n,ast.Name)})
    definitions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    needed=set(names)
    for _ in range(len(definitions)):
        extra={n.id for name in needed for n in ast.walk(definitions[name]) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in definitions}
        if extra <= needed: break
        needed.update(extra)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in needed]
    exec(compile(ast.Module(nodes,type_ignores=[]),str(path),"exec"),ns)
    return types.SimpleNamespace(**ns)


def control(actual, expected, accept, label):
    import torch
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise AssertionError(f"Invalid reference output for {label}")
    # Expected comes from arithmetic/explicit values independent of the reference.
    if not accept(actual, expected):
        raise AssertionError(f"Reference failed independent known answer: {label}")
    wrong = expected + 100
    if accept(wrong, expected):
        raise AssertionError(f"Comparator accepted deliberately incorrect output: {label}")
    return {"control":label,"known_answer":"PASS","negative_output":"rejected"}


def close(atol, rtol):
    import torch
    return lambda a,b: bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a,b,atol=atol,rtol=rtol))
