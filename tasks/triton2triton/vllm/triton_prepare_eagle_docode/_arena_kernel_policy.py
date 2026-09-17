"""Protected EAGLE launcher binding checks, outside the measured invocation.

The wrapper is protected separately by Arena's symbol guard. A retained JIT
decorator alone does not establish what that wrapper will actually dispatch.
These checks enforce the task's launcher contract, not a Python sandbox.
"""
import ast
import importlib.util
from pathlib import Path
from types import FunctionType

KERNEL = '_prepare_eagle_docode_kernel'


def inspect_definition(node):
    """Only the imported Triton compiler/tuning chain may wrap the target."""
    names = []
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if not (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name) and target.value.id == 'triton'
                and target.attr in {'jit', 'autotune', 'heuristics'}):
            raise ValueError(f'{KERNEL}: unsupported launcher decorator {ast.unparse(decorator)}')
        if target.attr != 'jit' and not isinstance(decorator, ast.Call):
            raise ValueError(f'{KERNEL}: tuning decorators require explicit arguments')
        names.append(target.attr)
    if not names or names[-1] != 'jit' or names.count('jit') != 1:
        raise ValueError(f'{KERNEL}: one innermost triton.jit is required')


def _runtime_types():
    from triton.runtime.jit import JITFunction
    from triton.runtime.autotuner import Autotuner, Heuristics
    return JITFunction, Autotuner, Heuristics


def check_binding(module, source, runtime_types, methods):
    """Reject host proxies, subclasses, cycles, and replacement launch methods."""
    jit_type = runtime_types[0]
    current = getattr(module, KERNEL, None)
    seen = set()
    while True:
        kind = type(current)
        if kind not in runtime_types or id(current) in seen:
            raise ValueError(f'{KERNEL}: binding must be a genuine Triton JIT/tuning chain')
        seen.add(id(current))
        # Capture methods before candidate import; checking class names or
        # isinstance alone would also accept a replacement/subclass launcher.
        run, getitem = methods[kind]
        bound_run = getattr(current, 'run', None)
        if (kind.run is not run or kind.__getitem__ is not getitem
                or getattr(bound_run, '__func__', None) is not run
                or getattr(bound_run, '__self__', None) is not current):
            raise ValueError(f'{KERNEL}: native Triton launch method was replaced')
        function = vars(current).get('fn')
        if kind is jit_type:
            if (type(function) is not FunctionType or function.__name__ != KERNEL
                    or function.__globals__ is not vars(module)
                    or Path(function.__code__.co_filename).resolve() != Path(source).resolve()):
                raise ValueError(f'{KERNEL}: JIT must bind the declared task function')
            return
        current = function


def load_checked(source):
    source = Path(source)
    tree = ast.parse(source.read_text(), filename=str(source))
    targets = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == KERNEL]
    if len(targets) != 1:
        raise ValueError(f'{KERNEL}: exactly one declared definition is required')
    inspect_definition(targets[0])  # Reject host decorators before executing them.
    runtime_types = _runtime_types()
    methods = {kind: (kind.run, kind.__getitem__) for kind in runtime_types}
    spec = importlib.util.spec_from_file_location('triton_kernel', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_binding(module, source, runtime_types, methods)
    return module
