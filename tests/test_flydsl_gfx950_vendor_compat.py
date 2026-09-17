"""Legacy/current API routing and real dependency failures for bundled fallback."""
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl'


@pytest.mark.parametrize('name', ['buffer_ops', 'vector'])
@pytest.mark.parametrize('runtime', ['legacy', 'current', 'broken_dependency'])
def test_vendor_fallback_only_handles_the_removed_module(name, runtime):
    source = ROOT/'flash_attn_func_kernel/flydsl_compat/__init__.py'
    block = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.Try)
                 and isinstance(node.body[0], ast.Import)
                 and node.body[0].names[0].name == 'flydsl.expr.' + name)
    legacy, current = object(), object()
    attempted = []
    def importer(module, globals=None, locals=None, fromlist=(), level=0):
        attempted.append((module, level))
        if level == 0:
            assert module == 'flydsl.expr.' + name
            if runtime == 'legacy':
                return SimpleNamespace(expr=SimpleNamespace(**{name: legacy}))
            missing = 'missing_mlir_dependency' if runtime == 'broken_dependency' else module
            raise ModuleNotFoundError('Missing dependency', name=missing)
        assert level == 1 and module == ''
        return SimpleNamespace(**{name: current})
    scope = {'__builtins__': {**vars(builtins), '__import__': importer}}
    code = compile(ast.Module(body=[block], type_ignores=[]), 'compat_import', 'exec')
    if runtime == 'broken_dependency':
        with pytest.raises(ModuleNotFoundError) as error:
            exec(code, scope)
        assert error.value.name == 'missing_mlir_dependency'
        assert len(attempted) == 1
    else:
        exec(code, scope)
        assert scope[name] is (legacy if runtime == 'legacy' else current)
