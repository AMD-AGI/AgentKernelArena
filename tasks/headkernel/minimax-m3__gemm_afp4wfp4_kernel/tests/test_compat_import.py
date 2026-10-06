"""Reproduce the pinned native import mechanism without Torch or a GPU."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_original_aiter_redirect_retains_a_different_registered_namespace(tmp_path):
    source = ROOT / "ut/native/triton_compat.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_backward_compat_find_spec")
    for name in ("aiter", "aiter/ops", "aiter/ops/triton", "aiter/ops/triton/gemm", "aiter/ops/triton/gemm/basic"):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("")
    (tmp_path / "keeper.py").write_text("registrations = {}\nexecutions = []\n")
    (tmp_path / "aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py").write_text(
        "import keeper\nkeeper.executions.append(globals())\n"
        "def gemm_afp4wfp4_(): return globals()\n"
        "if 'op' not in keeper.registrations: keeper.registrations['op'] = gemm_afp4wfp4_\n"
        "def gemm_afp4wfp4(): return keeper.registrations['op']()\n")
    script = ("import importlib, importlib.util, json, sys, types\n"
              "sys.path.insert(0, sys.argv[1])\n"
              "_BACKWARD_COMPAT_MAP = {'gemm_afp4wfp4': 'gemm.basic.gemm_afp4wfp4'}\n"
              "_warn_if_deprecated = lambda *args: None\n" + ast.unparse(function) + "\n"
              "sys.meta_path.insert(0, types.SimpleNamespace(find_spec=_backward_compat_find_spec))\n"
              "from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4\n"
              "basic = importlib.import_module('aiter.ops.triton.gemm.basic.gemm_afp4wfp4')\n"
              "import keeper\n"
              "print(json.dumps([len(keeper.executions), gemm_afp4wfp4 is basic.gemm_afp4wfp4, "
              "gemm_afp4wfp4() is basic.__dict__, gemm_afp4wfp4() is keeper.executions[0]]))\n")
    result = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path)], text=True,
                            capture_output=True, check=True)
    assert json.loads(result.stdout) == [2, True, False, True]
