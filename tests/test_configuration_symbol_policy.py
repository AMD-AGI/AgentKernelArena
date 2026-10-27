"""Configuration-only helpers consume scalar metadata, never helper objects.

These structural checks use the public workspace guard without importing an
editable candidate or requiring Triton/GPU execution.
"""

import ast
from pathlib import Path
import shutil

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec
from test_harness_guard_v2 import LAUNCH_RETURN, launch_workspace


CONFIG_PROBE = (
    "class ConfigProbe:\n"
    "    def __class_getitem__(cls, key):\n"
    "        print('host helper executed')\n"
    "        return key\n"
)


def guarded_config(tmp_path, *, imports=""):
    root, source = launch_workspace(tmp_path)
    if imports:
        source.write_text(imports + source.read_text())
    return source, snapshot_workspace_harness(root)


@pytest.mark.parametrize("body", [
    # All returned leaves originate in caller-provided scalar metadata or literals.
    ("    if n >= 4096:\n"
     "        block = 256\n"
     "        warps = 8\n"
     "    else:\n"
     "        block = 64\n"
     "        warps = 4\n"
     "    return {'BLOCK': block, 'num_warps': warps}\n"),
    ("    choices = {'small': [32, 64], 'large': (128, 256)}\n"
     "    block = choices['large'][1] if n > 4096 else choices['small'][0]\n"
     "    return {'BLOCK': block, 'num_warps': 4}\n"),
    ("    block: int = ((n + 31) // 32) * 32\n"
     "    block = max(16, min(256, block))\n"
     "    enabled = bool(n > 0 and n <= 4096)\n"
     "    warps = int(float(4)) if enabled else abs(-8)\n"
     "    count = len((block, warps))\n"
     "    return {'BLOCK': block, 'num_warps': warps + count - 2}\n"),
    ("    if n > 4096:\n"
     "        return {'BLOCK': 256, 'num_warps': 8}\n"
     "    else:\n"
     "        block = 64\n"
     "    return {'BLOCK': block, 'num_warps': 4}\n"),
    ("    if n > 4096:\n"
     "        block = 256\n"
     "    else:\n"
     "        return {'BLOCK': 64, 'num_warps': 4}\n"
     "    return {'BLOCK': block, 'num_warps': 8}\n"),
])
def test_configuration_data_arithmetic_branches_and_local_collections(tmp_path, body):
    source, before = guarded_config(tmp_path)
    source.write_text(source.read_text().replace(LAUNCH_RETURN, body))
    verify_workspace_harness(before, discard_added=False)


@pytest.mark.parametrize("imports,expression", [
    ("import triton\n", "triton.next_power_of_2(triton.cdiv(n, 32))"),
    ("from triton import cdiv as ceil_div\n", "ceil_div(n, 32)"),
])
def test_unshadowed_pure_metadata_calls_are_allowed(tmp_path, imports, expression):
    source, before = guarded_config(tmp_path, imports=imports)
    body = f"    block = {expression}\n    return {{'BLOCK': block, 'num_warps': 4}}\n"
    source.write_text(source.read_text().replace(LAUNCH_RETURN, body))
    verify_workspace_harness(before, discard_added=False)


def test_scalar_signature_annotations_and_literal_defaults_are_allowed(tmp_path):
    source, before = guarded_config(tmp_path)
    source.write_text(source.read_text().replace(
        "def _launch_config(n):",
        "def _launch_config(n: int, mode: 'str' = 'default') -> dict:",
    ))
    verify_workspace_harness(before, discard_added=False)


@pytest.mark.parametrize("body", [
    "    block = ConfigProbe[0]\n" + LAUNCH_RETURN,
    "    alias = ConfigProbe\n    block = alias[0]\n" + LAUNCH_RETURN,
    "    return ConfigProbe\n",
    "    block = launch(n)\n" + LAUNCH_RETURN,
    "    block: ConfigProbe[0] = 64\n" + LAUNCH_RETURN,
    "    block = [item for item in (64, 128)][0]\n" + LAUNCH_RETURN,
    "    block = (lambda: 64)\n" + LAUNCH_RETURN,
    "    block = f'{ConfigProbe}'\n" + LAUNCH_RETURN,
    "    block = min(n, 64, key=ConfigProbe)\n" + LAUNCH_RETURN,
    "    block = min(*(n, 64))\n" + LAUNCH_RETURN,
    "    block = min(**{'iterable': (n, 64)})\n" + LAUNCH_RETURN,
])
def test_configuration_symbols_cannot_reach_host_objects_or_callbacks(tmp_path, body):
    source, before = guarded_config(tmp_path)
    # The helper is itself legal implementation code. Reading it from the
    # configuration function must still fail, before a subscript can dispatch it.
    source.write_text(source.read_text().replace(LAUNCH_RETURN, body) + CONFIG_PROBE)
    with pytest.raises(RuntimeError, match="configuration symbol"):
        verify_workspace_harness(before, discard_added=False)


@pytest.mark.parametrize("signature", [
    "def _launch_config(n, extra=ConfigProbe[0]):",
    "def _launch_config(n: ConfigProbe[0]):",
    "def _launch_config(n) -> ConfigProbe[0]:",
])
def test_configuration_signature_cannot_execute_implicit_helper_code(tmp_path, signature):
    source, before = guarded_config(tmp_path)
    changed = source.read_text().replace(
        "def _launch_config(n):", CONFIG_PROBE + signature,
    )
    source.write_text(changed)
    with pytest.raises(RuntimeError, match="configuration symbol"):
        verify_workspace_harness(before, discard_added=False)


@pytest.mark.parametrize("signature,body", [
    ("def _launch_config(n, min=None):", "    block = min(n, 64)\n"),
    ("def _launch_config(n):", "    min = n\n    block = min(n, 64)\n"),
    ("def _launch_config(n, triton=None):", "    block = triton.cdiv(n, 32)\n"),
    ("def _launch_config(n):", "    triton = n\n    block = triton.cdiv(n, 32)\n"),
    ("def _launch_config(n, ceil_div=None):", "    block = ceil_div(n, 32)\n"),
    ("def _launch_config(n):", "    ceil_div = n\n    block = ceil_div(n, 32)\n"),
])
def test_metadata_call_names_cannot_be_shadowed(tmp_path, signature, body):
    source, before = guarded_config(
        tmp_path, imports="import triton\nfrom triton import cdiv as ceil_div\n",
    )
    source.write_text(source.read_text().replace("def _launch_config(n):", signature)
                      .replace(LAUNCH_RETURN, body + LAUNCH_RETURN))
    with pytest.raises(RuntimeError, match="configuration symbol"):
        verify_workspace_harness(before, discard_added=False)


def test_branch_local_data_must_be_definitely_assigned(tmp_path):
    source, before = guarded_config(tmp_path)
    source.write_text(source.read_text().replace(
        LAUNCH_RETURN,
        "    if n > 4096:\n        block = 256\n"
        "    return {'BLOCK': block, 'num_warps': 4}\n",
    ))
    with pytest.raises(RuntimeError, match="configuration symbol"):
        verify_workspace_harness(before, discard_added=False)


def test_bmm_launch_configuration_cannot_be_used_as_a_definition_time_factory(tmp_path):
    """An editable decorator must not supply helper objects to the scalar caller."""
    original = Path(__file__).resolve().parents[1] / 'tasks/triton2triton/vllm/triton_bmm'
    root = tmp_path / 'task'
    shutil.copytree(original, root, ignore=shutil.ignore_patterns('__pycache__', 'build'))
    spec = load_task_spec(root / 'config.yaml', task_id='triton2triton/vllm/triton_bmm')
    before = snapshot_workspace_harness(root, task_spec=spec)
    source = root / 'source/triton_bmm.py'
    tree = ast.parse(source.read_text())
    config = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_launch_config')
    tree.body.remove(config)
    # This is valid pure-data code when the protected caller supplies its normal
    # scalar metadata. An editable definition-time caller could pass a class.
    config.body.insert(1, ast.parse("if dtype_name == 'probe':\n    probe = m[0]\n").body[0])
    kernel = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'bmm_kernel')
    kernel.decorator_list = [ast.parse(
        "triton.jit(do_not_specialize=["
        "_launch_config('probe', 1, ConfigProbe, 16, 16)['num_stages']])",
        mode='eval',
    ).body]
    index = tree.body.index(kernel)
    tree.body[index:index] = [ast.parse(CONFIG_PROBE).body[0], config]
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    # Reject it structurally, without importing candidate code or triggering its
    # __class_getitem__ during decorator evaluation.
    with pytest.raises(RuntimeError, match='Unapproved definition-time factory'):
        verify_workspace_harness(before, discard_added=False)
