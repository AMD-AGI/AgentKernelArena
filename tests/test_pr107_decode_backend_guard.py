"""The actual v2 guard protects the decode wrapper that invokes the kernel."""
import ast
import importlib.util
from pathlib import Path
import shutil

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec

TASK = Path(__file__).resolve().parents[1] / 'tasks/triton2triton/vllm/triton_prepare_eagle_docode'


@pytest.fixture
def guarded(tmp_path):
    root = tmp_path / 'task'
    shutil.copytree(TASK, root)
    spec = load_task_spec(root / 'config.yaml', task_id='triton2triton/vllm/triton_prepare_eagle_docode')
    guard = snapshot_workspace_harness(root, task_spec=spec)
    return root, root / spec.candidate.editable[0].path, guard


def write_tree(path, tree):
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + '\n')


def test_real_guard_allows_declared_kernel_and_new_implementation_helper(guarded):
    _, source, guard = guarded
    tree = ast.parse(source.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_eagle_docode_kernel')
    kernel.body.append(ast.Pass())
    tree.body.extend(ast.parse('@triton.jit\ndef copy_helper(x):\n    return x\n').body)
    write_tree(source, tree)
    verify_workspace_harness(guard)


@pytest.mark.parametrize('attack', ['torch_wrapper', 'unused_kernel', 'rebind_triton', 'rebind_wrapper'])
def test_real_guard_rejects_wrapper_backend_bypass_even_when_jit_inspection_passes(guarded, attack):
    root, source, guard = guarded
    tree = ast.parse(source.read_text())
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_eagle_decode')
    if attack == 'torch_wrapper':
        wrapper.body = ast.parse('''n = draft_tokens.numel()
input_ids[:n].copy_(draft_tokens)
input_hidden_states[:n].copy_(output_hidden_states[last_token_indices])
positions[:n].copy_((positions[:n] + 1).clamp(max=max_model_len - 1))
seq_lens[:n].copy_((target_seq_lens - num_rejected + 1).clamp(max=max_model_len))
seq_lens[n:max_num_reqs].zero_()
query_start_loc.copy_(torch.arange(max_num_reqs + 1, device=query_start_loc.device).clamp(max=n))
''').body
    elif attack == 'unused_kernel':
        wrapper.body = [ast.Return(ast.Constant(None))]
    elif attack == 'rebind_triton':
        tree.body.insert(0, ast.Import(names=[ast.alias(name='torch', asname='triton')]))
    else:
        tree.body.extend(ast.parse('prepare_eagle_decode = lambda *args: None').body)
    write_tree(source, tree)
    loader = importlib.util.spec_from_file_location('decode_eval', root / '_arena_eval.py')
    adapter = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(adapter)
    # Merely checking that a JIT definition exists is insufficient; the shared
    # guard must independently reject changes to the wrapper/import boundary.
    assert adapter.inspect_candidate(adapter.load_manifest(), require_implemented=True) == 'implemented'
    with pytest.raises(RuntimeError, match='[Hh]arness|protected'):
        verify_workspace_harness(guard)
