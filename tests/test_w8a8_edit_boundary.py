"""Real guard checks; no GPU execution or performance qualification here."""
from pathlib import Path
import shutil
import pytest
import yaml
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/triton2triton/vllm/triton_w8a8_block_scaled_mm'
SOURCE = 'source/triton_w8a8_block_scaled_mm.py'

def workspace(tmp_path):
    task=tmp_path/'task'
    shutil.copytree(TASK, task)
    spec=load_task_spec(task/'config.yaml', task_id='triton2triton/vllm/triton_w8a8_block_scaled_mm')
    return task, snapshot_workspace_harness(task, task_spec=spec)

def test_actual_guard_keeps_device_body_and_compiler_helpers_editable(tmp_path):
    task, guard=workspace(tmp_path)
    source=task/SOURCE
    text=source.read_text()
    needle='    pid = tl.program_id(axis=0)'
    assert text.count(needle)==1
    source.write_text(text.replace(needle, needle+'\n    _implementation_scalar = 0')+
                      '\n\n@triton.jit\ndef _implementation_helper(x):\n    return x * 1.0\n')
    verify_workspace_harness(guard, discard_added=False)

@pytest.mark.parametrize('kind', ['module_registry', 'wrapper_return', 'dtype_helper', 'numerical_checker'])
def test_actual_guard_rejects_changes_to_execution_wrapper_dtype_and_oracle(tmp_path, kind):
    task, guard=workspace(tmp_path)
    source=task/SOURCE
    text=source.read_text()
    if kind=='module_registry':
        source.write_text(text+'\nimport sys\n_module_registry = sys.modules\n')
    elif kind=='wrapper_return':
        assert text.count('    return C\n')==1
        source.write_text(text.replace('    return C\n','    return C * 0\n'))
    elif kind=='dtype_helper':
        source.write_text(text.replace('    return torch.float8_e4m3fn\n','    return torch.float32\n'))
    else:
        checker=task/'_arena_checks.py'
        checker.write_text(checker.read_text()+'\n_override_reference = True\n')
    with pytest.raises(RuntimeError):
        verify_workspace_harness(guard, discard_added=False)

def test_historical_whole_file_boundary_accepts_same_module_registry_route(tmp_path):
    task=tmp_path/'legacy_task'
    shutil.copytree(TASK, task)
    config=task/'config.yaml'
    data=yaml.safe_load(config.read_text())
    data['candidate']['editable']=[SOURCE]
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    spec=load_task_spec(config, task_id='triton2triton/vllm/triton_w8a8_block_scaled_mm')
    guard=snapshot_workspace_harness(task, task_spec=spec)
    source=task/SOURCE
    source.write_text(source.read_text()+'\nimport sys\n_module_registry = sys.modules\n')
    verify_workspace_harness(guard, discard_added=False)
