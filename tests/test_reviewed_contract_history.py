"""Historical normalizers reject unreviewed task changes and restore test imports."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from reviewed_contract_history import (
    historical_kda_manifest, historical_ssd_runner, known_feedforward_geometry,
    SSD_BASELINE_SHA,
)
from task_import_isolation import isolated_task_imports

ROOT = Path(__file__).resolve().parents[1]


def test_ssd_inverse_is_exact_and_rejects_even_unrelated_byte_change():
    text = (ROOT / 'tasks/triton2triton/vllm/triton_ssd_chunk_scan/scripts/task_runner.py').read_text()
    assert hashlib.sha256(historical_ssd_runner(text).encode()).hexdigest() == SSD_BASELINE_SHA
    with pytest.raises(AssertionError):
        historical_ssd_runner(text + '\n# Unreviewed change\n')


@pytest.mark.parametrize('side', ['inter', 'intra'])
@pytest.mark.parametrize('tamper', ['control_shape', 'scored_seed', 'scored_check'])
def test_kda_inverse_rejects_control_and_original_workload_tampering(side, tamper):
    text = (ROOT / f'tasks/triton2triton/vllm/triton_kda_dot_kkt_{side}/workloads.json').read_text()
    assert len(json.loads(historical_kda_manifest(text))['cases']) == 5
    data = json.loads(text)
    if tamper == 'control_shape':
        data['semantic_control_cases'][0]['K'] = 38
    elif tamper == 'scored_seed':
        data['cases'][0]['params']['configuration'] += 1
    else:
        data['cases'][0]['checks'] = ['correctness']
    with pytest.raises(AssertionError):
        historical_kda_manifest(json.dumps(data, indent=2) + '\n')


def test_feedforward_geometry_requires_a_reviewed_complete_module(tmp_path):
    original = ROOT / 'tasks/torch2hip/gpumode/10024_Feedforward/pytorch_code_module/py_10024_Feedforward.py'
    source = tmp_path / 'source.py'
    source.write_bytes(original.read_bytes())
    assert len(known_feedforward_geometry(source)) == 5
    source.write_bytes(source.read_bytes() + b'\n# unreviewed\n')
    with pytest.raises(AssertionError, match='Unreviewed Feedforward'):
        known_feedforward_geometry(source)


def test_isolation_restores_existing_names_and_drops_new_task_modules(tmp_path, monkeypatch):
    task_root = tmp_path / 'tasks'
    task_root.mkdir()
    original = ModuleType('scripts.contract_checks')
    original.__file__ = str(task_root / 'old.py')
    unrelated = ModuleType('unrelated_source')
    unrelated.__file__ = str(tmp_path / 'unrelated.py')
    monkeypatch.setitem(sys.modules, 'scripts.contract_checks', original)
    monkeypatch.setitem(sys.modules, 'unrelated_source', unrelated)
    with isolated_task_imports(task_root):
        assert 'scripts.contract_checks' not in sys.modules
        assert sys.modules['unrelated_source'] is unrelated
        inserted = ModuleType('new_task_local')
        inserted.__file__ = str(task_root / 'new.py')
        sys.modules['new_task_local'] = inserted
        sys.modules['scripts.contract_checks'] = ModuleType('scripts.contract_checks')
    assert sys.modules['scripts.contract_checks'] is original
    assert sys.modules['unrelated_source'] is unrelated
    assert 'new_task_local' not in sys.modules


def test_isolation_also_restores_names_when_test_raises(tmp_path, monkeypatch):
    original = ModuleType('scripts')
    monkeypatch.setitem(sys.modules, 'scripts', original)
    with pytest.raises(ValueError, match='sentinel'):
        with isolated_task_imports(tmp_path):
            assert 'scripts' not in sys.modules
            raise ValueError('sentinel')
    assert sys.modules['scripts'] is original
