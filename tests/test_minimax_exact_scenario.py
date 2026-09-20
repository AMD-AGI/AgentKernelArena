"""CPU checks for MiniMax observed-control scoring and proxy exclusion."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = {p.parent.name: p.parent for p in (ROOT / 'tasks/head_kernels/minimax-m3-mxfp4').rglob('config.yaml')}


def load_cases(name):
    task = TASKS[name]
    spec = importlib.util.spec_from_file_location('minimax_exact_cases', task / 'ut/cases.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return task, module, json.loads((task / 'ut/meta.json').read_text())


def test_prefill_timing_uses_the_full_observed_record(monkeypatch):
    task, cases, meta = load_cases('gqa_share_sparse_fwd_kernel')
    records = json.loads((task / 'ut/generated_cases.json').read_text())['records']
    selected = records[2]
    geometry = json.loads((task / 'ut/timing_geometry.json').read_text())['records']
    monkeypatch.setattr(cases, '_geo', lambda: geometry)
    calls = []
    def build(record, seed, torch, device):
        calls.append((record, seed, device))
        return (), record['kwargs']  # Do not allocate multi-GB GPU buffers in this CPU wiring test.
    helper = SimpleNamespace(load_contract=lambda _: {'records': records}, build_record=build)
    monkeypatch.setitem(sys.modules, 'generated_contract', helper)
    monkeypatch.setattr(cases, '_build_args', lambda *a, **kw: pytest.fail('compacting builder was used for timing'))
    rows = cases.timing_cases(None, meta)
    assert len(rows) == 1 and rows[0]['sig'] == 'prefill_m8192_s1'
    assert calls == [(selected, 0, 'cuda')]
    args = rows[0]['args']
    assert args['q']['shape'] == [8192, 8, 128]
    assert args['k_cache']['shape'] == args['v_cache']['shape'] == [4358330, 1, 128]
    assert args['req_to_token']['shape'] == [4097, 11268]
    assert args['req_to_token']['stride'] == [11268, 1]
    assert args['req_to_token']['row_indices'] == [4]
    assert args['slot_ids']['dtype'] == 'torch.int64' and args['slot_ids']['values'] == [4]
    assert args['topk_idx']['values'] == selected['kwargs']['topk_idx']['values']


def test_prefill_never_promotes_warmup_or_missing_capture(monkeypatch):
    task, cases, meta = load_cases('gqa_share_sparse_fwd_kernel')
    geometry = json.loads((task / 'ut/timing_geometry.json').read_text())['records']
    monkeypatch.setattr(cases, '_geo', lambda: geometry[:2])
    with pytest.raises(RuntimeError, match='M=8192'):
        cases.timing_buckets(None, meta)
    monkeypatch.setattr(cases, '_geo', lambda: geometry)
    monkeypatch.setitem(sys.modules, 'generated_contract', SimpleNamespace(load_contract=lambda _: {'records': []}))
    with pytest.raises(RuntimeError, match='control record is missing'):
        cases.timing_cases(None, meta)


@pytest.mark.parametrize('name', ['decode_score_kernel', 'gqa_share_sparse_decode_kernel'])
def test_decode_proxies_remain_validation_only(name, monkeypatch):
    task, cases, meta = load_cases(name)
    geometry = json.loads((task / 'ut/timing_geometry.json').read_text())['records']
    monkeypatch.setattr(cases, '_geo', lambda: geometry)
    assert len(cases.validation_buckets(None, meta)) == 2
    assert len(cases.random_shapes(None, meta)) == 4
    assert len(cases.replay_shapes(None, meta)) == 2
    assert meta['performance_contract']['case_ids'] == []
    with pytest.raises(RuntimeError, match='no captured per-call control'):
        cases.timing_cases(None, meta)
    monkeypatch.setattr(cases, '_geo', lambda: [x for x in geometry if x.get('source') == 'recorded'])
    with pytest.raises(RuntimeError, match='warmup is not a substitute'):
        cases.validation_buckets(None, meta)


@pytest.mark.parametrize('name', ['decode_score_kernel', 'gqa_share_sparse_decode_kernel'])
def test_decode_performance_blocks_before_gpu_startup(name, tmp_path):
    task = TASKS[name]
    (tmp_path / 'scripts').mkdir(); (tmp_path / 'ut').mkdir()
    for filename in ['generated_task_runner.py', 'generated_correctness.py', 'task_runner.py']:
        shutil.copyfile(task / 'scripts' / filename, tmp_path / 'scripts' / filename)
    shutil.copyfile(task / 'ut/meta.json', tmp_path / 'ut/meta.json')
    proc = subprocess.run([sys.executable, 'scripts/generated_task_runner.py', 'performance'],
                          cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 1 and 'Performance: BLOCKED' in proc.stdout
    assert not proc.stderr
    report = json.loads((tmp_path / 'build/performance_report.json').read_text())
    assert report['status'] == 'fail' and report['test_cases'] == []
    assert report['reason'] == 'missing_observed_workload_controls'
