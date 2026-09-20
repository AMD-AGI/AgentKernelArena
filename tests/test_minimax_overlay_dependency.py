"""CPU import/build regressions for the captured MiniMax baseline overlay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = {p.parent.name: p.parent for p in (ROOT / 'tasks/head_kernels/minimax-m3-mxfp4').rglob('config.yaml')}
PREFILL = 'sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse'
DECODE = 'sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse'
RELATIVE = '_patched/' + PREFILL + '.py'
DIGEST = 'bf26c1d9ad6ad6d7716727c1ab55d09a786abf59a2cd8a7a390b90285f1e8421'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def image_modules(monkeypatch):
    def package(name):
        if name in sys.modules:
            return sys.modules[name]
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(package(parent), child, module)
        return module
    for name in [PREFILL.rsplit('.', 1)[0], DECODE.rsplit('.', 1)[0],
                 'sglang.kernels.ops.attention.minimax_sparse.common']:
        package(name)
    utils = package('sglang.kernels.ops.attention.minimax_sparse.common.utils')
    for name in ['check_sparse_kv_fp8', 'get_cu_seqblocks', 'robust_allocator', 'sparse_out_dtype', 'unit_scale']:
        setattr(utils, name, lambda *a, **k: None)
    triton = ModuleType('triton'); triton.__path__ = []
    triton.jit = lambda fn: fn
    triton.heuristics = triton.autotune = lambda *a, **k: lambda fn: fn
    triton.Config = lambda *a, **k: SimpleNamespace()
    tl = ModuleType('triton.language'); tl.constexpr = object()
    triton.language = tl
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', tl)
    # Capture keys so monkeypatch also removes the real overlay modules afterwards.
    monkeypatch.setitem(sys.modules, PREFILL, ModuleType(PREFILL))
    native = ModuleType(DECODE)
    native.flash_decode_with_gqa_share_sparse = lambda: 'baseline'
    monkeypatch.setitem(sys.modules, DECODE, native)
    return native


def task_copy(tmp_path, name):
    original = TASKS[name]
    ut = tmp_path / 'ut'; ut.mkdir()
    shutil.copytree(original / 'ut/baseline_overlay', ut / 'baseline_overlay')
    for filename in ['harness_lib.py', 'overlay_setup.py']:
        shutil.copyfile(original / 'ut' / filename, ut / filename)
    (ut / 'kernel_src').mkdir()
    return original, ut


def test_all_three_declared_overlay_maps_are_complete():
    for name, task in TASKS.items():
        tool = load('overlay_validation', task / 'ut/overlay_setup.py')
        tool._validate_overlay_modules(str(task / 'ut/baseline_overlay'))
        data = json.loads((task / 'ut/baseline_overlay/_overlay_manifest.json').read_text())
        if name == 'gqa_share_sparse_fwd_kernel':
            assert data['modules'] == []
        else:
            assert data['modules'][0]['sha256'] == DIGEST
            assert hashlib.sha256((task / 'ut/baseline_overlay' / RELATIVE).read_bytes()).hexdigest() == DIGEST


def test_real_build_preserves_dependency_and_candidate_independence(tmp_path, image_modules):
    task, ut = task_copy(tmp_path, 'gqa_share_sparse_decode_kernel')
    # Preserve the actual candidate module body, adding a CPU sentinel at its public seam.
    data = (task / 'source/topk_sparse.py').read_text()
    (ut / 'kernel_src/topk_sparse.py').write_text(data + '\ndef flash_decode_with_gqa_share_sparse(*a, **kw): return "candidate"\n')
    meta = json.loads((task / 'ut/meta.json').read_text())
    harness = load('overlay_harness', ut / 'harness_lib.py')
    baseline, candidate = harness.build_candidate_overlay(str(ut), meta)
    for overlay in [Path(baseline), Path(candidate)]:
        assert hashlib.sha256((overlay / RELATIVE).read_bytes()).hexdigest() == DIGEST
    load('baseline_sitecustomize', Path(baseline) / 'sitecustomize.py')
    pref = sys.modules[PREFILL]
    assert callable(pref.flash_prefill_with_gqa_share_sparse)
    assert sys.modules[DECODE].flash_decode_with_gqa_share_sparse() == 'baseline'
    load('candidate_sitecustomize', Path(candidate) / 'sitecustomize.py')
    assert callable(sys.modules[PREFILL].flash_prefill_with_gqa_share_sparse)
    assert sys.modules[DECODE].flash_decode_with_gqa_share_sparse() == 'candidate'
    namespace = {}
    exec('from ' + PREFILL + ' import flash_prefill_with_gqa_share_sparse\nfrom ' + DECODE + ' import flash_decode_with_gqa_share_sparse', namespace)
    assert namespace['flash_decode_with_gqa_share_sparse']() == 'candidate'


@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_missing_or_changed_baseline_dependency_fails_before_build(tmp_path, damage):
    task, ut = task_copy(tmp_path, 'decode_score_kernel')
    dependency = ut / 'baseline_overlay' / RELATIVE
    if damage == 'missing':dependency.unlink()
    else:dependency.write_text('not the captured dependency')
    shutil.copyfile(task / 'source/flash_with_topk_idx.py', ut / 'kernel_src/flash_with_topk_idx.py')
    harness = load('damaged_overlay_harness', ut / 'harness_lib.py')
    meta = json.loads((task / 'ut/meta.json').read_text())
    with pytest.raises(RuntimeError, match='candidate overlay build failed'):
        harness.build_candidate_overlay(str(ut), meta)


def test_failed_injection_is_fatal_and_does_not_poison_module(tmp_path, image_modules):
    _, ut = task_copy(tmp_path, 'decode_score_kernel')
    dependency = ut / 'baseline_overlay' / RELATIVE
    dependency.unlink()
    previous = sys.modules[PREFILL]
    with pytest.raises(SystemExit, match='required overlay module'):
        load('missing_sitecustomize', ut / 'baseline_overlay/sitecustomize.py')
    assert sys.modules[PREFILL] is previous
