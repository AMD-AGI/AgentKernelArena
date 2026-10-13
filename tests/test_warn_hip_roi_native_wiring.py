"""CPU device-redirection probes of the actual ROI native-control call paths."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location('roi_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_harness(task):
    import ast
    path = ROOT / 'tasks/hip2hip' / task / 'scripts/task_runner.py'
    wanted = {'cpu_roiaware_pool3d'} if task == 'roiaware_pool3d' else {
        'check_point_in_box', 'cpu_roipoint_pool3d'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(TEST_SHAPES=[], **{name: namespace[name] for name in wanted})


def redirect_cuda(monkeypatch, checks):
    for name in ('tensor', 'zeros', 'empty', 'zeros_like'):
        original = getattr(torch, name)
        def factory(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, factory)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda value, *a, **kw: value)
    contract = checks.contract
    monkeypatch.setattr(checks, 'contract', lambda actual, expected, gpu=False:
                        contract(actual, expected, gpu=False))


@pytest.mark.parametrize('fault', [None, 'zero_clamp', 'y_face'])
def test_roiaware_native_controls_reject_missing_negative_or_included_face(monkeypatch, fault):
    checks = load(ROOT / 'tasks/hip2hip/roiaware_pool3d/scripts/reference_checks.py')
    h = cpu_harness('roiaware_pool3d')
    redirect_cuda(monkeypatch, checks)
    seen = []
    def forward(rois, points, features, argmax, indices, output, mask, mode_id):
        out_size = output.shape[1]
        assert output.shape[1:4] == (out_size, out_size, out_size)
        mode = ('max', 'avg')[mode_id]
        seen.append((out_size, mode))
        current = rois.clone()
        # Keep the original size-1 negatives targeted at their original
        # boundary/negative controls; size-2/4 have separate focused negatives.
        if fault == 'y_face' and out_size == 1:
            current[:, 4] += .002
        got = h.cpu_roiaware_pool3d(current, points, features, out_size, mode)
        if fault == 'zero_clamp' and out_size == 1:
            got.clamp_min_(0)
        output.copy_(got)
    monkeypatch.setitem(sys.modules, 'kernel_loader', SimpleNamespace(
        roiaware_pool3d_ext=SimpleNamespace(forward=forward)))
    if fault is None:
        checks.check_additional_paths(h)
        assert {(size, mode) for size, mode in seen} == {
            (size, mode) for size in (1, 2, 4) for mode in ('max', 'avg')}
    else:
        with pytest.raises(AssertionError):
            checks.check_additional_paths(h)
        assert {(2, mode) for mode in ('max', 'avg')} <= set(seen)
        assert {(4, mode) for mode in ('max', 'avg')} <= set(seen)
        assert any(size == 1 for size, _ in seen)


@pytest.mark.parametrize('fault', [None, 'y_face', 'heading_clamp'])
def test_roipoint_native_controls_reject_included_face_and_clamped_heading(monkeypatch, fault):
    checks = load(ROOT / 'tasks/hip2hip/roipoint_pool3d/scripts/reference_checks.py')
    h = cpu_harness('roipoint_pool3d')
    redirect_cuda(monkeypatch, checks)
    def forward(points, boxes, features, output, flags, assignments, indices):
        current = boxes.clone()
        if fault == 'y_face' and current.shape[1] == 3:
            current[:, 0, 4] += .002
        if fault == 'heading_clamp' and current.shape[1] == 5:
            current[..., 6].clamp_(-.5, .5)
        got, empty = h.cpu_roipoint_pool3d(points, features, current, output.shape[2])
        output.copy_(got)
        flags.copy_(empty)
    monkeypatch.setitem(sys.modules, 'kernel_loader', SimpleNamespace(
        roipoint_pool3d_ext=SimpleNamespace(forward=forward)))
    if fault is None:
        checks.check_additional_paths(h)
    else:
        with pytest.raises(AssertionError):
            checks.check_additional_paths(h)
