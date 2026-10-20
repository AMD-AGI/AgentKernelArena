"""CPU controls for the ROI size-2/4 native correctness paths."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / 'tasks/hip2hip/roiaware_pool3d'


def load_checks():
    path = TASK / 'scripts/reference_checks.py'
    spec = importlib.util.spec_from_file_location('roi_size24_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_harness():
    path = TASK / 'scripts/task_runner.py'
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'cpu_roiaware_pool3d']
    assert len(nodes) == 1
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(cpu_roiaware_pool3d=namespace['cpu_roiaware_pool3d'])


@pytest.mark.parametrize('out_size,channels', [(2, 3), (4, 3), (4, 6)])
def test_hand_known_answers_match_independent_cpu_oracle(out_size, channels):
    checks, h = load_checks(), cpu_harness()
    rois, points, features, expected = checks.negative_multivoxel_control(out_size, channels)
    assert features.min().item() < 0 and features.max().item() > 0
    for mode in ('max', 'avg'):
        actual = h.cpu_roiaware_pool3d(rois, points, features, out_size, mode)
        checks.known_answer(actual, expected[mode])
        assert torch.equal(expected[mode][0, -1, -1, -1],
                           torch.tensor([-3., -4.]).repeat((channels + 1) // 2)[:channels])
        assert (expected[mode][0, 0, 0, 0] < 0).all()


@pytest.mark.parametrize('out_size,channels,mode,fault', [
    (2, 3, 'max', 'clamp'), (2, 3, 'avg', 'clamp'),
    (4, 3, 'max', 'clamp'), (4, 3, 'avg', 'clamp'),
    (4, 6, 'max', 'clamp'), (4, 6, 'avg', 'clamp'),
    (2, 3, 'max', 'drop_last'), (4, 6, 'avg', 'drop_last'),
    (2, 3, 'max', 'include_face'), (4, 6, 'avg', 'include_face'),
])
def test_native_call_rejects_size_specific_wrong_output(monkeypatch, out_size, channels, mode, fault):
    checks, h = load_checks(), cpu_harness()
    def forward(rois, points, features, argmax, indices, output, mask, mode_id):
        current = rois.clone()
        if (fault == 'include_face' and output.shape[1] == out_size and
                output.shape[-1] == channels and mode_id == ('max', 'avg').index(mode)):
            current[:, 3:5] += .002
        got = h.cpu_roiaware_pool3d(current, points, features, output.shape[1], ('max', 'avg')[mode_id])
        if (output.shape[1] == out_size and output.shape[-1] == channels and
                mode_id == ('max', 'avg').index(mode)):
            if fault == 'clamp':
                got.clamp_min_(0)
            elif fault == 'drop_last':
                got[:, -1, -1, -1] = 0
        output.copy_(got)
    monkeypatch.setitem(sys.modules, 'kernel_loader', SimpleNamespace(
        roiaware_pool3d_ext=SimpleNamespace(forward=forward)))
    with pytest.raises(AssertionError):
        checks.check_negative_multivoxel_paths(h, device='cpu')


def test_honest_native_call_and_production_wiring(monkeypatch):
    checks, h = load_checks(), cpu_harness()
    def forward(rois, points, features, argmax, indices, output, mask, mode_id):
        output.copy_(h.cpu_roiaware_pool3d(
            rois, points, features, output.shape[1], ('max', 'avg')[mode_id]))
    monkeypatch.setitem(sys.modules, 'kernel_loader', SimpleNamespace(
        roiaware_pool3d_ext=SimpleNamespace(forward=forward)))
    assert checks.check_negative_multivoxel_paths(h, device='cpu') == [
        (2, 3, 'max'), (2, 3, 'avg'), (4, 3, 'max'), (4, 3, 'avg'),
        (4, 6, 'max'), (4, 6, 'avg')]
    tree = ast.parse((TASK / 'scripts/reference_checks.py').read_text())
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'check_additional_paths')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'check_negative_multivoxel_paths'
               for node in ast.walk(func))


def test_correct_native_output_with_mutated_input_rejected(monkeypatch):
    checks, h = load_checks(), cpu_harness()
    def forward(rois, points, features, argmax, indices, output, mask, mode_id):
        output.copy_(h.cpu_roiaware_pool3d(
            rois, points, features, output.shape[1], ('max', 'avg')[mode_id]))
        if output.shape[1] == 2 and mode_id == 0:
            features[0, 0] += 1
    monkeypatch.setitem(sys.modules, 'kernel_loader', SimpleNamespace(
        roiaware_pool3d_ext=SimpleNamespace(forward=forward)))
    with pytest.raises(ValueError, match='changed caller-owned features'):
        checks.check_negative_multivoxel_paths(h, device='cpu')
