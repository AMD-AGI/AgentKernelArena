"""CPU known-answer counterexamples for ROI task controls."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'tasks/hip2hip'


def load_checks(task):
    path = ROOT / task / 'scripts/reference_checks.py'
    spec = importlib.util.spec_from_file_location(f'{task}_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_functions(task, names):
    path = ROOT / task / 'scripts/task_runner.py'
    source = ast.parse(path.read_text())
    body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == set(names)
    env = {'torch': torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), env)
    return SimpleNamespace(**{name: env[name] for name in names})


def test_roiaware_all_negative_max_and_four_excluded_faces():
    checks = load_checks('roiaware_pool3d')
    h = cpu_functions('roiaware_pool3d', ('cpu_roiaware_pool3d',))
    checks.self_test(h)
    rois, points, features, known = checks.all_negative_faces_control()
    assert torch.all(features[:3] < 0)
    assert torch.all(features[3:] > 0)
    assert torch.equal(known['max'].flatten(), torch.tensor([-3., -2.]))
    assert torch.equal(known['avg'].flatten(), torch.tensor([-5., -4.]))
    for mode in ('max', 'avg'):
        checks.close(h.cpu_roiaware_pool3d(rois, points, features, 1, mode), known[mode])
        with pytest.raises(AssertionError):
            checks.close(torch.maximum(known[mode], torch.zeros_like(known[mode])), known[mode])
        with pytest.raises(AssertionError):
            checks.close(torch.full_like(known[mode], 104.), known[mode])


def test_roipoint_broad_headings_order_and_empty_flag():
    checks = load_checks('roipoint_pool3d')
    h = cpu_functions('roipoint_pool3d', ('check_point_in_box', 'cpu_roipoint_pool3d'))
    checks.self_test(h)
    points, features, boxes, expected, flags = checks.check_heading_reference(h)
    assert torch.equal(flags, torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.int32))
    actual, actual_flags = h.cpu_roipoint_pool3d(points, features, boxes, 5)
    checks.close(actual, expected)
    checks.close(actual_flags, flags)
    with pytest.raises(AssertionError):
        checks.close(actual[:, :, [1, 0, 2, 3, 4]], expected)
    with pytest.raises(AssertionError):
        checks.close(torch.zeros_like(actual_flags), flags)
    unrotated = boxes.clone()
    unrotated[..., 6] = 0
    wrong, _ = h.cpu_roipoint_pool3d(points, features, unrotated, 5)
    with pytest.raises(AssertionError):
        checks.close(wrong, expected)
