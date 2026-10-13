"""CPU controls for negative-heading points-in-boxes behavior."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/hip2hip/points_in_boxes"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_oracle():
    """Load the two real task CPU reference functions without runner side effects."""
    source = TASK / "scripts/task_runner.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("cpu_points_in_boxes_part", "cpu_points_in_boxes_all")]
    assert len(functions) == 2
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**{node.name: namespace[node.name] for node in functions})


@pytest.fixture
def controls():
    return load(TASK / "scripts/reference_checks.py", "points_heading_reference_controls"), cpu_oracle()


def native(h, kind, heading_transform=None, mutate=None):
    oracle = getattr(h, f"cpu_points_in_boxes_{kind}")

    def run(points, boxes):
        selected = boxes.clone()
        if heading_transform is not None:
            selected[..., 6] = heading_transform(selected[..., 6])
        output = oracle(points, selected)
        if mutate is not None:
            mutate(points, boxes)
        return output

    return run


def test_ordered_known_answers_and_honest_both_modes(controls):
    checks, h = controls
    points, boxes, expected_part, expected_all = checks.check_heading_reference(h)
    assert tuple(boxes[:, 0, 6].tolist()) == pytest.approx(
        (-.7, -torch.pi / 2, -2.2, -torch.pi, .7, torch.pi / 2, 2.2, torch.pi))
    assert expected_part.shape == (8, 8)
    assert expected_all.shape == (8, 8, 1)
    assert torch.equal(h.cpu_points_in_boxes_part(points, boxes), expected_part)
    checks.check_heading_controls(h, native(h, "part"), native(h, "all"), device="cpu")


@pytest.mark.parametrize("transform", [torch.abs, torch.neg, torch.zeros_like])
@pytest.mark.parametrize("mode", ["part", "all"])
def test_wrong_heading_in_each_native_mode_is_rejected(controls, transform, mode):
    checks, h = controls
    part = native(h, "part", transform if mode == "part" else None)
    all_boxes = native(h, "all", transform if mode == "all" else None)
    with pytest.raises(AssertionError):
        checks.check_heading_controls(h, part, all_boxes, device="cpu")


@pytest.mark.parametrize("kind", ["part", "all"])
def test_correct_output_then_input_mutation_is_rejected(controls, kind):
    checks, h = controls
    changed_boxes = []

    def change_zero_sign(points, boxes):
        changed_boxes.append(boxes)
        boxes[0, 0, 1] = -0.0  # Numerically equal to the original +0.0.

    part = native(h, "part", mutate=change_zero_sign if kind == "part" else None)
    all_boxes = native(h, "all", mutate=change_zero_sign if kind == "all" else None)
    with pytest.raises(ValueError, match="changed a caller-owned input"):
        checks.check_heading_controls(h, part, all_boxes, device="cpu")
    original_boxes = checks.heading_control_inputs()[1]
    assert len(changed_boxes) == 1
    assert torch.equal(changed_boxes[0].view(torch.uint8), original_boxes.view(torch.uint8))


def test_sign_erasing_cpu_reference_is_rejected(controls):
    checks, h = controls
    original = h.cpu_points_in_boxes_part
    h.cpu_points_in_boxes_part = lambda points, boxes: native(
        SimpleNamespace(cpu_points_in_boxes_part=original), "part", torch.abs)(points, boxes)
    with pytest.raises(AssertionError):
        checks.check_heading_reference(h)


def test_new_controls_are_additional_to_existing_random_and_score_paths():
    tree = ast.parse((TASK / "scripts/reference_checks.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "check_additional_paths")
    names = [node.func.id for node in ast.walk(function)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "check_heading_controls" in names
    assert "enumerate" in names  # Existing randomized TEST_SHAPES loop retained.
    import json
    rows = json.loads((TASK / "workload.json").read_text())["cases"]
    assert len(rows) == 20
