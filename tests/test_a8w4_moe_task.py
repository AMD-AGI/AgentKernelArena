"""Independent CPU controls for the A8W4 oracle and replay rejection gates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")
TASK = Path(__file__).resolve().parents[1] / "tasks/torch2flydsl/moe_a8w4_kernel"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TASK / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantized_two_expert_known_answer():
    model = load("a8w4_oracle_control", "model.py").Model(32, 32, 3, 2)
    with torch.no_grad():
        model.w1.fill_(99)  # Unselected experts must not contribute.
        model.w2.fill_(99)
        model.w1[0].fill_(1)
        model.w2[0].fill_(1)
        model.w1[1].fill_(0.5)
        model.w2[1].fill_(0.25)
        output = model.forward_with_routing(
            torch.ones(1, 32, dtype=torch.bfloat16),
            torch.tensor([[0.25, 0.75]]), torch.tensor([[0, 1]], dtype=torch.int32))
    # Expert 0: 32 * (32 * silu(32)) = 32768 after rounding.
    # Expert 1: 32 * .25 * (16 * silu(16)) = 2048 after rounding.
    assert torch.equal(output, torch.full_like(output, 9728))


def test_original_fp4_scale_boundary_known_answer():
    oracle = load("a8w4_scale_control", "model.py")
    values = torch.full((1, 32), 1.5, dtype=torch.bfloat16)
    values[0, 0] = 6
    values[0, 1] = -1.5
    expected = torch.full((1, 32), 2.0)
    expected[0, 0] = 6
    expected[0, 1] = -2
    assert torch.equal(oracle._mxfp4_dequant(values), expected)


def test_bf16_oracle_does_not_overflow_through_fp16():
    checks = load("a8w4_range_control", "test_kernel_harness.py")
    # A literal 262144 result exceeds FP16 range but remains finite in BF16.
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(TASK))
        patch.setitem(sys.modules, "model", load("a8w4_range_model", "model.py"))
        assert checks.reference_controls()[0]["known_answer"] == "PASS"


def test_capture_output_is_poisoned_before_collector_binding():
    checks = load("a8w4_poison_control", "test_kernel_harness.py")
    output = torch.ones(1, 32, dtype=torch.bfloat16)
    invocation = checks.ObservedInvocation(SimpleNamespace(call=lambda kernel: output), None)
    invocation.prepare()
    assert invocation.call() is output
    invocation.prepare()  # No TimedRun has been bound yet.
    assert torch.isnan(output).all()


def test_task_manifest_satisfies_the_framework_protocol(monkeypatch, capsys):
    from src.task_protocol import CaseManifest, parse_command_result
    with monkeypatch.context() as patch:
        patch.setattr(sys, "path", list(sys.path))
        patch.setitem(sys.modules, "test_kernel_harness",
                      load("a8w4_manifest_harness", "test_kernel_harness.py"))
        patch.setitem(sys.modules, "model", load("a8w4_manifest_model", "model.py"))
        runner = load("a8w4_manifest_runner", "scripts/evaluate.py")
        assert runner.run(["validate-task"]) == 0
    result = parse_command_result(capsys.readouterr().out, role="task",
                                  action="validate-task", returncode=0)
    CaseManifest.from_result(result)
    assert len(result.cases) == 3
    assert sum("performance" in row["checks"] for row in result.cases) == 2


@pytest.mark.parametrize("bad", ["zero", "double", "nan", "dtype", "shape"])
def test_output_gate_rejects_wrong_answers(bad):
    checks = load("a8w4_output_control", "test_kernel_harness.py")
    expected = torch.ones(1, 32, dtype=torch.bfloat16)
    wrong = {"zero": torch.zeros_like(expected), "double": expected * 2,
             "nan": torch.full_like(expected, float("nan")),
             "dtype": expected.float(), "shape": expected[:, :16]}[bad]
    with pytest.raises(AssertionError):
        checks.compare(wrong, expected)


def make_cpu_fixture():
    checks = load("a8w4_replay_control", "test_kernel_harness.py")
    fixture = checks.Fixture.__new__(checks.Fixture)
    fixture.model = SimpleNamespace(experts=4)
    fixture.inputs = (torch.ones(1, 32, dtype=torch.bfloat16),
                      torch.ones(1), torch.ones(1),
                      torch.tensor([[0.25, 0.75]]), torch.tensor([[0, 1]], dtype=torch.int32))
    fixture.originals = tuple(x.clone() for x in fixture.inputs)
    fixture.signatures = tuple((x.shape, x.stride(), x.dtype, x.device, x.data_ptr())
                               for x in fixture.inputs)
    fixture.reference = lambda **kwargs: fixture.inputs[0].clone() * (
        1 + fixture.inputs[4].sum().item()) * fixture.inputs[1].sum() * fixture.inputs[2].sum()
    fixture.expected = fixture.reference()
    return fixture


def test_stale_graph_output_rejected_and_inputs_restored():
    fixture = make_cpu_fixture()
    timed = SimpleNamespace(bound=True, outputs=fixture.expected.clone())
    timed.rerun = lambda: timed.outputs
    with pytest.raises(AssertionError, match="Nonfinite"):
        fixture.verify_replay(timed)
    fixture.unchanged()


def test_graph_replay_checks_all_changed_input_draws():
    fixture = make_cpu_fixture()
    timed = SimpleNamespace(bound=True, outputs=fixture.expected.clone())
    draws = []
    def replay():
        draws.append((fixture.inputs[0].clone(), fixture.inputs[4].clone(), fixture.inputs[1].clone()))
        timed.outputs.copy_(fixture.reference())
        return timed.outputs
    timed.rerun = replay
    result = fixture.verify_replay(timed)
    assert result["replay_input_draws"] == 3
    assert not torch.equal(draws[0][0], fixture.originals[0])
    assert not torch.equal(draws[1][1], fixture.originals[4])
    assert not torch.equal(draws[2][2], fixture.originals[1])
    fixture.unchanged()


def test_read_only_weight_mutation_rejected():
    fixture = make_cpu_fixture()
    fixture.inputs[1].zero_()
    with pytest.raises(AssertionError, match="Read-only"):
        fixture.unchanged()
