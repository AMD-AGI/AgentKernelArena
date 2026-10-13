"""CPU controls for the task-local paged ALiBi mathematical reference."""
import ast
from pathlib import Path

import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_paged_prefix_prefill_alibi"


def reference_from_source():
    source = TASK / "scripts/task_runner.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "reference_attention_alibi"]
    assert len(functions) == 1
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["reference_attention_alibi"]


def test_zero_slope_still_masks_future_new_keys():
    reference = reference_from_source()
    q = torch.zeros(2, 1, 8)
    k = torch.zeros_like(q)
    v = torch.stack((torch.ones(1, 8), torch.full((1, 8), 100.)))
    empty_k = torch.empty(1, 0, 1, 8)
    empty_v = torch.empty_like(empty_k)
    out = reference(q, k, v, empty_k, empty_v,
                    torch.tensor([0, 2], dtype=torch.int32),
                    torch.tensor([2], dtype=torch.int32),
                    torch.zeros(1), 1, 0, 2, 1, 1, 8)
    torch.testing.assert_close(out[0], torch.ones(1, 8), rtol=0, atol=0)
    torch.testing.assert_close(out[1], torch.full((1, 8), 50.5), rtol=0, atol=0)
    # A missing causal mask would mix the future value into the first row.
    assert not torch.equal(out[0], torch.full((1, 8), 50.5))


def test_scored_geometry_and_measurement_contract_are_unchanged():
    source = (TASK / "scripts/task_runner.py").read_text()
    tree = ast.parse(source)
    values = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
              for target in node.targets if isinstance(target, ast.Name)}
    assert ast.literal_eval(values["TEST_SHAPES"]) == [
        (1, 64, 32, 8, 8, 64, 16),
        (2, 128, 64, 16, 4, 64, 16),
        (1, 256, 128, 32, 8, 128, 32),
        (2, 512, 64, 16, 16, 64, 32),
        (4, 64, 32, 8, 1, 64, 16),
    ]
    assert ast.literal_eval(values["WARMUP_ITERATIONS"]) == 10
    assert ast.literal_eval(values["BENCHMARK_ITERATIONS"]) == 100
