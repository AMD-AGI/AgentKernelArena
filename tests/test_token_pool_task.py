"""CPU regressions for the protected token-pool contract and replay checks."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/triton2triton/sglang/write_req_to_token_pool"
spec = importlib.util.spec_from_file_location("token_pool_harness_test", TASK / "test_kernel_harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def control(name="mixed_boundaries"):
    return next(row for row in harness.manifest() if row["test_case_id"] == name)


def test_original_scored_workloads_and_unscored_controls():
    rows = harness.manifest()
    scored = [r for r in rows if "performance" in r["checks"]]
    assert [r["test_case_id"] for r in scored] == ["c2", "c32", "c64"]
    for r, b in zip(scored, (2, 32, 64)):
        assert r["shape"] == [2048, 202756]
        assert r["params"] == dict(B=b, max_batch=2048, max_context_len=202756,
                                   seed=42, pre_lens=[16]*b, extend_lens=[1008]*b)
    assert len(rows) == 7
    assert harness.reference_controls()[0]["negative_output"] == "rejected"


@pytest.mark.parametrize("name", ["mixed_boundaries", "all_empty", "prefix_only", "extend_only"])
def test_reference_and_preparation_preserve_guards(name):
    f = harness.Fixture(control(name), device="cpu")
    f.output.fill_(123)
    f.prepare()  # Must run even before a TimedRun is bound.
    rows, _, pre, seq, extend, cache, *prefixes = f.inputs
    offset = 0
    for row, pl, sl, ex, prefix in zip(rows, pre, seq, extend, prefixes):
        assert torch.all(f.output[row, :sl] == harness.POISON)
        f.output[row, :pl] = prefix[:pl].to(torch.int32)
        f.output[row, pl:sl] = cache[offset:offset+ex].to(torch.int32)
        offset += int(ex)
    assert f.check(f.output)["exact_match"]
    f.output[0, -1] += 1
    with pytest.raises(AssertionError, match="Exact table"):
        f.check(f.output)


def test_noop_wrong_output_and_pointee_mutation_are_rejected():
    f = harness.Fixture(control(), device="cpu")
    f.prepare()
    with pytest.raises(AssertionError, match="Exact table"):
        f.check(f.output)
    f.output.copy_(f.expected)
    with pytest.raises(AssertionError, match="actual backing"):
        f.check(f.output.clone())
    f.inputs[-1][0] += 1
    with pytest.raises(AssertionError, match="pointed-to"):
        f.check(f.output)


@pytest.mark.parametrize("stale", [False, True])
def test_changed_input_replay_checks_actual_buffers_and_restores_inputs(stale):
    f = harness.Fixture(control(), device="cpu")
    f.output.copy_(f.expected)
    original = tuple(x.clone() for x in f.inputs)
    class Replay:
        bound = True
        outputs = f.output
        calls = 0
        def rerun(self):
            self.calls += 1
            f.prepare()
            if not stale:
                self.outputs.copy_(f.expected)
            return self.outputs
    replay = Replay()
    if stale:
        with pytest.raises(AssertionError, match="Exact table"):
            f.verify_replay(replay)
    else:
        assert f.verify_replay(replay)["replay_input_draws"] == 3
        assert replay.calls == 3
    for value, pristine in zip(f.inputs, original):
        assert torch.equal(value.view(torch.uint8), pristine.view(torch.uint8))


def test_reference_rejects_invalid_lengths_and_duplicate_rows():
    f = harness.Fixture(control(), device="cpu")
    rows, _, pre, seq, extend, cache, *prefixes = f.inputs
    for bad_rows, bad_extend in ((rows, extend + 1), (rows * 0, extend)):
        with pytest.raises(ValueError):
            harness.reference(torch.zeros(f.shape, dtype=torch.int32), bad_rows,
                              pre, seq, bad_extend, cache, prefixes, f.width)


def test_standalone_validation_and_shared_task_spec():
    from src.task_spec import load_task_spec
    task = load_task_spec(TASK / "config.yaml", task_id="triton2triton/sglang/write_req_to_token_pool")
    assert task.candidate.initial_state == "implemented"
    process = subprocess.run([sys.executable, str(TASK / "scripts/evaluate.py"), "validate-task"],
                             capture_output=True, text=True, check=True)
    result = json.loads(process.stdout.split("ARENA_EVAL_RESULT=")[-1])
    assert result["status"] == "PASS"
    assert len(result["cases"]) == 7
    assert all(r["status"] == "PASS" for r in result["cases"])
    assert result["metadata"]["oracle_device"] == "cpu"
