"""CPU checks for the combine task's correctness-only decode controls.

These stand-ins exercise the task-owned correctness action; they do not qualify
the Triton kernel or its device timing.
"""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TASK = (Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm"
        / "triton_combine_sampled_and_draft_tokens")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    old_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


@pytest.fixture
def task_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(TASK))
    contract = load(TASK / "_arena_contract.py", "_combine_contract_test")
    replay = load(TASK / "_arena_replay.py", "_combine_replay_test")
    monkeypatch.setitem(sys.modules, "_arena_replay", replay)
    harness = load(TASK / "scripts/task_runner.py", "_combine_harness_test")
    adapter = load(TASK / "_arena_eval.py", "_combine_adapter_test")
    return contract, replay, harness, adapter


def test_known_decode_answer_and_prefill_control(task_modules):
    contract, _, harness, _ = task_modules
    prefill, decode = list(contract.control_inputs(harness))

    prefill_ids, prefill_indices = contract.reference(harness, prefill)
    assert prefill_ids.tolist() == [-9, -9, 10, 1, 12, 7, 8]
    assert prefill_indices.tolist() == [0, 2, 3, 4, 5, 6]

    assert decode[1].tolist() == [2, 0, 1]
    assert decode[7].diff().tolist() == [1, 3, 2]
    decode_ids, decode_indices = contract.reference(harness, decode)
    assert decode_ids.tolist() == [303, 101, 111, 112, 202, 211]
    assert decode_indices.tolist() == [0, 1, 2, 3, 4, 5]

    padded, storage = contract.padded_draft_inputs(harness, "cpu")
    assert padded[6].stride() == (5, 1)
    assert storage[:, 2:].tolist() == [[-777] * 3] * 3
    padded_ids, padded_indices = contract.reference(harness, padded)
    assert torch.equal(padded_ids, decode_ids)
    assert torch.equal(padded_indices, decode_indices)


@pytest.mark.parametrize("fault", [None, "omit_zero_draft_sampled", "use_batch_row",
                                  "contiguous_draft_stride"])
def test_native_correctness_action_rejects_decode_faults(task_modules, monkeypatch, fault):
    contract, replay, harness, adapter = task_modules
    if fault in ("omit_zero_draft_sampled", "use_batch_row"):
        # The new remapped decode case must reject each fault on its own.
        decode = tuple(contract.control_inputs(harness))[1]
        monkeypatch.setattr(contract, "control_inputs", lambda harness: iter((decode,)))
    observed_strides = []

    def combine_sampled_and_draft_tokens(
        input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
        seq_lens, prefill_len, draft_tokens, cu_num_logits, num_logits,
    ):
        observed_strides.append(draft_tokens.stride(0))
        original = input_ids.clone()
        expected_ids, indices = harness.reference_combine(
            input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
            seq_lens, prefill_len, draft_tokens, cu_num_logits,
        )
        assert num_logits == indices.numel()
        input_ids.copy_(expected_ids)

        if fault == "omit_zero_draft_sampled":
            for batch in range(seq_lens.numel()):
                state = idx_mapping[batch].item()
                if (cu_num_logits[batch + 1] - cu_num_logits[batch] == 1
                        and seq_lens[batch] > prefill_len[state]):
                    position = query_start_loc[batch + 1].item() - 1
                    input_ids[position] = original[position]
        elif fault == "use_batch_row":
            for batch in range(seq_lens.numel()):
                state = idx_mapping[batch].item()
                if seq_lens[batch] <= prefill_len[state]:
                    continue
                count = (cu_num_logits[batch + 1] - cu_num_logits[batch]).item()
                start = query_start_loc[batch + 1].item() - count
                input_ids[start] = last_sampled_tokens[batch]
                if count > 1:
                    input_ids[start + 1:start + count] = draft_tokens[batch, :count - 1]
        elif fault == "contiguous_draft_stride":
            wrong_drafts = torch.as_strided(
                draft_tokens, draft_tokens.shape, (draft_tokens.shape[1], 1))
            wrong_ids, _ = harness.reference_combine(
                original, idx_mapping, last_sampled_tokens, query_start_loc,
                seq_lens, prefill_len, wrong_drafts, cu_num_logits,
            )
            input_ids.copy_(wrong_ids)
        return indices

    # Route the real task adapter through the normal installed correctness
    # checker. The five scored cases are left to their separate GPU runner.
    harness.load_module = lambda: SimpleNamespace(
        combine_sampled_and_draft_tokens=combine_sampled_and_draft_tokens)
    harness.run_correctness = lambda *, case_index=None: (True, None)
    native_controls = contract.controls
    monkeypatch.setattr(contract, "controls",
                        lambda harness, function, device: native_controls(harness, function, "cpu"))
    replay.install(harness, contract)
    monkeypatch.setattr(adapter, "load_harness", lambda: harness)

    result = adapter.evaluate("candidate", "correctness")
    rows = {row["test_case_id"]: row for row in result["cases"]}
    assert all(rows[f"perf{i}"]["status"] == "PASS" for i in range(1, 6))
    if fault is None:
        assert result["status"] == rows["contract_controls"]["status"] == "PASS"
    else:
        assert result["status"] == rows["contract_controls"]["status"] == "FAIL"
        assert "exact integer reference" in rows["contract_controls"]["reason"]
        if fault == "contiguous_draft_stride":
            assert observed_strides == [3, 2, 5]


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="Triton and a compatible GPU are required",
)
def test_real_kernel_passes_native_correctness_action(task_modules, monkeypatch):
    contract, replay, _, adapter = task_modules
    monkeypatch.setitem(sys.modules, "_arena_contract", contract)
    original_padded = contract.padded_draft_inputs
    observed = []

    def record_padded(harness, device):
        args, storage = original_padded(harness, device)
        observed.append((args[6].stride(), storage.device.type))
        return args, storage

    monkeypatch.setattr(contract, "padded_draft_inputs", record_padded)

    result = adapter.evaluate("candidate", "correctness")
    assert result["status"] == "PASS", result
    assert [row["test_case_id"] for row in result["cases"]] == [
        "perf1", "perf2", "perf3", "perf4", "perf5", "contract_controls",
    ]
    assert all(row["status"] == "PASS" for row in result["cases"])
    assert observed == [((5, 1), "cuda")]
