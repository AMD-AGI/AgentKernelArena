"""Offline archive CLI coverage with tiny CPU tensors and explicit paths."""
import hashlib
import json

import pytest
import torch

from src.tools.extract_head_kernel_contract import main


def test_local_archive_cli_preserves_structure_without_numeric_payloads(tmp_path):
    task = tmp_path / "task"
    (task / "ut").mkdir(parents=True)
    archive = tmp_path / "reference_io.pt"
    blob = {"records": [{"sig": "case", "regime": "decode", "args": [], "kwargs": {
        "q": torch.tensor([[1.0, 2.0]]),
        "req_to_token": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "slot_ids": torch.tensor([1], dtype=torch.int64),
    }, "output": torch.tensor([[3.0, 4.0]])}]}
    torch.save(blob, archive)
    original = archive.read_bytes()
    meta = {"num_cases": 1, "cases": [{"sig": "case", "regime": "decode", "max_slots": 8}],
            "archival_capture": {"reference_io_sha256": hashlib.sha256(original).hexdigest()}}
    (task / "ut/meta.json").write_text(json.dumps(meta))
    output = tmp_path / "extracted.json"
    args = ["--family", "minimax", "--task", str(task), "--archive", str(archive), "--output", str(output)]
    assert main(args) == 0
    record = json.loads(output.read_text())["records"][0]
    assert record["kwargs"]["q"]["shape"] == [1, 2]
    assert "values" not in record["kwargs"]["q"]
    assert record["kwargs"]["req_to_token"]["rows"] == [[3, 4]]
    assert record["output_contract"]["recipe"] == "runtime_reference"
    assert archive.read_bytes() == original
    with pytest.raises(FileExistsError):
        main(args)


def test_bad_declared_hash_fails_before_archive_deserialization(tmp_path, monkeypatch):
    task = tmp_path / "task"
    (task / "ut").mkdir(parents=True)
    (task / "ut/meta.json").write_text(json.dumps({"reference_io_sha256": "0" * 64}))
    archive = tmp_path / "input.pt"
    archive.write_bytes(b"not a tensor archive")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unverified archive was loaded"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main(["--family", "qwen", "--task", str(task), "--archive", str(archive),
              "--output", str(tmp_path / "output.json")])


def test_kimi_cli_uses_task_kind_and_preserves_routing_zero_mask(tmp_path):
    task = tmp_path / "moe_gemm2_stage2"
    (task / "ut").mkdir(parents=True)
    spec = {"sig": "small-case", "m": 1}
    blob = {"cases": [{"spec": spec, "routing": {
        "sorted_token_ids": torch.tensor([0, 1], dtype=torch.int32),
        "sorted_expert_ids": torch.tensor([0], dtype=torch.int32),
        "num_valid_ids": torch.tensor([2], dtype=torch.int32),
        "sorted_weights": torch.tensor([0.25, 0.0]),
    }, "ref": torch.tensor([[3.0, 4.0]])}]}
    archive = tmp_path / "kimi-reference.pt"
    torch.save(blob, archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    meta = {"case_specs": [spec], "geometry": {"topk": 2},
            "archival_capture": {"reference_io_sha256": checksum}}
    (task / "ut/meta.json").write_text(json.dumps(meta))
    output = tmp_path / "kimi-contract.json"
    assert main(["--family", "kimi", "--task", str(task), "--archive", str(archive),
                 "--output", str(output)]) == 0
    contract = json.loads(output.read_text())
    assert contract["kind"] == "moe" and contract["case_count"] == 1
    assert contract["source_reference_sha256"] == checksum
    assert contract["contains_numeric_input_values"] is False
    assert contract["contains_reference_output_values"] is False
    routing = contract["records"][0]["routing"]
    assert routing["sorted_token_ids"]["recipe"] == "captured_integer"
    assert routing["sorted_weights"]["recipe"] == "generated_routing_weights"
    assert routing["sorted_weights"]["zero_mask"]["raw_bytes"] == 2
