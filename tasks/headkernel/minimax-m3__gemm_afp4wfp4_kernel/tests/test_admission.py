"""CPU-only negative admission controls; generated data here is not a workload."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from admission import case_from_fixture, oracle_policy
from fixture_codec import validate_fixture


@pytest.fixture
def fixture(tmp_path):
    bindings, groups = {}, {}
    for name, shape, stride, size in (("x", [2, 16], [16, 1], 1), ("w", [3, 16], [16, 1], 1),
                                     ("x_scales", [2, 1], [1, 1], 1), ("w_scales", [3, 1], [1, 1], 1),
                                     ("y", [2, 3], [3, 1], 2)):
        length = shape[0] * shape[1] * size
        data = bytes(length)
        path = tmp_path / (name + ".bin")
        path.write_bytes(data)
        bindings[name] = {"shape": shape, "stride": stride, "dtype": "torch.uint8" if size == 1 else "torch.bfloat16",
            "storage_offset": 0, "storage_nbytes": length, "element_size": size, "alias": name,
            "role": {"footprint": "full_storage"}}
        groups[name] = {"storage_nbytes": length, "segments": [{"blob": path.name,
            "offset_bytes": 0, "bytes": length, "sha256": hashlib.sha256(data).hexdigest()}]}
    arguments = {"M": 2, "N": 3, "K": 16, "stride_am": 16, "stride_ak": 1,
        "stride_bk": 1, "stride_bn": 16, "stride_ck": 0, "stride_cm": 3, "stride_cn": 1,
        "stride_asm": 1, "stride_ask": 1, "stride_bsn": 1, "stride_bsk": 1,
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 32}
    return {"schema": "served-tensor-fixture-v1", "startup_values": False, "family": "minimax_fp4_gemm",
        "origin": "served_eager", "case_key": "unit-test-only", "served": {"stage": "prefill"},
        "inputs": bindings, "outputs": {"result": bindings["y"]},
        "payload": {"inputs": groups, "outputs": {"y": groups["y"]}},
        "controls": {"dtype": "torch.bfloat16", "config_requested": None, "skip_reduce": False,
            "use_splitk_bf16": False, "packing": "e2m1_low_nibble_first", "scale_format": "e8m0_group32_unshuffled",
            "launches": [{"kernel": "_gemm_afp4wfp4_kernel", "arguments": arguments, "grid": [1]}]}}


def test_full_storage_and_counts_are_preserved(tmp_path, fixture):
    validate_fixture(tmp_path, fixture)
    case = case_from_fixture(fixture, {"0": 3, "1": 5}, {"path": "fixture.json", "sha256": "a" * 64})
    assert case["occurrences"] == 8 and case["calls_per_sample"] == 1
    assert case["occurrences_by_rank"] == {"0": 3, "1": 5}
    assert case["scalars"]["native_arguments"]["K"] == 16
    assert case["tensors"]["w"]["strides"] == [16, 1]


@pytest.mark.parametrize("attack", ["startup", "synthetic", "gap", "changed_blob", "symlink", "extent"])
def test_fixture_admission_rejects_incomplete_or_diagnostic_payload(tmp_path, fixture, attack):
    if attack == "startup":
        fixture["startup_values"] = True
    elif attack == "synthetic":
        fixture["provenance"] = {"synthetic": True}
    elif attack == "gap":
        fixture["payload"]["inputs"]["x"]["segments"][0]["offset_bytes"] = 1
    elif attack == "changed_blob":
        (tmp_path / "x.bin").write_bytes(b"changed")
    elif attack == "symlink":
        (tmp_path / "x.bin").rename(tmp_path / "real.bin")
        (tmp_path / "x.bin").symlink_to("real.bin")
    else:
        fixture["inputs"]["x"]["storage_offset"] = 1
    with pytest.raises(ValueError):
        validate_fixture(tmp_path, fixture)


@pytest.mark.parametrize("attack", ["logical_k", "stride", "second_launch", "zero_count", "alias"])
def test_case_contract_rejects_changed_native_work(fixture, attack):
    counts = {"0": 1}
    if attack == "logical_k":
        fixture["controls"]["launches"][0]["arguments"]["K"] = 32
    elif attack == "stride":
        fixture["controls"]["launches"][0]["arguments"]["stride_bk"] = 16
    elif attack == "second_launch":
        fixture["controls"]["launches"] *= 2
    elif attack == "zero_count":
        counts = {"0": 0}
    else:
        fixture["inputs"]["y"]["alias"] = fixture["inputs"]["x"]["alias"]
    with pytest.raises(ValueError):
        case_from_fixture(fixture, counts, {})


@pytest.mark.parametrize("value", [0, -1, 0.03, float("nan")])
def test_oracle_policy_cannot_be_missing_or_weakened(value):
    with pytest.raises(ValueError):
        oracle_policy({"metric": "mixed_rms", "tolerance": value, "basis": "unit-test-only"})


def test_runner_rejects_absent_capture_before_importing_gpu_dependencies(tmp_path):
    spec = importlib.util.spec_from_file_location("fp4_task_runner", ROOT / "scripts/task_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with pytest.raises(ValueError, match="missing"):
        runner.run("compile", tmp_path)
