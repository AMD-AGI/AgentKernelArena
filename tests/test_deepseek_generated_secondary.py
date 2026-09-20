"""Numerical gates for consumed scales and LSE, using CPU tensor fixtures."""

import base64
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib

import pytest
from head_kernel_generated_test_utils import generated_helper, generated_task
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper = load("_secondary_contract", generated_helper("deepseek", "generated_contract.py"))


def test_consumed_scale_corruption_is_rejected_and_padding_is_ignored():
    kw = {
        "a": torch.zeros((2, 64), dtype=torch.float8_e4m3fn),
        "w1": torch.empty((3, 128, 32), dtype=torch.float4_e2m1fn_x2),
        "topk": 2,
        "tile_m": 32,
        "sorted_expert_ids": torch.tensor([1], dtype=torch.int32),
        "num_valid_ids": torch.tensor([32, 1], dtype=torch.int32),
        "sorted_token_ids": torch.tensor(
            [0, 1, 1 << 24, (1 << 24) | 1] + [2] * 28, dtype=torch.int32
        ),
    }
    args = {"args": [], "kwargs": kw}
    values = torch.ones((2, 2, 64), dtype=torch.float32).to(torch.float8_e4m3fn)
    raw = torch.full((256, 8), 255, dtype=torch.uint8)
    offsets = helper.scale_byte_offsets(torch.arange(4), torch.arange(2), 8)
    raw.reshape(-1)[offsets] = 127
    scales = raw.view(torch.float8_e8m0fnu)
    expected = helper.moe1_comparison((values, scales), args, torch)
    assert expected[1].shape == (4, 64)
    assert torch.equal(expected[1], torch.ones_like(expected[1]))
    baseline = helper.encode_output(expected, torch)
    corrupt = raw.clone()
    corrupt.reshape(-1)[offsets[1, 1]] = 128
    observed = helper.moe1_comparison(
        (values, corrupt.view(torch.float8_e8m0fnu)), args, torch
    )
    assert not helper.compare_output(
        helper.encode_output(observed, torch), baseline, 0.15, torch
    )
    padding = raw.clone()
    padding[-1, -1] = 0
    observed = helper.moe1_comparison(
        (values, padding.view(torch.float8_e8m0fnu)), args, torch
    )
    assert helper.compare_output(
        helper.encode_output(observed, torch), baseline, 0, torch
    )
    bad = raw.clone()
    bad.reshape(-1)[offsets[0, 0]] = 255
    with pytest.raises(RuntimeError, match="undefined E8M0"):
        helper.moe1_comparison((values, bad.view(torch.float8_e8m0fnu)), args, torch)


def test_lse_defined_values_and_signed_empty_context_sentinels_are_gated():
    mask = bytes([0, 1, 0, 0])
    record = {
        "undefined_rows": [],
        "undefined_partial_rows": [],
        "undefined_lse": {
            "shape": [2, 1, 2],
            "mask": {
                "codec": "zlib-base64",
                "bytes": len(mask),
                "data": base64.b64encode(zlib.compress(mask)).decode(),
            },
        },
    }
    args = {"q": torch.zeros((2, 1, 2, 512), dtype=torch.bfloat16), "head_dim_v": 512}
    values = torch.ones_like(args["q"])
    lse = torch.tensor([[[1.0, float("nan")]], [[float("inf"), 2.0]]])
    baseline = helper.encode_output(
        helper.dsa_comparison((values, lse), args, record, torch), torch
    )
    changed = lse.clone()
    changed[1, 0, 1] = 4
    observed = helper.encode_output(
        helper.dsa_comparison((values, changed), args, record, torch), torch
    )
    assert not helper.compare_output(observed, baseline, 0.02, torch)
    changed = lse.clone()
    changed[0, 0, 1] = 200
    observed = helper.encode_output(
        helper.dsa_comparison((values, changed), args, record, torch), torch
    )
    assert helper.compare_output(observed, baseline, 0, torch)
    changed = lse.clone()
    changed[1, 0, 0] = -float("inf")
    observed = helper.encode_output(
        helper.dsa_comparison((values, changed), args, record, torch), torch
    )
    assert not helper.compare_output(observed, baseline, 100, torch)


def test_performance_transform_receives_the_actual_complete_return(monkeypatch):
    bench = load("_secondary_bench", ROOT / "tasks/head_kernels/_support/_bench.py")
    full = (object(), object())
    args = {"x": object()}
    calls = []
    module = SimpleNamespace(
        comparison_output=lambda output, values: calls.append((output, values))
        or ("validated",)
    )
    checked = []
    monkeypatch.setitem(
        sys.modules,
        "runtime_integrity",
        SimpleNamespace(
            ACTIVE_GUARD=SimpleNamespace(
                check_module=lambda value: checked.append(value)
            )
        ),
    )
    transform = bench.output_transform(module, {}, {"args": args}, torch)
    assert transform(full) == ("validated",)
    assert calls == [(full, args)]
    assert checked == [module]


def test_actual_lse_masks_are_frozen_independently_of_primary_nan_masks():
    task = next(
        (ROOT / "tasks/head_kernels/deepseek-v4-pro").glob("*/*/dsa_sparse_mla_attn")
    )
    records = helper.load_contract(task / "ut")["records"]
    assert [row["undefined_lse"]["undefined_count"] for row in records] == [
        0,
        96357,
        0,
        101,
    ]
    for row in records:
        node = row["undefined_lse"]
        raw = helper.decode_bytes(node["mask"])
        assert sum(raw) == node["undefined_count"]
        assert len(raw) == node["shape"][0] * node["shape"][1] * node["shape"][2]


def test_compound_adapter_cannot_run_without_shared_attestation(monkeypatch):
    bench = load(
        "_untrusted_adapter_bench", ROOT / "tasks/head_kernels/_support/_bench.py"
    )
    monkeypatch.delitem(sys.modules, "runtime_integrity", raising=False)
    module = SimpleNamespace(comparison_output=lambda output, args: output)
    with pytest.raises(RuntimeError, match="trusted bootstrap"):
        bench.output_transform(module, {}, {"args": {}}, torch)
