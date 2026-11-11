"""Decoded comparison of packed outputs: ``compare.<name>.decode`` in the contract and the evaluator."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.task_contract import load_task_contract, validate_task_contract
from src.task_spec import TaskConfigError
from src.tools.arena_eval import values
from src.tools.arena_eval.compare import compare_tensor
from src.tools.arena_eval.contract import Contract, ContractError
from src.tools.arena_eval.evaluate import _check

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "tasks/triton/normalization/rms_norm/bf16_h8192"
PILOT_ID = "triton/normalization/rms_norm/bf16_h8192"
DIM = 32
FP8_MAX = 448.0
MODULE = "_aka_task_decode_cache"
DECODE_SOURCE = f'''
import torch


def decode_rows(packed):
    """[rows, {DIM} E4M3 codes | f32 scale] -> float64 values [rows, {DIM}]."""
    codes = packed[:, :{DIM}].contiguous().view(torch.float8_e4m3fn).float().double()
    return codes * packed[:, {DIM}:].contiguous().view(torch.float32).double()


def not_a_tensor(packed):
    return [1.0]
'''
HOOK = {"file": "decode.py", "symbol": "decode_rows", "precision": "float8_e4m3fn"}


def pack(v: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Each row's E4M3 codes of v / scale, then the float32 scale's 4 bytes."""
    codes = (v / scale.double()[:, None]).clamp(-FP8_MAX, FP8_MAX).float().to(torch.float8_e4m3fn)
    return torch.cat([codes.view(torch.uint8), scale.float().contiguous().view(-1, 1).view(torch.uint8)], 1)


def row_scale(v: torch.Tensor) -> torch.Tensor:
    return (v.abs().amax(-1) / FP8_MAX).float()


@pytest.fixture
def hook(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    (workspace / "decode.py").write_text(DECODE_SOURCE)
    monkeypatch.setattr(values, "WORKSPACE", workspace)
    sys.modules.pop(MODULE, None)
    yield
    sys.modules.pop(MODULE, None)


def _contract(**declaration):
    compare = {"bound": "baseline_relative", "k": 2.0, "decode": dict(HOOK), **declaration}
    return SimpleNamespace(checked_tensors=("cache",), element_order="fixed", compare={"cache": compare})


@pytest.fixture
def caches():
    """The reference's packed rows, a baseline that rounds keys differently, and a 1-ulp scale change."""
    g = torch.Generator().manual_seed(7)
    v = torch.randn(16, DIM, generator=g, dtype=torch.float64)
    reference = pack(v, row_scale(v))
    noisy = v * (1 + 2.0 ** -8 * torch.randn(16, DIM, generator=g, dtype=torch.float64))
    baseline = pack(noisy, row_scale(noisy))
    scale = row_scale(v)
    candidate = pack(v, torch.nextafter(scale, torch.full_like(scale, torch.inf)))
    return reference, baseline, candidate


def test_a_one_ulp_scale_change_fails_bytes_but_passes_decoded(hook, caches):
    reference, baseline, candidate = caches
    assert bool((candidate != reference).any(-1).all())  # every row's scale bytes changed
    assert not compare_tensor({"bound": "exact"}, candidate, reference, dtype_name="uint8").passed
    ok, metrics = _check(_contract(), "candidate", {"cache": candidate}, {"cache": reference}, {"cache": baseline})
    assert ok and metrics["cache"]["bound"] == "baseline_relative"
    ok, metrics = _check(_contract(), "baseline", {"cache": baseline}, {"cache": reference}, None)
    assert ok and metrics["cache"]["bound"] == "baseline_sanity" and metrics["cache"]["sqnr_db"] > 25


def test_wrong_decoded_values_still_fail(hook, caches):
    reference, baseline, candidate = caches
    zeroed = candidate.clone()
    zeroed[3] = 0
    rescaled = candidate.clone()
    rescaled[5, DIM:] = (candidate[5, DIM:].view(torch.float32) * 2).view(torch.uint8)
    unwritten = candidate.clone()
    unwritten[9, :DIM] = 0x7F  # NaN codes
    for wrong in (zeroed, rescaled, unwritten):
        ok, _ = _check(_contract(), "candidate", {"cache": wrong}, {"cache": reference}, {"cache": baseline})
        assert not ok


def test_precision_sets_one_code_step_as_the_rounding_unit(hook, caches):
    reference, _, _ = caches
    reference = reference.clone()
    reference[0, 0] = 0x50  # E4M3 8.0, the bottom of its binade
    baseline = reference.clone()  # exact: no baseline error to lean on
    candidate = reference.clone()
    candidate[0, 0] = 0x51  # 9.0: one code step up, as a key that rounds the other way
    expected, written, base = {"cache": reference}, {"cache": candidate}, {"cache": baseline}
    assert _check(_contract(), "candidate", written, expected, base)[0]
    without = {"bound": "baseline_relative", "k": 2.0, "decode": {"file": "decode.py", "symbol": "decode_rows"}}
    contract = SimpleNamespace(checked_tensors=("cache",), element_order="fixed", compare={"cache": without})
    assert not _check(contract, "candidate", written, expected, base)[0]
    two_steps = reference.clone()
    two_steps[0, 0] = 0x52
    assert not _check(_contract(), "candidate", {"cache": two_steps}, expected, base)[0]


def test_the_decode_hook_must_return_floating_values_from_a_packed_reference(hook, caches):
    reference, baseline, candidate = caches
    with pytest.raises(ContractError, match="float32 or float64"):
        _check(_contract(decode=dict(HOOK, symbol="not_a_tensor")), "candidate", {"cache": candidate},
               {"cache": reference}, {"cache": baseline})
    with pytest.raises(ContractError, match="packed as uint8"):
        _check(_contract(), "candidate", {"cache": candidate}, {"cache": reference.double()}, {"cache": baseline})


def test_the_calibration_probe_compares_decoded_outputs(hook, monkeypatch):
    from src.tools.arena_eval import controls

    def reference(x):
        y = x.to(torch.float64) * 1.1  # float32 under the probe's patch: some scales round differently
        return {"cache": pack(y, (y.abs().amax(-1) / FP8_MAX).float())}

    monkeypatch.setattr(controls.values, "reference_function", lambda contract: reference)
    x = torch.randn(16, DIM, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 3.3
    expected = reference(x)
    data = SimpleNamespace(inputs=[SimpleNamespace(logical={"x": x})], references=[expected],
                           baselines=[{"cache": expected["cache"].clone()}])
    results = controls._calibration_probe(_contract(), SimpleNamespace(case_id="c0"), data)
    assert [(item["output"], item["bound"], item["passed"]) for item in results] == \
        [("cache", "baseline_relative", True)]


# ------------------------------------------------------------------------------------------- validation
def _packed_raw():
    raw = yaml.safe_load((PILOT / "task.yaml").read_text())
    raw["interface"]["outputs"]["out"]["dtype"] = "uint8"
    raw["compare"]["out"] = {"bound": "baseline_relative", "decode": dict(HOOK)}
    return raw, json.loads((PILOT / "workloads.json").read_text())


def test_a_decode_hook_is_declared_with_a_numeric_bound():
    raw, workloads = _packed_raw()
    contract = validate_task_contract(raw, workloads, task_id=PILOT_ID)
    assert contract.resolved_contract("triton")["compare"]["out"]["decode"] == HOOK


@pytest.mark.parametrize("declaration, message", [
    ({"bound": "exact", "decode": HOOK}, "needs a numeric bound"),
    ({"bound": "baseline_relative", "decode": dict(HOOK, file="../decode.py")}, "normalized task-relative path"),
    ({"bound": "baseline_relative", "decode": dict(HOOK, file="baseline/decode.py")}, "at the task root"),
    ({"bound": "baseline_relative", "decode": dict(HOOK, file="decode.txt")}, "at the task root"),
    ({"bound": "baseline_relative", "decode": dict(HOOK, symbol="_private")}, "public Python identifier"),
    ({"bound": "baseline_relative", "decode": dict(HOOK, precision="uint8")}, "precision must be one of"),
    ({"bound": "baseline_relative", "decode": dict(HOOK, dtype="float32")}, "unknown fields"),
    ({"bound": "baseline_relative", "decode": {"file": "decode.py"}}, "missing fields"),
    ({"bound": "baseline_relative", "decoder": HOOK}, "unknown fields"),
    ({"bound": "baseline_relative"}, "compare exactly unless they declare decode"),
])
def test_invalid_decode_declarations_fail(declaration, message):
    raw, workloads = _packed_raw()
    raw["compare"]["out"] = copy.deepcopy(declaration)
    with pytest.raises(TaskConfigError, match=message):
        validate_task_contract(raw, workloads, task_id=PILOT_ID)


def test_floating_outputs_compare_their_values_directly():
    raw, workloads = _packed_raw()
    raw["interface"]["outputs"]["out"]["dtype"] = "bfloat16"
    with pytest.raises(TaskConfigError, match="integral or packed outputs"):
        validate_task_contract(raw, workloads, task_id=PILOT_ID)


def _package(tmp_path, source: str | None) -> Path:
    package = tmp_path / "task"
    shutil.copytree(PILOT, package, ignore=shutil.ignore_patterns("__pycache__"))
    raw, _ = _packed_raw()
    (package / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    if source is not None:
        (package / "decode.py").write_text(source)
    return package


def test_the_decode_file_must_define_the_function(tmp_path):
    package = _package(tmp_path, DECODE_SOURCE)
    contract = load_task_contract(package / "task.yaml", task_id=PILOT_ID)
    resolved = contract.resolved_contract("triton")
    assert Contract(copy.deepcopy(resolved)).compare["out"]["decode"]["symbol"] == "decode_rows"
    resolved["compare"]["out"]["bound"] = "exact"
    with pytest.raises(ContractError, match="numeric bound"):
        Contract(resolved)
    (package / "decode.py").write_text("import torch\n\ndecode_rows = None\n")
    with pytest.raises(TaskConfigError, match="does not define function decode_rows"):
        load_task_contract(package / "task.yaml", task_id=PILOT_ID)


def test_the_decode_file_must_be_in_the_package(tmp_path):
    package = _package(tmp_path, None)
    with pytest.raises(TaskConfigError, match="is not a file in the task package"):
        load_task_contract(package / "task.yaml", task_id=PILOT_ID)
    outside = tmp_path / "outside.py"
    outside.write_text(DECODE_SOURCE)
    (package / "decode.py").symlink_to(outside)
    with pytest.raises(TaskConfigError, match="is not a file in the task package"):
        load_task_contract(package / "task.yaml", task_id=PILOT_ID)
