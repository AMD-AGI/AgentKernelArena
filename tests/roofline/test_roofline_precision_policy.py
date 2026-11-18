"""The arithmetic float32 matrix products are charged at (CPU only, hand-computed numbers)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.roofline import __main__ as cli
from src.roofline import precision_policy
from src.roofline.arch import load_peaks
from src.roofline.floor import case_floor
from src.roofline.interface import performance_cases
from src.roofline.precision_policy import (ARITHMETIC, charge_fp32_products, charge_key, fp32_arithmetic, load_probe,
                                           package_sha256, probed_arithmetic)
from src.task_contract import TASK_FILE, load_task_contract

ROOT = Path(__file__).resolve().parents[2]
PEAKS = load_peaks("mi355x_datasheet")
GEMM = 2 * 4096 ** 3  # 137438953472 FLOPs
GEMM_F32 = "triton/gemm/gemm_f32/nn_n4096_k4096"


def test_emulated_keys_divide_the_16_bit_peak_by_the_product_count():
    assert PEAKS.matrix_flops_per_s("matrix_bf16x3") == pytest.approx(2516.6e12 / 3)
    assert PEAKS.matrix_flops_per_s("matrix_fp16x3") == pytest.approx(838.8666e12)
    assert PEAKS.matrix_flops_per_s("matrix_bf16x9") == pytest.approx(279.6222e12)
    assert PEAKS.matrix_flops_per_s("matrix_fp32") == pytest.approx(157.3e12)
    assert [charge_key(a) for a in ARITHMETIC] == ["matrix_bf16", "matrix_bf16x3", "matrix_fp16x3",
                                                   "matrix_bf16x6", "matrix_bf16x9", "matrix_fp32"]


def test_without_a_probe_float32_products_are_charged_at_one_bf16_product():
    # Nothing has ruled out the fastest arithmetic for an unprobed case, so the floor stays a lower bound.
    charged, notes = charge_fp32_products("synthetic/none", "case", {"matrix_fp32": GEMM, "matrix_fp8": 10})
    assert charged == {"matrix_bf16": GEMM, "matrix_fp8": 10}
    assert notes == ["fp32_arithmetic:bf16 (no probe result; fastest arithmetic, a loose lower bound)"]


def test_products_without_float32_operands_are_untouched():
    flops = {"matrix_bf16": GEMM}
    assert charge_fp32_products("synthetic/none", "case", flops) == (flops, [])


def _probe_file(path, tasks):
    path.write_text(json.dumps({"source": "synthetic probe", "tasks": tasks}))
    return load_probe(path)


@pytest.fixture
def probed(tmp_path):
    """A one-file package probed with bf16x6, except case "big" (fp32) and "cheap" (bf16), and its probe."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text("schema_version: 3\n")
    probe = _probe_file(tmp_path / "probe.json", {"synthetic/t": {
        "package_sha256": package_sha256(task_dir), "arithmetic": "bf16x6", "cases": {"big": "fp32", "cheap": "bf16"}}})
    return task_dir, probe


def test_a_task_the_probe_did_not_cover_keeps_the_unprobed_charge(probed):
    charged, notes = charge_fp32_products("synthetic/none", "case", {"matrix_fp32": GEMM}, probe=probed[1])
    assert charged == {"matrix_bf16": GEMM}
    assert notes == ["fp32_arithmetic:bf16 (no probe result; fastest arithmetic, a loose lower bound)"]


def test_bf16_flops_merge_with_charged_float32_flops(probed):
    charged, _ = charge_fp32_products("synthetic/t", "cheap", {"matrix_fp32": 6, "matrix_bf16": 4}, *probed)
    assert charged == {"matrix_bf16": 10}


def test_every_probed_case_is_charged_at_the_arithmetic_the_probe_saw_pass(probed):
    task_dir, probe = probed
    assert probed_arithmetic(probe, "synthetic/t", "small") == "bf16x6"
    for case_id, key in (("small", "matrix_bf16x6"), ("big", "matrix_fp32"), ("cheap", "matrix_bf16")):
        charged, notes = charge_fp32_products("synthetic/t", case_id, {"matrix_fp32": GEMM}, task_dir, probe)
        assert charged == {key: GEMM}
        assert notes[0].startswith(f"fp32_arithmetic:{probed_arithmetic(probe, 'synthetic/t', case_id)} (cheapest")
        assert notes[0].endswith("; synthetic probe)")


def test_a_changed_package_falls_back_to_the_unprobed_charge(probed):
    task_dir, probe = probed
    (task_dir / "reference.py").write_text("# edited\n")
    package_sha256.cache_clear()
    arithmetic, note = fp32_arithmetic("synthetic/t", "small", task_dir, probe)
    assert arithmetic == "bf16"
    assert note.startswith("fp32_arithmetic:bf16 (task package changed since the probe, which saw bf16x6 pass;")


def test_a_missing_package_falls_back_to_the_unprobed_charge(probed):
    task_dir, probe = probed
    shutil.rmtree(task_dir)
    assert fp32_arithmetic("synthetic/t", "small", task_dir, probe)[0] == "bf16"


def test_a_probe_file_with_an_unknown_arithmetic_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown arithmetic 'bf16x2'"):
        _probe_file(tmp_path / "probe.json", {"synthetic/t": {"package_sha256": "0" * 64, "arithmetic": "bf16x2"}})


def _gemm_f32_floor(case_id, probe=None):
    task = load_task_contract(ROOT / "tasks" / GEMM_F32 / TASK_FILE, task_id=GEMM_F32)
    case = next(c for c in performance_cases(task) if c["id"] == case_id)
    return case_floor(task, case, PEAKS, probe=probe)


def test_gemm_f32_floor_is_charged_at_one_bf16_product_without_a_probe():
    floor = _gemm_f32_floor("m4096")
    assert floor["flops_matrix_by_precision"] == {"matrix_bf16": GEMM}
    assert floor["t_compute_matrix_us"] == pytest.approx(54.61295139156005)  # 2 * 4096^3 / 2516.6 TFLOPS
    assert "precision_substitution_sensitive" in floor["notes"]


@pytest.mark.parametrize("case_id, flops, arithmetic, products",
                         [("m4096", GEMM, "bf16x3", 3), ("m8192", 2 * GEMM, "fp16x3", 3)])
def test_gemm_f32_floor_is_charged_at_the_probed_arithmetic(tmp_path, case_id, flops, arithmetic, products):
    probe = _probe_file(tmp_path / "probe.json", {GEMM_F32: {
        "package_sha256": package_sha256(ROOT / "tasks" / GEMM_F32), "arithmetic": "bf16x3",
        "cases": {"m8192": "fp16x3"}}})
    floor = _gemm_f32_floor(case_id, probe)
    key = charge_key(arithmetic)
    assert floor["flops_matrix"] == flops  # the necessary FLOPs do not change, only their rate
    assert floor["flops_matrix_by_precision"] == {key: flops}
    assert floor["precision_key"] == key
    # 2 * 4096^3 / 2516.6 TFLOPS = 54.613 us per 16-bit product, against 873.738 us at the fp32 peak.
    assert floor["t_compute_matrix_us"] == pytest.approx(54.61295139156005 * products * flops / GEMM)
    assert "precision_substitution_sensitive" in floor["notes"]
    assert any(note.startswith(f"fp32_arithmetic:{arithmetic} (") for note in floor["notes"])


def test_floors_command_reads_the_probe_file(tmp_path):
    probe = tmp_path / "probe.json"
    _probe_file(probe, {GEMM_F32: {"package_sha256": package_sha256(ROOT / "tasks" / GEMM_F32),
                                   "arithmetic": "bf16x3"}})
    keys = {}
    for name, extra in (("plain", []), ("probed", ["--precision-probe", str(probe)])):
        assert cli.main(["floors", "--out", str(tmp_path / name), "--tasks", GEMM_F32, *extra]) == 0
        rows = [json.loads(line) for line in (tmp_path / name / "floors.jsonl").read_text().splitlines()]
        keys[name] = {row["precision_key"] for row in rows}
    assert keys == {"plain": {"matrix_bf16"}, "probed": {"matrix_bf16x3"}}


def test_the_unprobed_default_is_the_fastest_arithmetic():
    rates = {a: PEAKS.matrix_flops_per_s(charge_key(a)) for a in ARITHMETIC}
    assert max(rates, key=rates.get) == "bf16" == precision_policy.UNPROBED
