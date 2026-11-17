"""Roofline floors over the real task packages: properties, tiers and the CLI guards (CPU only)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from src.roofline import __main__ as cli
from src.roofline.arch import load_peaks
from src.roofline.bytes_model import g1_bytes, generic_bytes
from src.roofline.floor import case_floor, task_floors
from src.roofline.interface import case_shapes, performance_cases
from src.task_contract import TASK_FILE, load_task_contract

ROOT = Path(__file__).resolve().parents[2]
TASK_IDS = cli.discover_tasks()
PEAKS = load_peaks("mi355x_datasheet")


def contract(task_id):
    return load_task_contract(ROOT / "tasks" / task_id / TASK_FILE, task_id=task_id)


def test_discovery_finds_the_v3_kernel_tasks_only():
    assert len(TASK_IDS) >= 379
    assert not any(task_id.startswith("e2e/") for task_id in TASK_IDS)


def test_every_task_gets_a_floor_and_g2_never_exceeds_g1():
    for task_id in TASK_IDS:
        task = contract(task_id)
        floors = task_floors(task, PEAKS)
        assert floors, task_id
        for case, floor in zip(performance_cases(task), floors):
            assert floor["tier"] in ("verified", "estimated", "none")
            assert floor["t_floor_us"] > 0 or floor["tier"] == "none", task_id
            shapes = case_shapes(task, case)
            assert generic_bytes(shapes).bytes <= g1_bytes(shapes), task_id


def test_floor_is_the_maximum_of_its_parts_and_monotonic_in_peaks():
    task = contract("triton/gemm/gemm_a16w16/n1280_k8192")
    case = performance_cases(task)[-1]
    floor = case_floor(task, case, PEAKS)
    parts = [floor["t_compute_matrix_us"], floor["t_compute_vector_us"], floor["t_memory_us"]]
    assert floor["t_floor_us"] == max(parts)
    faster = dataclasses.replace(PEAKS, mem_bw_gbps=PEAKS.mem_bw_gbps * 2,
                                 tflops={k: v * 2 for k, v in PEAKS.tflops.items()})
    assert case_floor(task, case, faster)["t_floor_us"] == pytest.approx(floor["t_floor_us"] / 2)


def test_inference_layer_norm_moves_x_weight_bias_and_output_once():
    # Elementwise weight and bias as large as x: four tensors of 1 * 64 * 256 * 256 float32 values.
    task = contract("triton/normalization/layer_norm_wide/f32_c64_h256_w256")
    case = next(c for c in performance_cases(task) if c["id"] == "b1")
    floor = case_floor(task, case, PEAKS)
    assert floor["bytes_min"] == 4 * (64 * 256 * 256 * 4)
    assert floor["flops_matrix"] == 0 and floor["bound"] == "memory"


def test_two_product_kernel_floor_is_the_larger_of_total_compute_and_total_memory():
    # One kernel, an fp8 and a bf16 product: the compute times add, the bytes add, and the floor is the
    # larger of the two totals, not the sum of each product's own maximum.
    task = contract("triton/gemm/fused_gemm_a8w8_blockscale_a16w16/fp8_n512_bf16_n256_k7168")
    case = next(c for c in performance_cases(task) if c["id"] == "m512")
    floor = case_floor(task, case, PEAKS)
    by_precision = floor["flops_matrix_by_precision"]
    assert by_precision == {"matrix_fp8": 2 * 512 * 512 * 7168, "matrix_bf16": 2 * 512 * 256 * 7168}
    compute = sum(flops / PEAKS.matrix_flops_per_s(key) for key, flops in by_precision.items()) * 1e6
    memory = floor["bytes_min"] / (PEAKS.mem_bw_gbps * 1e9) * 1e6
    assert floor["t_compute_matrix_us"] == pytest.approx(compute)
    assert floor["t_floor_us"] == pytest.approx(max(compute, memory))


def test_operators_without_a_work_model_get_tier_none():
    for task_id in ("hip/other/three_nn", "triton/sorting/sort_rows"):
        matches = [t for t in TASK_IDS if t.startswith(task_id)]
        for match in matches:
            assert {f["tier"] for f in task_floors(contract(match), PEAKS)} == {"none"}


def test_memory_bound_operator_floor_matches_the_worked_example():
    task = contract("triton/activation/silu_and_mul/bf16_d4096")
    case = next(c for c in performance_cases(task) if c["axes"]["n_tokens"] == 4096)
    floor = case_floor(task, case, PEAKS)
    assert (floor["bytes_rule"], floor["bound"], floor["tier"]) == ("G1", "memory", "estimated")
    assert floor["t_floor_us"] == pytest.approx(100663296 / 8192e3)  # bytes / (8192 GB/s), in us


def test_output_inside_the_repository_is_refused():
    with pytest.raises(cli.OutputInsideRepository):
        cli.output_directory(str(ROOT / "roofline_out"))
    with pytest.raises(cli.OutputInsideRepository):
        cli.output_directory(str(ROOT))


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_floors_command_writes_outside_and_leaves_task_packages_unchanged(tmp_path):
    selected = ["triton/activation/silu_and_mul/bf16_d4096", "triton/gemm/gemm_a16w16/n1280_k8192"]
    before = {task_id: _tree_digest(ROOT / "tasks" / task_id) for task_id in selected}
    assert cli.main(["floors", "--out", str(tmp_path), "--tasks", *selected]) == 0
    rows = [json.loads(line) for line in (tmp_path / "floors.jsonl").read_text().splitlines()]
    assert {row["task_id"] for row in rows} == set(selected)
    assert (tmp_path / "floor_failures.jsonl").read_text() == ""
    assert before == {task_id: _tree_digest(ROOT / "tasks" / task_id) for task_id in selected}
