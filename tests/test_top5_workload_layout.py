"""CPU checks for workload discovery and preserved task-local contracts."""
from collections import Counter
import json
from pathlib import Path

import yaml

from head_kernel_test_utils import SUITE, task_directories
from src.tasks import get_task_config


WORKLOAD = "isl8192_osl1024_conc64_tp8_mi355x"
MODEL_COUNTS = {
    "deepseek-v4-pro": 3, "glm-5.3-flash": 4, "kimi-k3": 3,
    "minimax-m3-mxfp4": 3, "qwen3.8-2.4t-a95b-mxfp4": 5,
}


def test_eighteen_distinct_tasks_are_grouped_by_exact_model_workload_and_image():
    tasks = task_directories()
    assert len(tasks) == 18
    assert not list(SUITE.glob("*/config.yaml"))
    assert Counter(path.relative_to(SUITE).parts[0] for path in tasks.values()) == MODEL_COUNTS
    selectors = set()
    for operation, task in tasks.items():
        config = yaml.safe_load((task / "config.yaml").read_text())
        model, workload, image, kernel = task.relative_to(SUITE).parts
        assert model == config["headkernel"]["model"].lower()
        assert workload == WORKLOAD
        capture = config["headkernel"]["capture_runtime"]["image"]
        assert image == capture.rsplit("/", 1)[-1].replace(":", "_")
        assert config["headkernel"]["docker"].startswith("docker.io/rocm/hyperloom@sha256:")
        assert kernel == operation.split("__", 1)[1]
        assert config["headkernel"]["serving"] == "ISL 8192 / OSL 1024 / CONC 64 / TP 8"
        assert config["platform_support"]["required_arch"] == "gfx950"
        shapes = json.loads((task / "SHAPES.json").read_text())
        assert shapes["task_id"] == operation
        assert (task / "SHAPES.md").is_file()
        selectors.add(task.relative_to(SUITE.parent).as_posix())
    discovered = get_task_config(str(SUITE.parent), "head_kernels")
    assert set(discovered) == selectors


def test_six_image_workloads_list_every_kernel_and_its_local_shape_and_runtime():
    tasks = task_directories()
    groups = {task.parent for task in tasks.values()}
    assert len(groups) == 6
    seen = []
    for group in groups:
        workload = json.loads((group / "workload.json").read_text())
        assert workload["serving_capture"] == {
            "input_sequence_length": 8192, "output_sequence_length": 1024,
            "concurrency": 64, "tensor_parallel_world_size": 8,
        }
        assert workload["device"] == {"model": "MI355X", "architecture": "gfx950"}
        assert workload["workload_slug"] == WORKLOAD
        assert workload["kernel_count"] == len(workload["kernels"])
        assert (group / "README.md").is_file()
        for row in workload["kernels"]:
            task = tasks[row["operation_id"]]
            config = yaml.safe_load((group / row["config"]).read_text())
            assert task.parent == group
            assert row["selector"] == task.relative_to(SUITE.parent).as_posix()
            assert row["runtime"]["image"] == workload["docker"] == config["headkernel"]["docker"]
            assert workload["capture_runtime"] == config["headkernel"]["capture_runtime"]
            assert workload["image_group_role"] == "historical_serving_capture"
            assert row["runtime"]["expected_image_id"] == config["headkernel"]["runtime"]["expected_image_id"]
            assert (group / row["shapes"]).resolve() == task / "SHAPES.json"
            assert (group / row["shape_guide"]).resolve() == task / "SHAPES.md"
            assert row["runtime"]["gpu_arch"] == "gfx950"
            seen.append(row["operation_id"])
    assert Counter(seen) == Counter(tasks.keys())


def test_catalog_selectors_resolve_and_composite_moe_keeps_one_captured_seam():
    tasks = task_directories()
    catalog = json.loads((SUITE / "catalog.json").read_text())
    registered = [row for row in catalog["rows"] if row.get("state") == "registered_validation_pending"]
    seen = set()
    for row in registered:
        task = tasks[row["operation_id"]]
        assert row["task"] == task.relative_to(SUITE.parent).as_posix()
        assert row["task_path"] == task.relative_to(SUITE.parents[1]).as_posix()
        seen.add(row["operation_id"])
    assert seen == set(tasks)
    composite = "qwen3.8-2.4t__fused_moe_2stage_mxfp4"
    assert len({row["task"] for row in registered if row["operation_id"] == composite}) == 1
    workload = json.loads((tasks[composite].parent / "workload.json").read_text())
    entry, = [row for row in workload["kernels"] if row["operation_id"] == composite]
    assert "One captured composite MoE stage-1 + stage-2 seam" in entry["capture_contract"]


def test_all_internal_task_symlinks_remain_resolvable_inside_their_task():
    count = 0
    for task in task_directories().values():
        for path in task.rglob("*"):
            if path.is_symlink():
                count += 1
                assert path.resolve(strict=True).is_relative_to(task)
    assert count > 0
