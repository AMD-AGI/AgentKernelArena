# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for src/tasks.py task discovery (CPU-only, no GPU deps)."""
import os

from src.tasks import get_task_config


def _make_task(root, rel, body="task_type: hip2hip\n"):
    task_dir = root / rel
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "config.yaml").write_text(body, encoding="utf-8")


def test_discovers_all_configs(tmp_path):
    _make_task(tmp_path, "hip2hip/gpumode/GELU")
    _make_task(tmp_path, "triton2triton/vllm/triton_rms_norm", "task_type: triton2triton\n")

    result = get_task_config(str(tmp_path))

    expected = {
        os.path.join("hip2hip", "gpumode", "GELU"),
        os.path.join("triton2triton", "vllm", "triton_rms_norm"),
    }
    assert set(result.keys()) == expected


def test_category_filter(tmp_path):
    _make_task(tmp_path, "hip2hip/gpumode/GELU")
    _make_task(tmp_path, "torch2hip/gpumode/x", "task_type: torch2hip\n")

    result = get_task_config(str(tmp_path), category="hip2hip")

    assert list(result.keys()) == [os.path.join("hip2hip", "gpumode", "GELU")]


def test_config_path_points_to_config_yaml(tmp_path):
    _make_task(tmp_path, "a/b/c")

    result = get_task_config(str(tmp_path))

    (config_path,) = result.values()
    assert config_path.endswith("config.yaml")


def test_empty_root_returns_empty(tmp_path):
    assert get_task_config(str(tmp_path)) == {}
