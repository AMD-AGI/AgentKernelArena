"""src.tasks discovers schema-v3 task.yaml and schema-v2 config.yaml task packages."""
from pathlib import Path

import pytest

from src.tasks import get_task_config


def _write(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("schema_version: 3\n" if path.name == "task.yaml" else "schema_version: 2\n")


def test_discovers_task_yaml_and_config_yaml_tasks(tmp_path):
    _write(tmp_path, "normalization/rms_norm/bf16_h8192/task.yaml")
    _write(tmp_path, "gemm/gemm_a8w8/n4096_k4096/task.yaml")
    _write(tmp_path, "e2e/serving_fixture/config.yaml")
    _write(tmp_path, "normalization/rms_norm/bf16_h8192/baseline/kernel.py")
    found = get_task_config(str(tmp_path))
    assert found == {
        "e2e/serving_fixture": str(tmp_path / "e2e/serving_fixture/config.yaml"),
        "gemm/gemm_a8w8/n4096_k4096": str(tmp_path / "gemm/gemm_a8w8/n4096_k4096/task.yaml"),
        "normalization/rms_norm/bf16_h8192": str(tmp_path / "normalization/rms_norm/bf16_h8192/task.yaml"),
    }
    assert list(found) == sorted(found)


def test_category_prefix_filters_both_kinds(tmp_path):
    _write(tmp_path, "normalization/rms_norm/bf16_h8192/task.yaml")
    _write(tmp_path, "e2e/serving_fixture/config.yaml")
    assert list(get_task_config(str(tmp_path), category="normalization")) == ["normalization/rms_norm/bf16_h8192"]
    assert list(get_task_config(str(tmp_path), category="e2e")) == ["e2e/serving_fixture"]
    assert get_task_config(str(tmp_path), category="missing") == {}


def test_a_directory_declaring_both_files_is_rejected(tmp_path):
    _write(tmp_path, "normalization/rms_norm/bf16_h8192/task.yaml")
    _write(tmp_path, "normalization/rms_norm/bf16_h8192/config.yaml")
    with pytest.raises(ValueError, match="declares both task.yaml and config.yaml"):
        get_task_config(str(tmp_path))
