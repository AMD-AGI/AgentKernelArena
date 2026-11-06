"""src.tasks discovers schema-v3 task.yaml and schema-v2 config.yaml task packages."""
from pathlib import Path

import pytest

from src.tasks import get_task_config, task_category


def _write(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("schema_version: 3\n" if path.name == "task.yaml" else "schema_version: 2\n")


def test_discovers_task_yaml_and_config_yaml_tasks(tmp_path):
    _write(tmp_path, "triton/normalization/rms_norm/bf16_h8192/task.yaml")
    _write(tmp_path, "triton/gemm/gemm_a8w8/n4096_k4096/task.yaml")
    _write(tmp_path, "e2e/serving_fixture/config.yaml")
    _write(tmp_path, "triton/normalization/rms_norm/bf16_h8192/baseline/kernel.py")
    found = get_task_config(str(tmp_path))
    assert found == {
        "e2e/serving_fixture": str(tmp_path / "e2e/serving_fixture/config.yaml"),
        "triton/gemm/gemm_a8w8/n4096_k4096": str(tmp_path / "triton/gemm/gemm_a8w8/n4096_k4096/task.yaml"),
        "triton/normalization/rms_norm/bf16_h8192": str(tmp_path / "triton/normalization/rms_norm/bf16_h8192/task.yaml"),
    }
    assert list(found) == sorted(found)


def test_path_prefixes_select_by_baseline_language_category_and_operator(tmp_path):
    _write(tmp_path, "triton/normalization/rms_norm/bf16_h8192/task.yaml")
    _write(tmp_path, "flydsl/normalization/rms_norm/bf16_h7168/task.yaml")
    _write(tmp_path, "flydsl/gemm/gemm_a8w8/n5120_k1280/task.yaml")
    _write(tmp_path, "e2e/serving_fixture/config.yaml")
    assert list(get_task_config(str(tmp_path), category="flydsl")) == [
        "flydsl/gemm/gemm_a8w8/n5120_k1280", "flydsl/normalization/rms_norm/bf16_h7168"]
    assert list(get_task_config(str(tmp_path), category="triton/normalization")) == [
        "triton/normalization/rms_norm/bf16_h8192"]
    # A glob over the language component selects one category or operator under every baseline language.
    for selector in ("*/normalization", "*/normalization/rms_norm"):
        assert list(get_task_config(str(tmp_path), category=selector)) == [
            "flydsl/normalization/rms_norm/bf16_h7168", "triton/normalization/rms_norm/bf16_h8192"]
    assert list(get_task_config(str(tmp_path), category="e2e")) == ["e2e/serving_fixture"]
    # Prefixes match whole path components: a category alone no longer names a top-level directory.
    assert get_task_config(str(tmp_path), category="normalization") == {}
    assert get_task_config(str(tmp_path), category="triton/norm") == {}
    assert get_task_config(str(tmp_path), category="missing") == {}


def test_a_directory_declaring_both_files_is_rejected(tmp_path):
    _write(tmp_path, "triton/normalization/rms_norm/bf16_h8192/task.yaml")
    _write(tmp_path, "triton/normalization/rms_norm/bf16_h8192/config.yaml")
    with pytest.raises(ValueError, match="declares both task.yaml and config.yaml"):
        get_task_config(str(tmp_path))


@pytest.mark.parametrize(("task_name", "category"), [
    ("triton/normalization/rms_norm/bf16_h8192", "normalization"),
    ("tilelang/other/mhc_fused_post_pre/c4_d7168", "other"),
    ("e2e/qwen3_0_6b_sglang", "e2e"),
    ("SIKL-task/tiny_gemm", "SIKL-task"),
    ("standalone", ""),
])
def test_task_category_skips_the_baseline_language_directory(task_name, category):
    assert task_category(task_name) == category
