"""Task selectors retain one runnable package per migrated task identity."""

from pathlib import Path

import pytest
import yaml

from src.task_spec import load_task_spec
from src.tasks import get_task_config


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"


def test_flattened_hip_tasks_are_unique_and_loadable():
    discovered = get_task_config(str(TASKS), "hip2hip")
    assert len(discovered) == 34
    assert all(len(Path(task_id).parts) == 2 for task_id in discovered)
    # Ignored bytecode from an older checkout may retain these directories.
    # None may retain an authored task package.
    for group in ("gpumode", "others", "vllm"):
        assert not get_task_config(str(TASKS), f"hip2hip/{group}")
    for task_id, config_path in discovered.items():
        assert load_task_spec(Path(config_path), task_id=task_id).to_mapping()["schema_version"] == 2


def test_rocmbench_has_one_registered_flip_and_no_old_classification():
    discovered = get_task_config(str(TASKS), "triton2triton/rocmbench")
    assert len(discovered) == 31
    flip = "triton2triton/rocmbench/test_triton_flip"
    assert flip in discovered
    assert load_task_spec(Path(discovered[flip]), task_id=flip).to_mapping()["schema_version"] == 2
    assert (TASKS / "triton2triton/rocmbench/easy/test_triton_flip/test_triton_flip.py").is_file()
    assert not (TASKS / "triton2triton/rocmbench/easy/test_triton_flip/config.yaml").exists()
    assert not get_task_config(str(TASKS), "instruction2triton")
    serving = get_task_config(str(TASKS), "e2e")
    assert len(get_task_config(str(TASKS))) == 439 + len(serving)


@pytest.mark.parametrize("name", [
    "benchmark_cursor_mi355x.yaml",
    "quickstart_claude_mi355x.yaml",
])
def test_updated_example_selectors_resolve_without_duplicates(name):
    selectors = yaml.safe_load((ROOT / "example_configs" / name).read_text())["tasks"]
    assert len(selectors) == len(set(selectors))
    assert all("instruction2triton" not in selector for selector in selectors)
    for selector in selectors:
        assert get_task_config(str(TASKS), selector), selector


def test_cursor_example_preserves_unique_historical_task_selection():
    original = yaml.safe_load(
        (ROOT / "tests/fixtures/benchmark_cursor_original_60.yaml").read_text()
    )["tasks"]
    assert len(original) == 60
    retained = get_task_config(str(TASKS), "triton2triton/rocmbench")
    mapped = []
    for selector in original:
        parts = selector.split("/")
        if parts[0] == "hip2hip":
            assert parts[1] in {"gpumode", "others", "vllm"}
            replacement = "hip2hip/" + "/".join(parts[2:])
        elif parts[0] == "instruction2triton":
            matches = [task_id for task_id in retained
                       if task_id.rsplit("/", 1)[-1] == parts[-1]]
            assert len(matches) == 1, selector
            replacement = matches[0]
        else:
            replacement = selector
        mapped.append(replacement)
    expected = list(dict.fromkeys(mapped))
    assert len(expected) == 54
    actual = yaml.safe_load((ROOT / "example_configs/benchmark_cursor_mi355x.yaml").read_text())["tasks"]
    assert actual == expected
