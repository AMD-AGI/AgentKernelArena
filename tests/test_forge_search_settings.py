"""What the Forge adapter tells the engine about the search beyond the task itself."""
from __future__ import annotations

from pathlib import Path

import pytest

from test_forge_v2 import fixture_task
from agents.forge import adapter


def _values(command: list[str], option: str) -> list[str]:
    """Every value ``option`` carries on the command line, in order."""
    return [command[index + 1] for index, word in enumerate(command) if word == option]


def _loop_command(tmp_path, overrides=None, *, workflow="optimize", capabilities=None):
    context, plan, _ = fixture_task(tmp_path, language="triton")
    (Path(plan["engine_root"]) / "solution/nested").mkdir()
    plan.update(workflow=workflow, **(capabilities or {}))
    config = adapter._config({"agent": overrides or {}})
    return adapter.build_command(plan, context, config, gpu_arch="gfx942", gpu_type="mi300x")


@pytest.mark.parametrize("workflow", ["optimize", "initialize"])
def test_the_loop_may_commit_a_new_file_wherever_the_task_lets_a_candidate_add_one(tmp_path, workflow):
    """A session that splits the kernel into a new module would otherwise have its whole change rolled back."""
    command = _loop_command(tmp_path, workflow=workflow)

    assert _values(command, "--commit-new-path") == ["solution/*", "solution/nested/*"]


def test_the_lane_count_is_the_engines_unless_the_run_names_one(tmp_path):
    assert "--lanes" not in _loop_command(tmp_path)


def test_a_run_may_ask_for_a_lane_count(tmp_path):
    assert _values(_loop_command(tmp_path, {"lanes": 1}), "--lanes") == ["1"]


@pytest.mark.parametrize("lanes", [0, -1, "3", 1.0, True])
def test_a_lane_count_that_is_not_a_positive_integer_is_refused(lanes):
    with pytest.raises(ValueError, match="Forge lanes must be a positive integer"):
        adapter._config({"agent": {"lanes": lanes}})


def test_an_engine_that_accepts_it_is_told_arena_scores_by_the_geometric_mean(tmp_path):
    """An engine searching by another mean keeps candidates Arena scores lower."""
    command = _loop_command(tmp_path, capabilities={"case_aggregate": True})

    assert _values(command, "--case-aggregate") == ["geometric"]


@pytest.mark.parametrize("capabilities", [None, {"case_aggregate": False}], ids=["unreported", "declined"])
def test_an_engine_that_cannot_be_told_the_aggregate_still_runs(tmp_path, capabilities):
    assert "--case-aggregate" not in _loop_command(tmp_path, capabilities=capabilities)
