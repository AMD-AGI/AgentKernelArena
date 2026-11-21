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
