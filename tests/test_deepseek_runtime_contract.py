"""Bind the documented initial-runtime dependencies to the executable wrappers."""

import ast
import json
from pathlib import Path

import pytest
import yaml

from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = yaml.safe_load(
    (ROOT / "example_configs/task_validator_deepseek_drafts_mi355x.yaml").read_text()
)
TASKS = [
    ROOT / "tasks" / name for name in RUN_CONFIG["tasks"]
    if Path(name).name.startswith(("gemm_", "mhc_"))
]


def production_entrypoint(path):
    """Resolve the active imported operator, including GEMM's partial binding."""
    tree = ast.parse(path.read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    function = functions["run"]
    if "_blockwise_scaled_baseline" in functions:
        binding = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_callable"
                    for target in node.targets)
        )
        assert isinstance(binding, ast.Call)
        assert binding.args[0].id == "_blockwise_scaled_baseline"
        options = ast.literal_eval(binding.keywords[0].value)
        # All retained GEMMs use the preshuffled operator; the generic plain
        # branch in the wrapper is unreachable for these fixed task bindings.
        assert options["a_scale_storage"] in {"raw", "logical"}
        function = functions["_blockwise_scaled_baseline"]
    result = next(node for node in reversed(function.body) if isinstance(node, ast.Return))
    assert isinstance(result.value, ast.Call)
    assert isinstance(result.value.func, ast.Name)
    called_name = result.value.func.id
    imports = {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for node in ast.walk(function) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    entrypoint = imports[called_name]
    assert entrypoint.startswith("aiter.")
    return entrypoint


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_initial_runtime_matches_wrappers_and_declared_target(task):
    spec = load_task_spec(task / "config.yaml", task_id=f"Aiter-task/{task.name}")
    assert spec.candidate.initial_state == "implemented"
    assert spec.candidate.initial_language == "python"
    assert spec.candidate.language == "triton"
    assert spec.baseline.kind == "provided"
    initial = production_entrypoint(task / "source/implementation/main.py")
    baseline = production_entrypoint(task / "scripts/baseline/main.py")
    assert initial == baseline

    readme = (task / "BUNDLE_README.md").read_text()
    config = yaml.safe_load((task / "config.yaml").read_text())
    for declaration in (readme, config["description"]):
        assert initial in declaration
        assert RUN_CONFIG["docker_image"] in declaration
        assert "task_validation" in declaration
        assert "unchanged initial wrapper" in declaration
        assert "final submitted candidate must implement its own gpu" in " ".join(
            declaration.split()
        ).lower()
        assert "ROCm PyTorch and Triton" in declaration
    workload = json.loads((task / "scripts/workload.json").read_text())
    assert workload["bundle_readme"] == readme
