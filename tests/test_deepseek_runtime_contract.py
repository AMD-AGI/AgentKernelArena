"""Bind the documented runtime dependencies to the executable baseline wrappers."""

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


def test_run_config_lists_every_functional_package():
    packages = {p.parents[1] for p in (ROOT / "tasks/Aiter-task").glob("*/scripts/workload.json")}
    assert set(TASKS) == packages and len(TASKS) == 16


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
def test_declared_runtime_matches_baseline_wrapper_and_target(task):
    spec = load_task_spec(task / "config.yaml", task_id=f"Aiter-task/{task.name}")
    assert spec.candidate.initial_state == "unimplemented"
    assert spec.candidate.language == "flydsl"
    assert spec.baseline.kind == "provided"
    baseline = production_entrypoint(task / "scripts/baseline/main.py")

    readme = (task / "BUNDLE_README.md").read_text()
    config = yaml.safe_load((task / "config.yaml").read_text())
    for declaration in (readme, config["description"]):
        flat = " ".join(declaration.split())
        assert baseline in declaration
        assert RUN_CONFIG["docker_image"] in declaration
        assert f"build_{task.name}_module" in flat
        assert "FlyDSL" in declaration and "unimplemented" in declaration
        assert "Triton" not in declaration and "run(**kwargs)" not in declaration
    workload = json.loads((task / "scripts/workload.json").read_text())
    assert workload["bundle_readme"] == readme
