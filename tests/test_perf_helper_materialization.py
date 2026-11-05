"""Serving helper materialization and the committed benchmark-entrypoint audit."""
import json
import shutil
import sys
from pathlib import Path

from src.perf_helper_materialization import (
    SERVING_BACKEND_HELPERS,
    SERVING_HELPERS,
    SERVING_RECIPE_RUNNER_STUB,
    materialize_perf_helpers_in_workspace,
)
from src.tools.sync_perf_helpers import audit_task_benchmark_entrypoints


ROOT = Path(__file__).resolve().parents[1]
SERVING_TASK = ROOT / "tasks/e2e/qwen3_0_6b_sglang"


def _fake_root(tmp_path):
    root = tmp_path / "repo"
    for source in {**SERVING_HELPERS, **SERVING_BACKEND_HELPERS}.values():
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# canonical {path.name}\n")
    return root


def _runner(workspace, text):
    scripts = workspace / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "evaluate.py").write_text(text)
    return scripts


def test_serving_client_receives_canonical_helpers_idempotently(tmp_path):
    root = _fake_root(tmp_path)
    scripts = _runner(tmp_path / "ws", "from _aka_serving import main\n")

    written = materialize_perf_helpers_in_workspace(tmp_path / "ws", root=root)

    assert sorted(path.name for path in written) == sorted(SERVING_HELPERS)
    for name, source in SERVING_HELPERS.items():
        assert (scripts / name).read_text() == (root / source).read_text()
    assert not any((scripts / name).exists() for name in SERVING_BACKEND_HELPERS)
    assert materialize_perf_helpers_in_workspace(tmp_path / "ws", root=root) == []


def test_scenario_client_also_receives_backend_helpers(tmp_path):
    root = _fake_root(tmp_path)
    scripts = _runner(tmp_path / "ws", "from _aka_serving import main\nimport _aka_serving_backends\n")

    materialize_perf_helpers_in_workspace(tmp_path / "ws", root=root)

    for name, source in {**SERVING_HELPERS, **SERVING_BACKEND_HELPERS}.items():
        assert (scripts / name).read_text() == (root / source).read_text()


def test_recipe_stub_and_other_workspaces_are_left_unchanged(tmp_path):
    root = _fake_root(tmp_path)
    _runner(tmp_path / "recipe", SERVING_RECIPE_RUNNER_STUB)
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel/kernel.py").write_text("import _aka_benchmark\n")

    assert materialize_perf_helpers_in_workspace(tmp_path / "recipe", root=root) == []
    assert materialize_perf_helpers_in_workspace(tmp_path / "kernel", root=root) == []
    assert sorted(p.name for p in (tmp_path / "recipe/scripts").iterdir()) == ["evaluate.py"]
    assert not (tmp_path / "kernel/_aka_benchmark.py").exists()


def test_real_serving_task_materializes_the_repository_helpers(tmp_path):
    workspace = tmp_path / "task"
    shutil.copytree(SERVING_TASK, workspace, ignore=shutil.ignore_patterns("__pycache__"))

    materialize_perf_helpers_in_workspace(workspace)

    assert ((workspace / "scripts/_aka_serving.py").read_text()
            == (ROOT / "src/tools/perf/serving.py").read_text())
    assert ((workspace / "scripts/_aka_measurement.py").read_text()
            == (ROOT / "src/measurement.py").read_text())


def _task_tree(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(SERVING_TASK, root / "tasks/e2e/client", ignore=shutil.ignore_patterns("__pycache__"))
    kernel = root / "tasks/normalization/rms_norm/bf16"
    kernel.mkdir(parents=True)
    (kernel / "task.yaml").write_text("schema_version: 3\n")
    return root


def test_audit_accepts_serving_clients_and_counts_v3_tasks(tmp_path):
    counts, problems = audit_task_benchmark_entrypoints(_task_tree(tmp_path))

    assert problems == []
    assert counts == {"kernel_v3": 1, "serving_client": 1}


def test_audit_rejects_non_serving_v2_tasks_and_committed_generated_helpers(tmp_path):
    root = _task_tree(tmp_path)
    legacy = root / "tasks/hip2hip/old"
    legacy.mkdir(parents=True)
    (legacy / "config.yaml").write_text("schema_version: 2\n")
    (root / "tasks/normalization/rms_norm/bf16/_aka_benchmark.py").write_text("# copy\n")
    (root / "tasks/e2e/client/scripts/_aka_serving.py").write_text("# copy\n")
    (root / "tasks/normalization/rms_norm/bf16/_aka_eval").mkdir()

    _, problems = audit_task_benchmark_entrypoints(root)

    assert any("tasks/hip2hip/old" in problem and "serving" in problem for problem in problems)
    for generated in ("bf16/_aka_benchmark.py", "scripts/_aka_serving.py", "bf16/_aka_eval"):
        assert any(generated in problem and "committed" in problem for problem in problems), generated


def test_audit_command_fails_on_problems_and_passes_otherwise(tmp_path, monkeypatch, capsys):
    import src.tools.sync_perf_helpers as audit

    root = _task_tree(tmp_path)
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["sync_perf_helpers.py"])
    assert audit.main() == 0
    assert "serving_client" in capsys.readouterr().out

    (root / "tasks/stray").mkdir()
    (root / "tasks/stray/config.yaml").write_text(json.dumps({"schema_version": 2}))
    assert audit.main() == 1
    assert "tasks/stray" in capsys.readouterr().err
