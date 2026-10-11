"""The native quant task freezes new UT imports without freezing GPU edits."""
from pathlib import Path
import shutil

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness


TASK = Path(__file__).resolve().parents[1] / "tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8"
SOURCE = "source/quant_kernels.cu"


@pytest.fixture
def workspace(tmp_path):
    target = tmp_path / "workspace"
    shutil.copytree(TASK, target, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "build", ".pytest_cache"))
    return target


@pytest.fixture(params=[False, True], ids=["workspace_only", "original_task_root"])
def snapshot(workspace, request):
    return snapshot_workspace_harness(workspace, **({"task_root": TASK} if request.param else {}))


@pytest.mark.parametrize("relative", ["ut/torch.py", "ut/torch/__init__.py", "ut/json.py"])
def test_new_ut_shadow_module_is_discarded_before_evaluation(workspace, snapshot, relative):
    injected = workspace / relative
    injected.parent.mkdir(parents=True, exist_ok=True)
    # Source text only: the probe is never imported, compiled, or executed.
    injected.write_text("# Harmless source-guard regression probe.\nPROBE = True\n")
    verify_workspace_harness(snapshot)
    assert not injected.exists()


def test_existing_ut_oracle_remains_immutable(workspace, snapshot):
    oracle = workspace / "ut/oracle.py"
    oracle.write_bytes(oracle.read_bytes() + b"\n# changed oracle\n")
    with pytest.raises(RuntimeError, match="Protected test/harness files changed"):
        verify_workspace_harness(snapshot)


def test_kernel_body_edits_remain_allowed(workspace, snapshot):
    path = workspace / SOURCE
    source = path.read_text()
    start = source.index("{", source.index("dynamic_per_group_scaled_quant_kernel(")) + 1
    path.write_text(source[:start] + "\n    const int guard_test_tuning = 1;\n" + source[start:])
    verify_workspace_harness(snapshot)
    assert "guard_test_tuning" in path.read_text()


def test_build_and_framework_runtime_outputs_remain_writable(workspace, snapshot):
    outputs = ["build/performance_report.json", "build/aiter_jit/candidate.so", "logs/worker.log",
               ".validator_audit/harness_inventory.json", "task_result.yaml",
               "validation_report.yaml", ".validation_complete", "baseline_perf.yaml", "optimized_perf.yaml"]
    for relative in outputs:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime output\n")
    verify_workspace_harness(snapshot)
    assert all((workspace / relative).read_text() == "runtime output\n" for relative in outputs)
