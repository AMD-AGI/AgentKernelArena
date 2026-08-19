import subprocess
import sys
from pathlib import Path

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.perf_helper_materialization import (
    AKA_HELPER_FILE_NAME,
    MARK_END,
    MARK_START,
    VLLM_HELPER_STUB_BLOCK,
    canonical_aka_helper,
    configured_performance_entrypoints,
    materialize_perf_helpers_in_workspace,
)
from src.tools.sync_perf_helpers import audit_task_benchmark_entrypoints


ROOT = Path(__file__).parents[1]


def test_materializes_helper_beside_root_performance_entrypoint(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "performance_command:\n  - python3 test_kernel_harness.py --full-benchmark\n"
    )
    (tmp_path / "test_kernel_harness.py").write_text(
        "from _aka_benchmark import benchmark_cuda_graph_or_events\n"
    )

    changed = materialize_perf_helpers_in_workspace(tmp_path)

    helper = tmp_path / AKA_HELPER_FILE_NAME
    assert helper in changed
    assert helper.read_text() == canonical_aka_helper(ROOT)
    assert materialize_perf_helpers_in_workspace(tmp_path) == []


def test_materializes_vllm_adapter_and_sibling_helper(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "task_runner.py"
    runner.write_text(
        f"{MARK_START}\n{VLLM_HELPER_STUB_BLOCK}{MARK_END}\n"
        "_benchmark_cuda_graph_or_events(lambda: None)\n"
    )
    (tmp_path / "config.yaml").write_text(
        "performance_command:\n  - python3 scripts/task_runner.py performance\n"
    )

    changed = materialize_perf_helpers_in_workspace(tmp_path)

    assert runner in changed
    assert "from _aka_benchmark import benchmark_cuda_graph_or_events" in runner.read_text()
    assert (scripts / AKA_HELPER_FILE_NAME).read_text() == canonical_aka_helper(ROOT)


def test_materializes_helper_beside_eval_tools_entrypoint(tmp_path):
    eval_tools = tmp_path / "eval_tools"
    eval_tools.mkdir()
    entrypoint = eval_tools / "cal_kernel_perf.py"
    entrypoint.write_text(
        "from _aka_benchmark import benchmark_cuda_graph_or_events_samples\n"
    )
    (tmp_path / "config.yaml").write_text(
        "performance_command:\n  - python3 eval_tools/cal_kernel_perf.py --hip_file hip/kernel.hip\n"
    )

    materialize_perf_helpers_in_workspace(tmp_path)

    assert (eval_tools / AKA_HELPER_FILE_NAME).read_text() == canonical_aka_helper(ROOT)


def test_dependency_free_yaml_fields_support_folded_and_flow_commands(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.perf_helper_materialization.yaml", None)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    folded = scripts / "folded runner.py"
    folded.write_text("print('folded')\n")
    direct = tmp_path / "direct.py"
    direct.write_text("print('direct')\n")
    harness = tmp_path / "harness.py"
    harness.write_text("print('harness')\n")
    (tmp_path / "config.yaml").write_text(
        "harness_path: 'harness.py'\n"
        "performance_command:\n"
        "  - >-\n"
        "    python3 'scripts/folded\n"
        "    runner.py' --benchmark\n"
        "  - python3 direct.py\n"
    )

    assert configured_performance_entrypoints(tmp_path) == {
        folded,
        direct,
        harness,
    }

    (tmp_path / "config.yaml").write_text(
        "performance_command: [python3 direct.py]\n"
    )
    assert configured_performance_entrypoints(tmp_path) == {direct}


def test_perf_helper_audit_runs_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "src/tools/sync_perf_helpers.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "benchmark entrypoints=" in result.stdout


def test_materializes_native_header_only_when_driver_includes_it(tmp_path):
    fake_root = tmp_path / "repo"
    perf = fake_root / "src/tools/perf"
    perf.mkdir(parents=True)
    (perf / "aka_benchmark.py").write_text("# canonical python\n")
    (perf / "native_hip_graph_benchmark.hpp").write_text("// canonical native\n")

    workspace = tmp_path / "workspace"
    native = workspace / "scripts/native"
    native.mkdir(parents=True)
    (native / "benchmark_driver.hip").write_text(
        '#include "hip_graph_benchmark.hpp"\n'
    )

    materialize_perf_helpers_in_workspace(workspace, root=fake_root)

    assert (native / "hip_graph_benchmark.hpp").read_text() == "// canonical native\n"


def test_native_helper_supports_paired_forced_event_baseline():
    header = (ROOT / "src/tools/perf/native_hip_graph_benchmark.hpp").read_text()

    assert 'std::getenv("AKA_BENCHMARK_FORCE_EVENT")' in header
    assert '"forced_event_baseline"' in header


def test_rocmbench_adapter_forwards_state_and_method_policy():
    adapter = (ROOT / "src/tools/perf/performance_utils_pytest.py").read_text()

    assert "prepare_fn: Callable[[], Any] | None = None" in adapter
    assert "prepare_fn=self.prepare_fn" in adapter
    assert "use_cuda_graph=self.use_cuda_graph" in adapter
    assert "fallback_reason=self.fallback_reason" in adapter


def test_config_declared_entrypoint_and_generated_helper_are_protected(tmp_path):
    eval_tools = tmp_path / "eval_tools"
    eval_tools.mkdir()
    entrypoint = eval_tools / "cal_kernel_perf.py"
    entrypoint.write_text("print('honest benchmark')\n")
    helper = eval_tools / AKA_HELPER_FILE_NAME
    helper.write_text("# generated\n")
    (tmp_path / "config.yaml").write_text(
        "performance_command:\n  - python3 eval_tools/cal_kernel_perf.py --hip_file hip/kernel.hip\n"
    )

    snapshot = snapshot_workspace_harness(tmp_path)
    entrypoint.write_text("print('fake score')\n")

    with pytest.raises(RuntimeError, match="cal_kernel_perf.py"):
        verify_workspace_harness(snapshot)

    entrypoint.write_text("print('honest benchmark')\n")
    snapshot = snapshot_workspace_harness(tmp_path)
    helper.unlink()
    with pytest.raises(RuntimeError, match="_aka_benchmark.py"):
        verify_workspace_harness(snapshot)


def test_native_driver_and_materialized_header_are_protected(tmp_path):
    native = tmp_path / "scripts/native"
    native.mkdir(parents=True)
    driver = native / "benchmark_driver.hip"
    header = native / "hip_graph_benchmark.hpp"
    driver.write_text('#include "hip_graph_benchmark.hpp"\n')
    header.write_text("// generated helper\n")

    snapshot = snapshot_workspace_harness(tmp_path)
    driver.write_text("// bypass timing\n")
    with pytest.raises(RuntimeError, match="benchmark_driver.hip"):
        verify_workspace_harness(snapshot)

    driver.write_text('#include "hip_graph_benchmark.hpp"\n')
    snapshot = snapshot_workspace_harness(tmp_path)
    header.write_text("// fake timing\n")
    with pytest.raises(RuntimeError, match="hip_graph_benchmark.hpp"):
        verify_workspace_harness(snapshot)


def test_every_task_performance_entrypoint_uses_a_supported_family():
    counts, problems = audit_task_benchmark_entrypoints(ROOT)
    config_count = len(list((ROOT / "tasks").rglob("config.yaml")))

    assert config_count == 426
    assert problems == []
    assert sum(counts.values()) == config_count
    assert counts == {
        "canonical_python": 220,
        "native_graph_driver": 6,
        "rocmbench_adapter": 61,
        "vllm_adapter": 139,
    }


def test_colocated_kernel_body_is_editable_but_harness_remains_protected(tmp_path):
    entrypoint = tmp_path / "combined_benchmark.py"
    entrypoint.write_text(
        "def kernel(x):\n"
        "    return x\n\n"
        "def test_performance():\n"
        "    return kernel(1)\n"
    )
    (tmp_path / "config.yaml").write_text(
        "source_file_path:\n"
        "  - combined_benchmark.py\n"
        "target_kernel_functions:\n"
        "  - kernel\n"
        "performance_command:\n"
        "  - pytest combined_benchmark.py -k test_performance\n"
    )

    snapshot = snapshot_workspace_harness(tmp_path)
    entrypoint.write_text(
        "def kernel(x):\n"
        "    return x * 2\n\n"
        "def test_performance():\n"
        "    return kernel(1)\n"
    )
    verify_workspace_harness(snapshot)

    snapshot = snapshot_workspace_harness(tmp_path)
    entrypoint.write_text(
        "def kernel(x):\n"
        "    return x * 2\n\n"
        "def test_performance():\n"
        "    return 0.000001\n"
    )
    with pytest.raises(RuntimeError, match="combined_benchmark.py"):
        verify_workspace_harness(snapshot)


def test_colocated_kernel_imports_structure_and_triton_helpers_are_editable(tmp_path):
    entrypoint = tmp_path / "combined_benchmark.py"
    entrypoint.write_text(
        "import triton\n\n"
        "@triton.jit\n"
        "def kernel(x):\n"
        "    return x\n\n"
        "def test_performance():\n"
        "    return kernel(1)\n"
    )
    (tmp_path / "config.yaml").write_text(
        "source_file_path: [combined_benchmark.py]\n"
        "target_kernel_functions: [kernel]\n"
        "performance_command: [python3 combined_benchmark.py]\n"
    )

    snapshot = snapshot_workspace_harness(tmp_path)
    entrypoint.write_text(
        "from math import prod\n"
        "import triton\n\n"
        "@triton.jit\n"
        "def helper(x):\n"
        "    return x * 2\n\n"
        "@triton.autotune(configs=[triton.Config({}, num_warps=8)], key=['x'])\n"
        "@triton.jit\n"
        "def kernel(x, extra=None):\n"
        "    return helper(x)\n\n"
        "def test_performance():\n"
        "    return kernel(1)\n"
    )

    verify_workspace_harness(snapshot)


def test_legacy_instruction2triton_entrypoint_is_implied_editable_source(tmp_path):
    entrypoint = tmp_path / "test_kernel.py"
    entrypoint.write_text(
        "import triton\n\n"
        "@triton.jit\n"
        "def kernel(x):\n"
        "    return x\n\n"
        "def test_performance():\n"
        "    return kernel(1)\n"
    )
    (tmp_path / "config.yaml").write_text(
        "task_type: instruction2triton\n"
        "source_file_path: []\n"
        "target_kernel_functions: [kernel]\n"
        "performance_command: [pytest test_kernel.py -k test_performance]\n"
    )

    snapshot = snapshot_workspace_harness(tmp_path)
    entrypoint.write_text(
        "import triton\n\n"
        "@triton.jit\n"
        "def kernel(x):\n"
        "    return x * 2\n\n"
        "def test_performance():\n"
        "    return kernel(1)\n"
    )
    verify_workspace_harness(snapshot)

    snapshot = snapshot_workspace_harness(tmp_path)
    entrypoint.write_text(
        "import triton\n\n"
        "@triton.jit\n"
        "def kernel(x):\n"
        "    return x * 2\n\n"
        "def test_performance():\n"
        "    return 0.000001\n"
    )
    with pytest.raises(RuntimeError, match="test_kernel.py"):
        verify_workspace_harness(snapshot)


def test_colocated_module_constants_and_non_triton_helpers_stay_protected(tmp_path):
    entrypoint = tmp_path / "combined_benchmark.py"
    entrypoint.write_text(
        "SHAPES = [1]\n\n"
        "def make_input():\n"
        "    return SHAPES[0]\n\n"
        "def kernel(x):\n"
        "    return x\n\n"
        "def test_performance():\n"
        "    return kernel(make_input())\n"
    )
    (tmp_path / "config.yaml").write_text(
        "source_file_path: [combined_benchmark.py]\n"
        "target_kernel_functions: [kernel]\n"
        "performance_command: [python3 combined_benchmark.py]\n"
    )

    snapshot = snapshot_workspace_harness(tmp_path)
    entrypoint.write_text(
        "SHAPES = [4096]\n\n"
        "def make_input():\n"
        "    return SHAPES[-1]\n\n"
        "def kernel(x):\n"
        "    return x * 2\n\n"
        "def test_performance():\n"
        "    return kernel(make_input())\n"
    )

    with pytest.raises(RuntimeError, match="combined_benchmark.py"):
        verify_workspace_harness(snapshot)
