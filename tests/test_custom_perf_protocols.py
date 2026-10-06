"""Task-local protocol recognition remains static and rejects harness drift."""

import hashlib
import json
from pathlib import Path

import pytest

from src.perf_helper_materialization import configured_performance_entrypoints
from src.tools.custom_perf_protocols import REGISTRY
from src.tools.sync_perf_helpers import audit_task_benchmark_entrypoints


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    ("headkernel/deepseek-v4-pro__moe_stage1_grouped_gemm_silu_opus_a8w4", "portable_case_contract"),
    ("headkernel/kimi-k3__moe_gemm1_stage1", "portable_case_contract"),
    ("headkernel/deepseek-v4-pro__dsa_sparse_mla_attn", "legacy_headkernel"),
    ("headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8", "isolated_native_graph"),
    ("headkernel/glm-5.3-flash__elementwise_copy_cluster", "blocked_entrypoint"),
    ("headkernel/minimax-m3__gqa_share_sparse_fwd_kernel", "portable_case_contract"),
    ("headkernel/deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl", "portable_case_contract"),
    ("headkernel/kimi-k3__attn_residual_aggregate_hip", "portable_case_contract"),
    ("headkernel/kimi-k3__dense_bf16_gemm_cijk", "portable_case_contract"),
    ("headkernel/minimax-m3__gemm_afp4wfp4_kernel", "portable_case_contract"),
    ("headkernel/minimax-m3__decode_score_kernel", "portable_case_contract"),
    ("headkernel/minimax-m3__gqa_share_sparse_decode_kernel", "portable_case_contract"),
]


def stage_protocol(root, example):
    """Copy only static audit inputs; no task imports, fixtures or GPU access."""
    source = ROOT / "tasks" / example
    entrypoint, = configured_performance_entrypoints(source)
    protocols = json.loads(REGISTRY.read_text())["protocols"]
    record, = [record for record in protocols if source / record["entrypoint"] == entrypoint
               and hashlib.sha256((source / record["entrypoint"]).read_bytes()).hexdigest()
               in record["entrypoint_sha256"]]
    task = root / "tasks" / "arbitrary_task_name"
    files = {record["entrypoint"], *record["files_sha256"]}
    if record["family"] == "portable_case_contract":
        files.update(("ut/evaluation_contract.py", "cases.json"))
    for relative in files:
        target = task / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / relative).read_bytes())
    (task / "config.yaml").write_text(
        "performance_command:\n- python3 " + record["entrypoint"] + " performance\n")
    return task, record["entrypoint"]


@pytest.mark.parametrize("example,family", EXAMPLES)
def test_recognizes_protocol_without_task_name_or_task_execution(tmp_path, example, family):
    stage_protocol(tmp_path, example)
    assert audit_task_benchmark_entrypoints(tmp_path) == ({family: 1}, [])


@pytest.mark.parametrize("relative", ["scripts/task_runner.py", "ut/fresh_runner.py", "ut/evaluation_contract.py"])
def test_changed_portable_runner_or_helper_is_rejected(tmp_path, relative):
    task, _ = stage_protocol(tmp_path, EXAMPLES[1][0])
    path = task / relative
    path.write_text(path.read_text() + "\n# changed implementation requires review\n")
    counts, problems = audit_task_benchmark_entrypoints(tmp_path)
    assert counts == {} and len(problems) == 1


@pytest.mark.parametrize("example,relative", [
    (EXAMPLES[2][0], "scripts/_bench.py"),
    (EXAMPLES[2][0], "ut/harness_lib.py"),
    (EXAMPLES[3][0], "scripts/worker.py"),
    (EXAMPLES[3][0], "ut/contract.py"),
])
def test_changed_legacy_or_native_timer_is_rejected(tmp_path, example, relative):
    task, _ = stage_protocol(tmp_path, example)
    (task / relative).write_text("def measure(): return 0.001\n")
    counts, problems = audit_task_benchmark_entrypoints(tmp_path)
    assert counts == {} and len(problems) == 1
    assert "reviewed implementation" in problems[0]


@pytest.mark.parametrize("example,relative", [
    ("headkernel/kimi-k3__attn_residual_aggregate_hip", "ut/fresh_runner.py"),
    ("headkernel/kimi-k3__dense_bf16_gemm_cijk", "scripts/production_comparison.py"),
    ("headkernel/minimax-m3__gemm_afp4wfp4_kernel", "ut/runtime.py"),
    ("headkernel/minimax-m3__gemm_afp4wfp4_kernel", "ut/admission.py"),
    ("headkernel/minimax-m3__gemm_afp4wfp4_kernel", "ut/reference.py"),
    ("headkernel/minimax-m3__gemm_afp4wfp4_kernel", "ut/fixture_codec.py"),
    ("headkernel/minimax-m3__decode_score_kernel", "ut/workload_controls.py"),
    ("headkernel/minimax-m3__gqa_share_sparse_decode_kernel", "ut/minimax_data.py"),
])
def test_current_portable_protocol_helper_drift_requires_review(tmp_path, example, relative):
    task, _ = stage_protocol(tmp_path, example)
    helper = task / relative
    helper.write_text(helper.read_text() + "\n# implementation changed after review\n")
    counts, problems = audit_task_benchmark_entrypoints(tmp_path)
    assert counts == {} and len(problems) == 1
    assert "reviewed implementation" in problems[0]


def test_invalid_portable_case_manifest_is_rejected(tmp_path):
    task, _ = stage_protocol(tmp_path, EXAMPLES[1][0])
    path = task / "cases.json"
    manifest = json.loads(path.read_text())
    manifest["measurement"]["benchmark_iterations"] = 0
    path.write_text(json.dumps(manifest))
    counts, problems = audit_task_benchmark_entrypoints(tmp_path)
    assert counts == {} and len(problems) == 1
    assert "benchmark_iterations" in problems[0]


@pytest.mark.parametrize("command", ["compile", "performance && true", "performance\n- python3 scripts/task_runner.py performance"])
def test_known_runner_cannot_hide_wrong_or_repeated_command(tmp_path, command):
    task, _ = stage_protocol(tmp_path, EXAMPLES[1][0])
    (task / "config.yaml").write_text("performance_command:\n- python3 scripts/task_runner.py " + command + "\n")
    counts, problems = audit_task_benchmark_entrypoints(tmp_path)
    assert counts == {} and len(problems) == 1
    assert "performance phase exactly once" in problems[0]


def test_symlinked_protocol_helper_is_rejected(tmp_path):
    task, _ = stage_protocol(tmp_path, EXAMPLES[1][0])
    helper = task / "ut/fresh_runner.py"
    helper.rename(task / "ut/renamed.py")
    helper.symlink_to("renamed.py")
    counts, problems = audit_task_benchmark_entrypoints(tmp_path)
    assert counts == {} and len(problems) == 1
    assert "symlink" in problems[0]


def test_custom_protocol_still_rejects_committed_generated_helper(tmp_path):
    task, _ = stage_protocol(tmp_path, EXAMPLES[1][0])
    (task / "_aka_benchmark.py").write_text("# forbidden generated helper\n")
    counts, problems = audit_task_benchmark_entrypoints(tmp_path)
    assert counts == {"portable_case_contract": 1}
    assert len(problems) == 1 and "generated Python helper is committed" in problems[0]
