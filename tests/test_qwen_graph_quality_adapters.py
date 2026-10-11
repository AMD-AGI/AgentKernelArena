"""A supported graph report cannot be relabelled as another Qwen contract."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src import quality_measurements
from src.benchmark_quality import comparison_quality
from src.quality_measurements import load_quality_measurements
from test_qwen_rmsnorm_quality import emit, qwen


@pytest.mark.parametrize("adapter", [
    "qwen_dense_bf16_graph_v1",
    "qwen_mxfp4_moe_graph_v1",
    "qwen_recurrent_graph_v1",
    "qwen_paged_attention_graph_v1",
])
def test_other_qwen_profile_cannot_admit_rmsnorm_sources_or_reports(qwen, adapter):
    root, config, _, _ = qwen
    emit(qwen, [1.0] * 100)
    config["benchmark_quality"]["adapter"] = adapter
    with pytest.raises(ValueError, match="unsupported benchmark-quality adapter|target contract differs"):
        load_quality_measurements(root, config)


def test_raw_and_published_invocations_must_agree(qwen):
    root, config, _, _ = qwen
    report = emit(qwen, [1.0] * 100)
    path = root / "build/_bench_raw.json"
    raw = json.loads(path.read_text())
    raw["request"] = report["request"]
    path.write_text(json.dumps(raw))
    assert len(load_quality_measurements(root, config)) == 2
    raw["request"]["request_id"] = "older-invocation"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="raw/published invocation differs"):
        load_quality_measurements(root, config)


def test_additional_case_contract_is_bound_and_cannot_change_between_legs(qwen, monkeypatch):
    root, config, _, _ = qwen
    profile = quality_measurements._QWEN_GRAPH_PROFILES["qwen_rmsnorm_graph_v1"]
    monkeypatch.setitem(quality_measurements._QWEN_GRAPH_PROFILES, "qwen_rmsnorm_graph_v1",
                        {**profile, "case_contract": "ut/benchmark_contract.json"})
    def measured(tolerance):
        contract = root / "ut/benchmark_contract.json"
        contract.write_text(json.dumps({"tolerance": tolerance}))
        report = emit(qwen, [1.0] * 100)
        report["request"]["case_contract_sha256"] = hashlib.sha256(contract.read_bytes()).hexdigest()
        raw_path = root / "build/_bench_raw.json"
        raw = json.loads(raw_path.read_text())
        raw.update(source_sha256=report["request"]["source_sha256"],
                   case_contract_sha256=report["request"]["case_contract_sha256"])
        raw_path.write_text(json.dumps(raw))
        (root / "build/performance_report.json").write_text(json.dumps(report))
        return load_quality_measurements(root, config)
    before = measured(0.02)
    after = measured(0.2)
    with pytest.raises(ValueError, match="quality manifests differ"):
        comparison_quality(before, after)
    (root / "ut/benchmark_contract.json").write_text("{}")
    with pytest.raises(ValueError, match="case contract binding differs"):
        load_quality_measurements(root, config)


@pytest.mark.parametrize("suffix", ["dense_bf16_gemm_cluster", "fused_moe_2stage_mxfp4",
                                    "fused_recurrent_gated_delta_rule_decode",
                                    "paged_attention_decode", "gemma_fused_add_rmsnorm"])
def test_shipped_qwen_profiles_have_unique_keys_and_current_protected_validator(suffix):
    task = Path(__file__).resolve().parents[1] / "tasks/headkernel" / ("qwen3.8-2.4t__" + suffix)
    text = (task / "config.yaml").read_text()
    def unique(node):
        if isinstance(node, yaml.MappingNode):
            keys = [key.value for key, _ in node.value]
            assert len(keys) == len(set(keys))
            for _, value in node.value:
                unique(value)
        elif isinstance(node, yaml.SequenceNode):
            for value in node.value:
                unique(value)
    unique(yaml.compose(text))
    config = yaml.safe_load(text)
    profile = quality_measurements._QWEN_GRAPH_PROFILES[config["benchmark_quality"]["adapter"]]
    assert config["source_file_path"] == profile["sources"]
    assert config["target_kernel_functions"] == profile["targets"]
    validator = task / profile.get("validator", "scripts/task_runner.py")
    assert hashlib.sha256(validator.read_bytes()).hexdigest() == config["benchmark_quality"]["validator_sha256"]
