"""Pure-Python coverage for Arena's forge-loop task metadata adapter."""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agents.forge.launch_agent import (
    _build_forge_command,
    _canonical_workload_key,
    _declared_editable_sources,
    _forge_max_hours,
    _infer_backend,
    _logical_operator,
    _normalize_logical_operator,
    _publication_status,
    _resolve_all_source_files,
    _resolve_framework,
    _resolve_kernel_kind,
    _resolve_shapes,
)


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _command(tmp_path: Path, **overrides) -> list[str]:
    values = {
        "forge_bin": "/usr/bin/kernel-agents",
        "kernel_file": tmp_path / "wrapper.py",
        "driver_dest": tmp_path / "forge_driver.py",
        "workspace": str(tmp_path),
        "experiments_dir": tmp_path / "forge_experiments",
        "result_json": tmp_path / "forge_experiments" / "forge_result.json",
        "program_md": tmp_path / "forge_program.md",
        "agent_config": {
            "max_iters": 1000,
            "timeout_seconds": 7200,
            "finalization_margin_seconds": 900,
        },
        "gpu_arch": "gfx950",
        "fellow": "triton-fellow",
        "task_type": "image_kernel",
        "source_files": [tmp_path / "wrapper.py", tmp_path / "kernel.py"],
        "target_functions": ["dispatch", "_device_kernel"],
        "logical_operator": "unified_attention",
        "framework": "aiter",
        "shapes": {
            "primary": {"N": 4096, "M": 64},
            "validation": [{"N": 4096, "M": 1}],
        },
    }
    values.update(overrides)
    return _build_forge_command(**values)


def test_supplied_kernel_identity_fields_are_forwarded(tmp_path):
    argv = _command(tmp_path)
    shapes = json.loads(_value(argv, "--shapes-json"))

    assert _value(argv, "--operator-name") == "unified_attention"
    assert _value(argv, "--framework") == "aiter"
    assert _value(argv, "--target-functions") == "dispatch,_device_kernel"
    assert _value(argv, "--source-files").split(",") == [
        str(tmp_path / "wrapper.py"),
        str(tmp_path / "kernel.py"),
    ]
    assert _value(argv, "--workload-key") == _canonical_workload_key(shapes)
    assert "--kernel-kind" not in argv
    assert "--resume" not in argv


def test_absent_kernel_identity_fields_are_omitted(tmp_path):
    argv = _command(
        tmp_path,
        logical_operator="",
        framework="",
        shapes=None,
    )
    assert "--operator-name" not in argv
    assert "--framework" not in argv
    assert "--shapes-json" not in argv
    assert "--workload-key" not in argv


def test_direct_triton_and_flydsl_backend_resolution():
    assert _infer_backend({"task_type": "triton2triton"}) == "triton"
    assert _infer_backend({"task_type": "instruction2triton"}) == "triton"
    assert _infer_backend({"task_type": "flydsl2flydsl"}) == "flydsl"
    assert _infer_backend(
        {"task_type": "image_kernel", "repository_language": "flydsl"}
    ) == "flydsl"
    assert _infer_backend(
        {
            "task_type": "image_kernel",
            "repository_language": "hip",
            "kernel_identity": {"kernel_kind": "ck"},
        }
    ) == "ck"


def test_balanced_template_logical_operator_matches_hyperloom():
    raw = " aiter :: launch<ck::Tuple<int, float>>:: operator()<Nested<A<B>>> "
    assert _normalize_logical_operator(raw) == "aiter::launch::operator()"
    assert _logical_operator(
        {"kernel_identity": {"logical_operator": raw}},
    ) == "aiter::launch::operator()"
    assert _logical_operator({}) == ""


def test_editable_sources_extend_complete_source_allowlist(tmp_path):
    kernel = tmp_path / "kernel.py"
    helper = tmp_path / "helper.py"
    kernel.write_text("def kernel():\n    pass\n")
    helper.write_text("def helper():\n    pass\n")
    config = {
        "source_file_path": ["kernel.py"],
        "editable_sources": ["helper.py", "kernel.py"],
    }
    declared = _declared_editable_sources(config)
    resolved = _resolve_all_source_files(
        str(tmp_path),
        declared,
        config,
        logging.getLogger(__name__),
        strict=True,
    )
    assert declared == ["kernel.py", "helper.py"]
    assert resolved == [kernel.resolve(), helper.resolve()]


def test_explicit_source_owner_wins_for_wrapper_anchor():
    config = {
        "image_repo_path": "/workspace/vllm/model_executor/attention.py",
        "kernel_identity": {"source_owner": "aiter"},
    }
    assert _resolve_framework(config) == "aiter"


def test_session_workload_is_structured_and_shape_derived(tmp_path):
    session = {
        "cases": [
            {"id": "small", "params": {"N": 1024, "M": 1}},
            {"id": "primary", "params": {"M": 64, "N": 1024}},
        ]
    }
    (tmp_path / "session_cases.json").write_text(json.dumps(session))
    task_config = {
        "kernel_identity": {
            "workload": {
                "source": "session_cases.json",
                "primary_case": "primary",
            }
        }
    }

    shapes = _resolve_shapes(
        task_config,
        str(tmp_path / "config.yaml"),
        {},
    )
    assert shapes == {
        "primary": {"M": 64, "N": 1024},
        "validation": [{"N": 1024, "M": 1}, {"M": 64, "N": 1024}],
    }
    assert _canonical_workload_key(shapes).startswith("shape-v2-")


def test_absent_workload_uses_optional_legacy_shapes():
    assert _resolve_shapes({}, "/tmp/config.yaml", {}) is None
    assert _resolve_shapes(
        {},
        "/tmp/config.yaml",
        {"shapes_json": '{"primary":{"M":32,"N":64}}'},
    ) == {"primary": {"M": 32, "N": 64}}


def test_multi_case_workload_requires_explicit_primary(tmp_path):
    (tmp_path / "session_cases.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"id": "a", "params": {"M": 1}},
                    {"id": "b", "params": {"M": 2}},
                ]
            }
        )
    )
    config = {
        "kernel_identity": {"workload": {"source": "session_cases.json"}}
    }
    with pytest.raises(ValueError, match="primary_case"):
        _resolve_shapes(config, str(tmp_path / "config.yaml"), {})


@pytest.mark.parametrize(
    ("payload", "published", "state"),
    [
        (
            {
                "best_commit": "best123",
                "remote_publication": {
                    "status": "published",
                    "pending_commit": "",
                    "published_commit": "best123",
                },
            },
            True,
            "published",
        ),
        (
            {
                "best_commit": "best789",
                "remote_publication": {
                    "status": "pending_retry",
                    "pending_commit": "best789",
                    "published_commit": "older",
                },
            },
            False,
            "pending_retry",
        ),
        ({}, False, "schema_unsupported"),
    ],
)
def test_publication_status_is_diagnostic(payload, published, state):
    status = _publication_status(payload)
    assert status["published"] is published
    assert status["state"] == state
    assert "required" not in status


def test_forge_budget_reserves_finalization_margin():
    assert _forge_max_hours(
        {
            "timeout_seconds": 7200,
            "finalization_margin_seconds": 900,
        }
    ) == 1.75
    assert _forge_max_hours(
        {
            "timeout_seconds": 600,
            "finalization_margin_seconds": 900,
        }
    ) == 1.0


@pytest.mark.parametrize(
    ("task_name", "logical_operator", "kernel_kind", "source_owner"),
    [
        (
            "mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3",
            "aiter_mxfp4_moe_2stage",
            "flydsl",
            "aiter",
        ),
        (
            "mi355x_vllm_triton_unified_attention",
            "unified_attention_with_output",
            "triton",
            "aiter",
        ),
        ("mi355x_vllm_ck_moe_2stage", "ck_moe_2stage", "ck", "aiter"),
        (
            "mi355x_vllm_ck_cktile_moe_2stage",
            "cktile_moe_2stage",
            "ck",
            "aiter",
        ),
        (
            "mi355x_vllm_ck_a8w8_blockscale_gemm",
            "gemm_a8w8_blockscale_ck",
            "ck",
            "aiter",
        ),
        (
            "mi355x_vllm_triton_kda_linear_attn_kimi_k3",
            "kda_linear_attn",
            "triton",
            "vllm",
        ),
        (
            "mi355x_vllm_triton_sparse_attn_prefill_ragged",
            "sparse_attn_prefill_ragged",
            "triton",
            "vllm",
        ),
        (
            "mi355x_vllm_triton_paged_attention_2d",
            "unified_attention_with_output",
            "triton",
            "vllm",
        ),
        (
            "mi355x_vllm_triton_fused_moe_gptq_awq",
            "fused_moe_gptq_awq",
            "triton",
            "vllm",
        ),
        (
            "mi355x_vllm_tilelang_mhc_fused_post_pre",
            "mhc_fused_post_pre",
            "tilelang",
            "vllm",
        ),
        (
            "mi355x_vllm_hip_dynamic_per_tensor_quant",
            "dynamic_per_tensor_quant",
            "hip",
            "aiter",
        ),
        (
            "mi355x_sglang_triton_mxfp8_linear",
            "mxfp8_linear",
            "triton",
            "sglang",
        ),
        (
            "mi355x_sglang_triton_mxfp8_grouped_gemm",
            "mxfp8_grouped_gemm",
            "triton",
            "sglang",
        ),
    ],
)
def test_all_mi355x_tasks_declare_kernel_identity(
    task_name,
    logical_operator,
    kernel_kind,
    source_owner,
):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "tasks" / "image_kernel" / task_name / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    identity = config["kernel_identity"]

    assert identity["logical_operator"] == logical_operator
    assert identity["kernel_kind"] == kernel_kind
    assert identity["source_owner"] == source_owner
    assert _logical_operator(config) == logical_operator
    assert _resolve_kernel_kind(config) == kernel_kind
    assert _resolve_framework(config) == source_owner
    shapes = _resolve_shapes(config, str(config_path), {})
    assert shapes["primary"]
    assert shapes["validation"]


def test_unified_attention_metadata_and_driver_contract():
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "tasks"
        / "image_kernel"
        / "mi355x_vllm_triton_unified_attention"
        / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    shapes = _resolve_shapes(config, str(config_path), {})

    assert _infer_backend(config) == "triton"
    assert _logical_operator(config) == "unified_attention_with_output"
    assert _resolve_kernel_kind(config) == "triton"
    assert _resolve_framework(config) == "aiter"
    assert shapes["primary"]["CASE_ID"] == "minimax-k004"

    driver_path = config_path.parent / "scripts" / "forge_driver.py"
    spec = importlib.util.spec_from_file_location(
        "_test_unified_forge_driver",
        driver_path,
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    task_runner = SimpleNamespace(
        CASES=[
            {"id": "minimax-k004"},
            {"id": "gemma-k002"},
        ]
    )
    assert driver._select_shape_case(task_runner, "default") == ""
    assert task_runner.CASES == [
        {"id": "minimax-k004"},
        {"id": "gemma-k002"},
    ]
    selected = driver._select_shape_case(
        task_runner,
        "CASE_ID=minimax-k004,QTOKENS=64,HEADSIZE=128",
    )
    assert selected == "minimax-k004"
    assert task_runner.CASES == [{"id": "minimax-k004"}]
