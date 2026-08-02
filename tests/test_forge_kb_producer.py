"""Pure-Python coverage for the Forge KB producer contract."""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agents.forge.launch_agent import (
    _apply_run_kb_config,
    _build_forge_command,
    _canonical_workload_key,
    _check_producer_cli_capabilities,
    _declared_editable_sources,
    _forge_max_hours,
    _infer_backend,
    _kb_mode,
    _logical_operator,
    _normalize_logical_operator,
    _producer_preflight,
    _publication_status,
    _resolve_all_source_files,
    _resolve_framework,
    _resolve_kernel_kind,
    _resolve_shapes,
    _validate_producer_outcome,
)
from src.evaluator import write_task_result

launch_agent_module = importlib.import_module("agents.forge.launch_agent")


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_producer_argv_separates_logical_operator_and_concrete_targets(tmp_path):
    shapes = {
        "primary": {"N": 4096, "M": 64},
        "validation": [{"N": 4096, "M": 1}],
    }
    argv = _build_forge_command(
        forge_bin="/usr/bin/kernel-agents",
        kernel_file=tmp_path / "wrapper.py",
        driver_dest=tmp_path / "forge_driver.py",
        workspace=str(tmp_path),
        experiments_dir=tmp_path / "forge_experiments",
        result_json=tmp_path / "forge_experiments" / "forge_result.json",
        program_md=tmp_path / "forge_program.md",
        agent_config={
            "max_iters": 1000,
            "timeout_seconds": 7200,
            "knowledge_base": {
                "mode": "producer",
                "finalization_margin_seconds": 900,
            },
        },
        gpu_arch="gfx950",
        fellow="triton-fellow",
        task_type="image_kernel",
        source_files=[tmp_path / "wrapper.py", tmp_path / "kernel.py"],
        target_functions=["dispatch", "_device_kernel"],
        logical_operator="unified_attention",
        framework="aiter",
        shapes=shapes,
    )

    assert _value(argv, "--operator-name") == "unified_attention"
    assert _value(argv, "--target-functions") == "dispatch,_device_kernel"
    assert "--kernel-kind" not in argv
    assert _value(argv, "--source-files").split(",") == [
        str(tmp_path / "wrapper.py"),
        str(tmp_path / "kernel.py"),
    ]
    assert _value(argv, "--framework") == "aiter"
    assert json.loads(_value(argv, "--shapes-json")) == shapes
    assert _value(argv, "--workload-key") == _canonical_workload_key(shapes)
    assert _value(argv, "--workload-key").startswith("shape-v2-")
    assert float(_value(argv, "--max-hours")) == 1.75
    assert "--resume" not in argv


def test_compatibility_mode_does_not_require_producer_metadata():
    config = {"knowledge_base": {"mode": "compatibility"}}
    assert _kb_mode(config) == "compatibility"
    assert _logical_operator({}, producer=False) == ""
    assert _resolve_kernel_kind({}, "triton", producer=False) == ""
    assert _resolve_shapes({}, "/tmp/config.yaml", config, producer=False) is None


def test_compatibility_mode_preserves_explicit_agent_shapes_json():
    task_config = {
        "knowledge_base": {
            "workload": {
                "shapes": {
                    "primary": {"producer": True},
                }
            }
        }
    }
    legacy_shapes = {"primary": {"M": 32, "N": 64}}

    assert _resolve_shapes(
        task_config,
        "/tmp/config.yaml",
        {"shapes_json": json.dumps(legacy_shapes)},
        producer=False,
    ) == legacy_shapes


def test_producer_mode_requires_task_workload_even_with_legacy_shapes():
    with pytest.raises(ValueError, match="requires knowledge_base.workload"):
        _resolve_shapes(
            {},
            "/tmp/config.yaml",
            {"shapes_json": '{"primary":{"M":32}}'},
            producer=True,
        )


def test_run_config_can_enable_producer_mode_declaratively():
    defaults = {
        "knowledge_base": {
            "mode": "compatibility",
            "finalization_margin_seconds": 900,
        }
    }
    merged = _apply_run_kb_config(
        defaults,
        {"agent": {"knowledge_base": {"mode": "producer"}}},
    )
    assert _kb_mode(merged) == "producer"
    assert merged["knowledge_base"]["finalization_margin_seconds"] == 900
    assert defaults["knowledge_base"]["mode"] == "compatibility"


def test_direct_triton_and_flydsl_backend_resolution():
    assert _infer_backend({"task_type": "triton2triton"}) == "triton"
    assert _infer_backend({"task_type": "instruction2triton"}) == "triton"
    assert _infer_backend({"task_type": "flydsl2flydsl"}) == "flydsl"
    assert _infer_backend(
        {"task_type": "image_kernel", "repository_language": "flydsl"}
    ) == "flydsl"


def test_balanced_template_logical_operator_matches_hyperloom():
    raw = " aiter :: launch<ck::Tuple<int, float>>:: operator()<Nested<A<B>>> "
    assert _normalize_logical_operator(raw) == "aiter::launch::operator()"
    assert _logical_operator(
        {"knowledge_base": {"logical_operator": raw}},
        producer=True,
    ) == "aiter::launch::operator()"


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
        "knowledge_base": {"source_owner": "aiter"},
    }
    assert (
        _resolve_framework(
            config,
            "/workspace/vllm/attention.py",
            [Path("/workspace/aiter/ops/triton/unified_attention.py")],
        )
        == "aiter"
    )


def test_source_owner_follows_full_source_set_when_anchor_is_unknown():
    assert (
        _resolve_framework(
            {},
            "/workspace/wrapper.py",
            [Path("/workspace/aiter/ops/triton/kernel.py")],
        )
        == "aiter"
    )


def test_session_workload_is_structured_and_shape_derived(tmp_path):
    session = {
        "cases": [
            {"id": "small", "params": {"N": 1024, "M": 1}},
            {"id": "primary", "params": {"M": 64, "N": 1024}},
        ]
    }
    (tmp_path / "session_cases.json").write_text(json.dumps(session))
    task_config = {
        "knowledge_base": {
            "workload": {
                "source": "session_cases.json",
                "primary_case": "primary",
            }
        }
    }

    shapes = _resolve_shapes(
        task_config,
        str(tmp_path / "config.yaml"),
        {"knowledge_base": {"mode": "producer"}},
        producer=True,
    )
    assert shapes == {
        "primary": {"M": 64, "N": 1024},
        "validation": [{"N": 1024, "M": 1}, {"M": 64, "N": 1024}],
    }
    workload_key = _canonical_workload_key(shapes)
    assert workload_key.startswith("shape-v2-")
    assert workload_key == _canonical_workload_key(
        {"primary": {"CASE_ID": "case_7", "N": 1024, "M": 64}}
    )


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
        "knowledge_base": {"workload": {"source": "session_cases.json"}}
    }
    with pytest.raises(ValueError, match="primary_case"):
        _resolve_shapes(
            config,
            str(tmp_path / "config.yaml"),
            {"knowledge_base": {"mode": "producer"}},
            producer=True,
        )


def test_producer_preflight_requires_inherited_gbrain_credentials(monkeypatch):
    monkeypatch.setattr(launch_agent_module, "_check_producer_cli_capabilities", lambda _: None)
    with pytest.raises(RuntimeError, match="GBRAIN_TOKEN"):
        _producer_preflight(
            forge_bin="/usr/bin/kernel-agents",
            env={"GBRAIN_BASE_URL": "https://gbrain.example"},
            logical_operator="rms_norm",
            kernel_kind="triton",
            framework="vllm",
            shapes={"primary": {"M": 1}},
            fellow="triton-fellow",
            backend="triton",
        )


def test_producer_preflight_rejects_unknown_source_owner(monkeypatch):
    monkeypatch.setattr(launch_agent_module, "_check_producer_cli_capabilities", lambda _: None)
    with pytest.raises(RuntimeError, match="Unknown KB source_owner"):
        _producer_preflight(
            forge_bin="/usr/bin/kernel-agents",
            env={
                "GBRAIN_BASE_URL": "https://gbrain.example",
                "GBRAIN_TOKEN": "token",
            },
            logical_operator="rms_norm",
            kernel_kind="triton",
            framework="aiterr",
            shapes={"primary": {"M": 1}},
            fellow="triton-fellow",
            backend="triton",
        )


def test_producer_preflight_rejects_kernel_kind_fellow_mismatch(monkeypatch):
    monkeypatch.setattr(launch_agent_module, "_check_producer_cli_capabilities", lambda _: None)
    with pytest.raises(RuntimeError, match="kernel_kind/fellow/backend mismatch"):
        _producer_preflight(
            forge_bin="/usr/bin/kernel-agents",
            env={
                "GBRAIN_BASE_URL": "https://gbrain.example",
                "GBRAIN_TOKEN": "token",
            },
            logical_operator="rms_norm",
            kernel_kind="triton",
            framework="aiter",
            shapes={"primary": {"M": 1}},
            fellow="hip-fellow",
            backend="hip",
        )


def test_producer_preflight_rejects_unknown_kernel_kind(monkeypatch):
    monkeypatch.setattr(launch_agent_module, "_check_producer_cli_capabilities", lambda _: None)
    with pytest.raises(RuntimeError, match="Unknown KB kernel_kind"):
        _producer_preflight(
            forge_bin="/usr/bin/kernel-agents",
            env={
                "GBRAIN_BASE_URL": "https://gbrain.example",
                "GBRAIN_TOKEN": "token",
            },
            logical_operator="rms_norm",
            kernel_kind="trtion",
            framework="aiter",
            shapes={"primary": {"M": 1}},
            fellow="triton-fellow",
            backend="triton",
        )


def test_cli_capability_check_uses_executable_and_reports_missing(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="Usage: forge-loop [--operator-name]",
            stderr="",
        )

    monkeypatch.setattr(launch_agent_module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="missing CLI flags"):
        _check_producer_cli_capabilities("/custom/bin/kernel-agents")
    assert calls[0][0] == ["/custom/bin/kernel-agents", "forge-loop", "--help"]


def test_producer_contract_accepts_complete_current_metadata(monkeypatch):
    required_help = (
        "--framework --operator-name --result-json --shapes-json "
        "--source-files --target-functions --workload-key"
    )
    monkeypatch.setattr(
        launch_agent_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=required_help,
            stderr="",
        ),
    )

    _producer_preflight(
        forge_bin="/usr/bin/kernel-agents",
        env={
            "GBRAIN_BASE_URL": "https://gbrain.example",
            "GBRAIN_TOKEN": "token",
        },
        logical_operator="rms_norm",
        kernel_kind="triton",
        framework="aiter",
        shapes={"primary": {"M": 1}},
        fellow="triton-fellow",
        backend="triton",
    )


@pytest.mark.parametrize(
    ("payload", "published", "state", "source"),
    [
        (
            {
                "best_commit": "best123",
                "remote_publication": {
                    "status": "not_better_than_kb",
                    "pending_commit": "",
                    "last_attempted_commit": "best123",
                    "published_commit": "best123",
                },
                "kb_experience": {
                    "write": {
                        "written": False,
                        "reason": "not_better_than_kb",
                    },
                    "publication": {
                        "status": "not_better_than_kb",
                        "pending_commit": "",
                        "published_commit": "best123",
                    },
                },
            },
            True,
            "published",
            "remote_publication",
        ),
        (
            {
                "best_commit": "best456",
                "kb_experience": {
                    "publication": {
                        "status": "published",
                        "pending_commit": "",
                        "last_attempted_commit": "best456",
                        "published_commit": "best456",
                    }
                },
            },
            True,
            "published",
            "kb_experience.publication",
        ),
        (
            {
                "best_commit": "warm123",
                "remote_publication": {
                    "status": "warm_start_existing",
                    "published_commit": "warm123",
                },
            },
            True,
            "published",
            "remote_publication",
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
            "remote_publication",
        ),
        ({}, False, "schema_unsupported", None),
    ],
)
def test_publication_status_normalization(payload, published, state, source):
    status = _publication_status(payload, required=True)
    assert status["required"] is True
    assert status["published"] is published
    assert status["latest_best_published"] is published
    assert status["state"] == state
    assert status.get("source") == source


def test_durable_incremental_publish_survives_final_summary_noop():
    result = {
        "best_commit": "best123",
        "remote_publication": {
            "status": "not_better_than_kb",
            "pending_commit": "",
            "published_commit": "best123",
        },
        "kb_experience": {
            "write": {"written": False, "reason": "not_better_than_kb"},
            "publication": {
                "status": "not_better_than_kb",
                "pending_commit": "",
                "published_commit": "best123",
            },
        },
    }
    status = _publication_status(result, required=True)
    assert status["latest_best_published"] is True
    assert status["final_write"] == {
        "written": False,
        "reason": "not_better_than_kb",
    }
    _validate_producer_outcome(
        returncode=0,
        timed_out=False,
        forge_result=result,
        kb_status=status,
    )


def test_warm_start_existing_is_authoritative_without_pending_field():
    result = {
        "best_commit": "warm123",
        "remote_publication": {
            "status": "warm_start_existing",
            "published_commit": "warm123",
        },
    }
    status = _publication_status(result, required=True)
    assert status["authoritative"] is True
    assert status["latest_best_published"] is True
    assert status["publication_state"] == "warm_start_existing"
    _validate_producer_outcome(
        returncode=0,
        timed_out=False,
        forge_result=result,
        kb_status=status,
    )


@pytest.mark.parametrize(
    ("returncode", "timed_out", "result", "status", "message"),
    [
        (
            2,
            False,
            {"best_commit": "best"},
            {"authoritative": True, "latest_best_published": True},
            "exit code 2",
        ),
        (
            0,
            True,
            {"best_commit": "best"},
            {"authoritative": True, "latest_best_published": True},
            "timed out",
        ),
        (
            0,
            False,
            {"best_commit": "best"},
            {
                "authoritative": True,
                "latest_best_published": False,
                "state": "pending_retry",
                "reason": "publication_pending:best",
            },
            "did not publish",
        ),
    ],
)
def test_producer_outcome_rejects_process_and_publication_failures(
    returncode,
    timed_out,
    result,
    status,
    message,
):
    with pytest.raises(RuntimeError, match=message):
        _validate_producer_outcome(
            returncode=returncode,
            timed_out=timed_out,
            forge_result=result,
            kb_status=status,
        )


def test_forge_publication_status_is_written_with_arena_score(tmp_path):
    forge_result = {
        "exit_code": 0,
        "timed_out": False,
        "kb": {
            "required": True,
            "published": True,
            "state": "published",
            "reason": "",
        },
    }
    write_task_result(
        tmp_path,
        {
            "pass_compilation": True,
            "pass_correctness": True,
            "best_optimized_execution_time": 1.0,
            "average_speedup": 1.2,
            "forge_result": forge_result,
        },
        [],
        "producer-task",
        "forge",
        create_plots=False,
    )
    task_result = yaml.safe_load((tmp_path / "task_result.yaml").read_text())
    assert task_result["forge_result"] == forge_result


def test_producer_budget_reserves_finalization_margin():
    assert _forge_max_hours(
        {
            "timeout_seconds": 7200,
            "knowledge_base": {
                "mode": "producer",
                "finalization_margin_seconds": 900,
            },
        }
    ) == 1.75
    with pytest.raises(ValueError, match="at least one hour"):
        _forge_max_hours(
            {
                "timeout_seconds": 600,
                "knowledge_base": {
                    "mode": "producer",
                    "finalization_margin_seconds": 900,
                },
            }
        )


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
def test_all_mi355x_tasks_declare_producer_metadata(
    task_name,
    logical_operator,
    kernel_kind,
    source_owner,
):
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root / "tasks" / "image_kernel" / task_name / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    kb = config["knowledge_base"]

    assert kb["logical_operator"] == logical_operator
    assert kb["kernel_kind"] == kernel_kind
    assert kb["source_owner"] == source_owner
    assert _logical_operator(config, producer=True) == logical_operator
    assert _resolve_kernel_kind(config, kernel_kind, producer=True) == kernel_kind
    assert _resolve_framework(config, "/workspace/unknown.py") == source_owner
    shapes = _resolve_shapes(
        config,
        str(config_path),
        {"knowledge_base": {"mode": "producer"}},
        producer=True,
    )
    assert shapes["primary"]
    assert shapes["validation"]


def test_unified_attention_task_is_producer_ready():
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "tasks"
        / "image_kernel"
        / "mi355x_vllm_triton_unified_attention"
        / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert _infer_backend(config) == "triton"
    assert _logical_operator(config, producer=True) == "unified_attention_with_output"
    assert _resolve_kernel_kind(config, "triton", producer=True) == "triton"
    assert _resolve_framework(config, "/workspace/wrapper.py") == "aiter"
    assert _declared_editable_sources(config) == [
        "ops/triton/_triton_kernels/attention/unified_attention.py",
    ]
    assert config["target_kernel_functions"] == [
        "kernel_unified_attention_2d",
        "kernel_unified_attention_3d",
        "reduce_segments",
    ]
    assert _resolve_shapes(
        config,
        str(config_path),
        {"knowledge_base": {"mode": "compatibility"}},
        producer=False,
    ) is None

    shapes = _resolve_shapes(
        config,
        str(config_path),
        {"knowledge_base": {"mode": "producer"}},
        producer=True,
    )
    selector_schema = config["knowledge_base"]["workload"]["selector_schema"]
    assert selector_schema == {
        "name": "hyperloom-v1",
        "fields": {
            "q_tokens": "QTOKENS",
            "num_q_heads": "QHEADS",
            "num_kv_heads": "KVHEADS",
            "head_size": "HEADSIZE",
        },
    }
    assert shapes["primary"]["CASE_ID"] == "minimax-k004"
    assert set(shapes) == {"primary", "minimal", "validation"}
    assert set(shapes["primary"]) == {
        "CASE_ID",
        "QTOKENS",
        "QHEADS",
        "KVHEADS",
        "HEADSIZE",
    }
    session_cases = json.loads((config_path.parent / "session_cases.json").read_text())
    assert len(shapes["validation"]) == len(session_cases["cases"])
    assert all(
        set(selector) == {
            "CASE_ID",
            "QTOKENS",
            "QHEADS",
            "KVHEADS",
            "HEADSIZE",
        }
        for selector in shapes["validation"]
    )
    primary_params = session_cases["cases"][0]["params"]
    assert primary_params["q_tokens"] == 64
    assert primary_params["ctx_len"] == 1024
    assert primary_params["q_dtype"] == "bf16"
    assert _canonical_workload_key(shapes).startswith("shape-v2-")

    source = Path(
        "/workspace/aiter/ops/triton/_triton_kernels/attention/"
        "unified_attention.py"
    )
    argv = _build_forge_command(
        forge_bin="/usr/bin/kernel-agents",
        kernel_file=source,
        driver_dest=Path("/workspace/forge_driver.py"),
        workspace="/workspace",
        experiments_dir=Path("/workspace/forge_experiments"),
        result_json=Path("/workspace/forge_experiments/forge_result.json"),
        program_md=Path("/workspace/forge_program.md"),
        agent_config={
            "timeout_seconds": 7200,
            "knowledge_base": {
                "mode": "producer",
                "finalization_margin_seconds": 900,
            },
        },
        gpu_arch="gfx950",
        fellow="triton-fellow",
        task_type="image_kernel",
        source_files=[source],
        target_functions=config["target_kernel_functions"],
        logical_operator="unified_attention_with_output",
        framework="aiter",
        shapes=shapes,
    )
    assert _value(argv, "--source-files") == str(source)
    assert _value(argv, "--target-functions") == ",".join(
        config["target_kernel_functions"]
    )
    assert json.loads(_value(argv, "--shapes-json")) == shapes

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
