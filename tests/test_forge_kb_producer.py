"""Pure-Python coverage for Arena's forge-loop task metadata adapter."""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agents.forge.launch_agent import (
    _build_forge_command,
    _declared_editable_sources,
    _forge_max_hours,
    _infer_backend,
    _logical_operator,
    _normalize_logical_operator,
    _publication_status,
    _resolve_all_source_files,
    _resolve_fellow,
    _resolve_framework,
    _resolve_kernel_kind,
)
from agents.forge.drivers import arena_task_adapter


CK_TASK_NAMES = (
    "mi355x_vllm_ck_a8w8_blockscale_gemm",
    "mi355x_vllm_ck_cktile_moe_2stage",
    "mi355x_vllm_ck_moe_2stage",
)


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _load_ck_forge_driver(task_name: str):
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "tasks"
        / "image_kernel"
        / task_name
        / "scripts"
        / "forge_driver.py"
    )
    spec = importlib.util.spec_from_file_location(f"_{task_name}_driver_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_k3_forge_driver():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "tasks/image_kernel/mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3"
        / "scripts/forge_driver.py"
    )
    spec = importlib.util.spec_from_file_location("_k3_forge_driver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_unified_attention_task_runner():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "tasks/image_kernel/mi355x_vllm_triton_unified_attention"
        / "scripts/task_runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_unified_attention_task_runner_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_paged_attention_module(filename: str):
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "tasks/image_kernel/mi355x_vllm_triton_paged_attention_2d"
        / "scripts"
        / filename
    )
    module_name = f"_paged_attention_{path.stem}_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command(tmp_path: Path, **overrides) -> list[str]:
    values = {
        "forge_bin": "/usr/bin/kernel-agents",
        "kernel_file": tmp_path / "wrapper.py",
        "driver_dest": tmp_path / "forge_driver.py",
        "workspace": str(tmp_path),
        "experiments_dir": tmp_path / "forge_experiments",
        "result_json": tmp_path / "forge_experiments" / "forge_result.json",
        "agent_config": {
            "max_iters": 1000,
            "timeout_seconds": 7200,
        },
        "gpu_arch": "gfx950",
        "fellow": "triton-fellow",
        "task_type": "image_kernel",
        "source_files": [tmp_path / "wrapper.py", tmp_path / "kernel.py"],
        "target_functions": ["dispatch", "_device_kernel"],
        "logical_operator": "unified_attention",
        "framework": "aiter",
    }
    values.update(overrides)
    return _build_forge_command(**values)


def test_supplied_kernel_identity_fields_are_forwarded(tmp_path):
    argv = _command(tmp_path)

    assert _value(argv, "--operator-name") == "unified_attention"
    assert _value(argv, "--framework") == "aiter"
    assert _value(argv, "--target-functions") == "dispatch,_device_kernel"
    assert _value(argv, "--source-files").split(",") == [
        str(tmp_path / "wrapper.py"),
        str(tmp_path / "kernel.py"),
    ]
    assert "--shapes-json" not in argv
    assert "--workload-key" not in argv
    assert "--kernel-kind" not in argv
    assert "--program-md-file" not in argv
    assert "--resume" not in argv


def test_absent_kernel_identity_fields_are_omitted(tmp_path):
    argv = _command(
        tmp_path,
        logical_operator="",
        framework="",
    )
    assert "--operator-name" not in argv
    assert "--framework" not in argv
    assert "--shapes-json" not in argv
    assert "--workload-key" not in argv


def test_configured_backend_resolution_is_forwarded_without_fallback():
    assert _infer_backend({"task_type": "triton2triton"}) == "triton"
    assert _infer_backend({"task_type": "instruction2triton"}) == "triton"
    assert _infer_backend({"task_type": "flydsl2flydsl"}) == "flydsl"
    assert _infer_backend({"task_type": "torch2torch"}) == "torch"
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
    tilelang = {
        "task_type": "image_kernel",
        "repository_language": "tilelang",
        "kernel_identity": {"kernel_kind": "tilelang"},
    }
    assert _infer_backend(tilelang) == "tilelang"
    assert _resolve_fellow(tilelang, {}) == "tilelang-fellow"


def test_repository_backend_resolution_requires_explicit_language():
    with pytest.raises(ValueError, match="requires .*repository_language"):
        _infer_backend({"task_type": "image_kernel"})


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


def test_forge_budget_reserves_internal_shutdown_margin():
    assert _forge_max_hours({"timeout_seconds": 7200}) == 1.75
    assert _forge_max_hours({"timeout_seconds": 600}) == 1.0


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
    assert _infer_backend(config) == kernel_kind
    assert _resolve_fellow(config, {}) == f"{kernel_kind}-fellow"


def test_unified_attention_metadata():
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
    assert _logical_operator(config) == "unified_attention_with_output"
    assert _resolve_kernel_kind(config) == "triton"
    assert _resolve_framework(config) == "aiter"
    assert _declared_editable_sources(config) == [
        "ops/triton/_triton_kernels/attention/unified_attention.py",
        "ops/triton/attention/unified_attention.py",
    ]
    assert {
        "unified_attention",
        "select_3d_config",
        "use_2d_kernel",
        "kernel_unified_attention_2d",
        "kernel_unified_attention_3d",
        "reduce_segments",
    }.issubset(config["target_kernel_functions"])


def test_tilelang_backend_is_forwarded_to_forge_without_substitution(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "tasks"
        / "image_kernel"
        / "mi355x_vllm_tilelang_mhc_fused_post_pre"
        / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    fellow = _resolve_fellow(config, {})

    assert fellow == "tilelang-fellow"
    assert _value(_command(tmp_path, fellow=fellow), "--fellow") == fellow


def test_unified_attention_correctness_covers_2d_and_3d(monkeypatch):
    runner = _load_unified_attention_task_runner()
    case = runner.CASES[0]
    calls = []

    def fake_make_attention(
        supplied_case,
        correctness=False,
        *,
        ctx_len_override=None,
        expected_path=None,
    ):
        calls.append(
            {
                "case": supplied_case,
                "correctness": correctness,
                "ctx_len": ctx_len_override,
                "expected_path": expected_path,
            }
        )
        return calls[-1]

    monkeypatch.setattr(runner, "_make_attention", fake_make_attention)

    variants = runner._attention_correctness_inputs(case)

    assert [name for name, _ in variants] == ["2d", "3d"]
    assert calls == [
        {
            "case": case,
            "correctness": True,
            "ctx_len": min(case["params"]["ctx_len"], 128),
            "expected_path": "2d",
        },
        {
            "case": case,
            "correctness": True,
            "ctx_len": case["params"]["ctx_len"],
            "expected_path": "3d",
        },
    ]


@pytest.mark.parametrize("filename", ["task_runner.py", "standalone_driver.py"])
def test_paged_attention_correctness_uses_full_scored_dimensions(filename):
    runner = _load_paged_attention_module(filename)

    dimensions = [runner._scored_dimensions(case) for case in runner.CASES]
    compile_smoke = runner._compile_smoke_case(runner.CASES[0])

    assert dimensions == [(64, 1024), (64, 2048), (64, 3072)]
    assert [
        (ctx_len + case["params"]["block_size"] - 1)
        // case["params"]["block_size"]
        for case, (_, ctx_len) in zip(runner.CASES, dimensions)
    ] == [64, 128, 192]
    assert runner._scored_dimensions(compile_smoke) == (8, 256)
    assert runner._scored_dimensions(runner.CASES[0]) == (64, 1024)


def test_existing_mi355x_forge_drivers_reject_case_selectors():
    root = Path(__file__).resolve().parents[1]
    driver_paths = sorted(
        (root / "tasks" / "image_kernel").glob("mi355x_*/scripts/forge_driver.py")
    )

    assert len(driver_paths) == 13
    for driver_path in driver_paths:
        source = driver_path.read_text()
        assert '"--shape"' not in source
        assert '"--profile-case"' not in source
        assert "parse_known_args" not in source
        assert '"--profile-run"' in source
        assert "case_ms:" in source
        assert "mean_ms:" in source


@pytest.mark.parametrize("task_name", CK_TASK_NAMES)
def test_ck_profile_run_launches_only_the_target_operator(task_name, tmp_path):
    driver = _load_ck_forge_driver(task_name)
    prepared = []
    launches = []
    synchronizations = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: synchronizations.append(True))
    )
    cases = [{"id": "first"}, {"id": "profile"}]

    def make(case, correctness=False):
        prepared.append((case, correctness))
        return {"case": case}

    task_runner = SimpleNamespace(
        WORKSPACE=tmp_path,
        CASES=cases,
        _torch=lambda: torch,
        _make=make,
        _run=lambda inputs: launches.append(inputs),
    )

    assert driver._run_profile(task_runner) == 0
    assert prepared == [(cases[-1], False)]
    assert launches == [{"case": cases[-1]}] * 8
    assert len(synchronizations) == 2


def test_k3_profile_run_uses_cached_inputs_without_preparing(monkeypatch):
    driver = _load_k3_forge_driver()
    launches = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        driver,
        "_load_profile_inputs",
        lambda _task, _case: {"cached": True},
    )

    def reject_prepare(*_args, **_kwargs):
        pytest.fail("profile-run must not prepare quantized inputs under counters")

    task = SimpleNamespace(
        CASES=[
            {
                "id": "prefill",
                "params": {"token": 7211, "topk": 16, "model_dim": 3584},
            },
            {
                "id": "coverage",
                "correctness_only": True,
                "params": {"token": 8192, "topk": 16, "model_dim": 3584},
            },
        ],
        _prepare=reject_prepare,
        _run=lambda inputs: launches.append(inputs),
    )

    assert driver._run_profile(task) == 0
    assert launches == [{"cached": True}] * 6


def test_k3_benchmark_refreshes_only_profiled_scored_case(monkeypatch):
    driver = _load_k3_forge_driver()
    refreshed = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        driver,
        "_save_profile_inputs",
        lambda _task, case, _inputs: refreshed.append(case["id"]),
    )
    cases = [
        {
            "id": "decode",
            "params": {"token": 62, "topk": 16, "model_dim": 3584},
        },
        {
            "id": "prefill",
            "params": {"token": 7211, "topk": 16, "model_dim": 3584},
        },
        {
            "id": "coverage",
            "correctness_only": True,
            "params": {"token": 8192, "topk": 16, "model_dim": 3584},
        },
    ]
    task = SimpleNamespace(
        CASES=cases,
        _prepare=lambda case, correctness: {"case_id": case["id"]},
        _run=lambda _inputs: None,
        _benchmark_cuda_graph_or_events=lambda _fn, **_kwargs: (
            1.0,
            {"benchmark_method": "cuda_graph"},
        ),
    )

    assert driver._run_bench(task, warmup=1, iters=1) == 0
    assert refreshed == ["prefill"]


def test_adapter_rejects_incomplete_and_invalid_performance_cases():
    cases = [
        SimpleNamespace(test_case_id="a", execution_time_ms=1.0),
        SimpleNamespace(test_case_id="b", execution_time_ms=2.0),
    ]
    assert arena_task_adapter._complete_case_timings(cases, ["a", "b"]) == [
        ("a", 1.0),
        ("b", 2.0),
    ]

    with pytest.raises(ValueError, match="missing=\\['b'\\]"):
        arena_task_adapter._complete_case_timings(cases[:1], ["a", "b"])
    cases[1].execution_time_ms = float("nan")
    with pytest.raises(ValueError, match="invalid timing"):
        arena_task_adapter._complete_case_timings(cases, ["a", "b"])


def test_adapter_benchmark_emits_every_case_and_mean(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "kernel_identity": {
                    "workload": {"source": "session_cases.json"},
                },
                "performance_command": ["unused"],
            }
        )
    )
    (tmp_path / "session_cases.json").write_text(
        '{"cases":[{"id":"a"},{"id":"b"}]}'
    )
    measured = [
        SimpleNamespace(test_case_id="a", execution_time_ms=1.0),
        SimpleNamespace(test_case_id="b", execution_time_ms=3.0),
    ]
    monkeypatch.setattr(
        "src.performance.measure_performance",
        lambda *args, **kwargs: measured,
    )

    assert arena_task_adapter.do_bench(
        str(tmp_path),
        str(config_path),
        str(Path(__file__).resolve().parents[1]),
    ) == 0
    assert capsys.readouterr().out.splitlines() == [
        "case_ms: a 1.000000",
        "case_ms: b 3.000000",
        "mean_ms: 2.000000",
    ]
