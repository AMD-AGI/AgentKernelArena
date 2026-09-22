"""Pure-Python coverage for Arena's forge-loop task metadata adapter."""
from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

# The module object, so a test can monkeypatch the globals the functions under
# test resolve against. They live in common.py and are only re-exported through
# the launcher, so patching the launcher's namespace would have no effect.
import agents.forge.common  # noqa: F401
from agents.forge.drivers import arena_task_adapter
from agents.forge import adapter
from src.task_spec import load_task_spec
forge_common = sys.modules["agents.forge.common"]

from agents.forge.common import (
    _capture_forge_edit_baseline,
    _verify_forge_edit_scope,
)
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
    _resolve_gpu_type,
    _resolve_kernel_backend,
)


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


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
        "gpu_type": "mi355x",
        "kernel_backend": "triton",
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

    assert _value(argv, "--gpu-target") == "gfx950"
    assert _value(argv, "--gpu-type") == "mi355x"
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
    # Inference reports what the task declares. Reconciling that against what
    # KernelForge actually serves belongs to _resolve_kernel_backend, below.
    assert _infer_backend(tilelang) == "tilelang"
    assert _resolve_fellow(tilelang, {}) == "tilelang-fellow"


def _backend_registry(monkeypatch, names):
    monkeypatch.setattr(
        forge_common, "_installed_kernel_backends", lambda: names
    )


def test_an_unserved_backend_fails_instead_of_becoming_flydsl(monkeypatch):
    """The whole point of the translation: upstream would not have complained.

    KernelForge maps an unknown --kernel-backend onto flydsl and says nothing, so
    a typo or an upstream rename produces a run that starts, finishes, and
    reports a speedup obtained under the wrong expertise prompt. Arena has the
    registry in-process and can refuse before a GPU-day is spent.
    """
    _backend_registry(monkeypatch, {"triton", "flydsl", "hip", "ck"})
    with pytest.raises(ValueError, match="does not serve the 'trtion' backend"):
        _resolve_kernel_backend("trtion-fellow", logging.getLogger(__name__))


def test_a_served_backend_passes_through_with_the_suffix_stripped(monkeypatch):
    _backend_registry(monkeypatch, {"triton", "flydsl", "hip", "ck"})
    logger = logging.getLogger(__name__)
    assert _resolve_kernel_backend("triton-fellow", logger) == "triton"
    assert _resolve_kernel_backend("ck-fellow", logger) == "ck"


def test_legacy_backend_helper_also_rejects_tilelang_substitution(monkeypatch):
    _backend_registry(monkeypatch, {"triton", "flydsl", "hip", "ck"})
    with pytest.raises(ValueError, match="does not serve the 'tilelang' backend"):
        _resolve_kernel_backend("tilelang-fellow", logging.getLogger(__name__))


def test_an_unreadable_registry_does_not_claim_backend_support(monkeypatch):
    _backend_registry(monkeypatch, None)
    with pytest.raises(RuntimeError, match="Cannot verify KernelForge's backend registry"):
        _resolve_kernel_backend("triton-fellow", logging.getLogger(__name__))


def test_the_registry_is_read_from_kernelforge_not_copied_here(monkeypatch):
    """A hardcoded list would drift silently, which is the bug being fixed.

    Both layouts are probed: Hyperloom's src/kernelforge and the pre-merge
    standalone src/kernel_agents.
    """
    modules = {
        "kernelforge.kernel_backends.constants": SimpleNamespace(
            KERNEL_BACKENDS=["Triton", "flydsl"]
        ),
        "kernel_agents.fellows.constants": SimpleNamespace(
            FELLOW_BACKENDS=["hip", "intellikit"]
        ),
    }

    def fake_import(name):
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert forge_common._installed_kernel_backends() == {"triton", "flydsl"}

    modules.pop("kernelforge.kernel_backends.constants")
    assert forge_common._installed_kernel_backends() == {"hip", "intellikit"}

    modules.clear()
    assert forge_common._installed_kernel_backends() is None


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


def _init_scope_test_repo(tmp_path: Path) -> str:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "forge-test@local"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "forge-test"], cwd=tmp_path, check=True
    )
    (tmp_path / ".gitignore").write_text("build/\nforge_experiments/\n")
    (tmp_path / "kernel.py").write_text("def kernel(): return 0\n")
    (tmp_path / "helper.py").write_text("def helper(): return 0\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True, capture_output=True
    )
    return _capture_forge_edit_baseline(str(tmp_path))


def test_forge_edit_scope_allows_declared_source_change(tmp_path):
    baseline = _init_scope_test_repo(tmp_path)
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel(): return 1\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "allowed edit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    _verify_forge_edit_scope(str(tmp_path), baseline, [kernel])


def test_forge_edit_scope_allows_ignored_runtime_artifacts(tmp_path):
    baseline = _init_scope_test_repo(tmp_path)
    kernel = tmp_path / "kernel.py"
    build = tmp_path / "build"
    build.mkdir()
    (build / "kernel.hsaco").write_bytes(b"runtime artifact")

    _verify_forge_edit_scope(str(tmp_path), baseline, [kernel])


def test_forge_edit_scope_discards_undeclared_scratch_directory(tmp_path):
    # The loop gives each lane its own git workspace, and git reports an embedded
    # repository as one opaque directory entry rather than its files. unlink raises
    # on that, which is what cost two rewrite runs their score.
    baseline = _init_scope_test_repo(tmp_path)
    kernel = tmp_path / "kernel.py"
    lane = tmp_path / "forge-lanes-abc123" / "1"
    lane.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "."], cwd=lane, check=True)
    (lane / "candidate.py").write_text("def lane(): return 1\n")

    violations = _verify_forge_edit_scope(str(tmp_path), baseline, [kernel])

    assert not lane.exists()
    assert violations == []


def test_forge_edit_scope_discards_undeclared_untracked_file(tmp_path):
    baseline = _init_scope_test_repo(tmp_path)
    kernel = tmp_path / "kernel.py"
    scratch = tmp_path / "new_helper.py"
    scratch.write_text("def bypass(): return 1\n")

    _verify_forge_edit_scope(str(tmp_path), baseline, [kernel])

    assert not scratch.exists()


@pytest.mark.parametrize("change_kind", ["tracked", "rename"])
def test_forge_edit_scope_reports_undeclared_changes(tmp_path, change_kind):
    baseline = _init_scope_test_repo(tmp_path)
    kernel = tmp_path / "kernel.py"
    helper = tmp_path / "helper.py"
    if change_kind == "tracked":
        helper.write_text("def helper(): return 1\n")
        subprocess.run(["git", "add", "helper.py"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "undeclared edit"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    else:
        helper.rename(tmp_path / "renamed_helper.py")

    # Named for the caller to carry into the report, not raised: a whole campaign
    # is not worth discarding over a verdict the agent could not see coming.
    violations = _verify_forge_edit_scope(str(tmp_path), baseline, [kernel])

    assert "helper.py" in violations


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


def test_gpu_type_uses_normalized_arena_hardware_model():
    assert _resolve_gpu_type({"target_gpu_model": "MI355X"}) == "mi355x"
    assert _resolve_gpu_type({"target_gpu_model": "mi300"}) == "mi300x"
    assert _resolve_gpu_type({"target_gpu_model": "MI300X"}) == "mi300x"
    assert _resolve_gpu_type({"target_gpu_model": "mi325"}) == "mi325x"
    assert _resolve_gpu_type({"target_gpu_model": "MI325X"}) == "mi325x"
    with pytest.raises(ValueError, match="target_gpu_model"):
        _resolve_gpu_type({"target_gpu_model": ""})


def _v2_command(tmp_path, spec, *, workflow="optimize"):
    """Exercise actual adapter argv construction without image/GPU execution."""
    engine = tmp_path / "engine"
    engine.mkdir()
    for scope in spec.candidate.editable:
        assert scope.scope != "tree", "This image-task fixture expects declared source files"
        path = engine / scope.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    context = SimpleNamespace(spec=spec, workspace=engine)
    anchor = spec.candidate.entrypoints[0].file if spec.candidate.entrypoints else spec.candidate.editable[0].path
    plan = dict(workflow=workflow, engine_root=str(engine), anchor=anchor,
                deadline_unix=time.time() + 3600, result=str(tmp_path / "result.json"),
                baseline=str(tmp_path / "baseline.json"), program=str(engine / "arena_program.md"))
    return adapter.build_command(plan, context, adapter._config({}), gpu_arch="gfx950", gpu_type="mi355x")


@pytest.mark.parametrize(
    ("task_name", "logical_operator", "language", "source_owner"),
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
        ("mi355x_vllm_ck_moe_2stage", "ck_moe_2stage", "hip", "aiter"),
        (
            "mi355x_vllm_ck_cktile_moe_2stage",
            "cktile_moe_2stage",
            "hip",
            "aiter",
        ),
        (
            "mi355x_vllm_ck_a8w8_blockscale_gemm",
            "gemm_a8w8_blockscale_ck",
            "hip",
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
            "paged_attention_2d",
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
    tmp_path,
    task_name,
    logical_operator,
    language,
    source_owner,
):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "tasks" / "image_kernel" / task_name / "config.yaml"
    spec = load_task_spec(config_path, task_id=f"image_kernel/{task_name}")
    config = spec.to_mapping()
    identity = config["kernel_identity"]

    assert identity["logical_operator"] == logical_operator
    assert spec.candidate.language == language
    assert identity["source_owner"] == source_owner
    assert _logical_operator(config) == logical_operator
    assert _resolve_framework(config) == source_owner
    assert _infer_backend(config) == language
    argv = _v2_command(tmp_path, spec)
    assert _value(argv, "--operator-name") == logical_operator
    assert _value(argv, "--framework") == source_owner
    assert _value(argv, "--kernel-backend") == language
    assert "--shapes-json" not in argv
    assert "--workload-key" not in argv
    assert "--kernel-kind" not in argv
    # This is command serialization; runtime capability rejection is tested
    # separately. Serializing TileLang must never silently substitute FlyDSL.


def test_unified_attention_metadata():
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "tasks"
        / "image_kernel"
        / "mi355x_vllm_triton_unified_attention"
        / "config.yaml"
    )
    spec = load_task_spec(config_path, task_id="image_kernel/mi355x_vllm_triton_unified_attention")
    config = spec.to_mapping()
    assert _infer_backend(config) == "triton"
    assert _logical_operator(config) == "unified_attention_with_output"
    assert spec.candidate.language == "triton"
    assert _resolve_framework(config) == "aiter"
    assert [scope.path for scope in spec.candidate.editable] == [
        "aiter/ops/triton/_triton_kernels/attention/unified_attention.py",
        "aiter/ops/triton/attention/unified_attention.py",
    ]
    assert {
        "unified_attention",
        "select_3d_config",
        "use_2d_kernel",
        "kernel_unified_attention_2d",
        "kernel_unified_attention_3d",
        "reduce_segments",
    } == {entry.symbol for entry in spec.candidate.entrypoints}
    assert config["evaluation"]["workloads"] == "workloads.json"
    assert (config_path.parent / config["evaluation"]["workloads"]).is_file()


def test_the_real_tilelang_task_is_explicitly_unsupported_by_pinned_forge(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "tasks"
        / "image_kernel"
        / "mi355x_vllm_tilelang_mhc_fused_post_pre"
        / "config.yaml"
    )
    spec = load_task_spec(config_path, task_id="image_kernel/mi355x_vllm_tilelang_mhc_fused_post_pre")
    assert spec.candidate.language == "tilelang"
    with pytest.raises(adapter.ForgeRunError, match="has no tilelang backend"):
        adapter.require_supported_backend(spec, {"backends": ["hip", "triton", "flydsl", "ck"]})
    argv = _v2_command(tmp_path, spec)
    assert _value(argv, "--kernel-backend") == "tilelang"
    assert "--fellow" not in argv
    assert "--max-iters" not in argv


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
            "correctness": False,
            "ctx_len": min(case["params"]["ctx_len"], 128),
            "expected_path": "2d",
        },
        {
            "case": case,
            "correctness": False,
            "ctx_len": case["params"]["ctx_len"],
            "expected_path": "3d",
        },
    ]


def test_paged_attention_correctness_uses_full_scored_dimensions():
    runner = _load_paged_attention_module("task_runner.py")

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
