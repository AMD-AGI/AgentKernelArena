"""Every schema-v3 task and run target language has an explicit Forge route (CPU, no GPU claims)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.forge.adapter import (INITIALIZATION_LANGUAGES, TARGET_LANGUAGES, ForgeRunError,
                                  choose_workflow, engine_backend, require_editable_assembly,
                                  require_supported_backend)
from agents.forge.task_context import require_kernel_task
from src.task_contract import TARGET_LANGUAGES as RUN_TARGET_LANGUAGES
from src.task_spec import TaskSpec, load_task_spec

ROOT = Path(__file__).resolve().parents[1]
# CPU fixture for the KernelForge registry and rewrite protocol; production
# launch uses the values the source-pinned probe reads from the installed engine.
CAPABILITIES = {"backends": ["aiter", "assembly", "ck", "flydsl", "fusion", "gluon", "hip", "hipblaslt", "triton"],
                "rewrite_source_languages": ["cpp", "cuda", "hip", "triton"]}


def _context(spec):
    contract = require_kernel_task(spec)
    return SimpleNamespace(spec=spec, target_language=contract["target_language"],
                           baseline_language=contract["baseline_language"])


def test_forge_targets_are_the_run_target_languages():
    assert set(TARGET_LANGUAGES) == set(RUN_TARGET_LANGUAGES)
    # FlyDSL is initialized by the rewrite PORT; KernelForge has no assembly initializer.
    assert set(INITIALIZATION_LANGUAGES) == set(TARGET_LANGUAGES) - {"flydsl", "asm"}


def test_asm_runs_on_kernelforges_assembly_backend():
    assert engine_backend("asm") == "assembly"
    assert engine_backend("triton") == "triton"
    path = ROOT / "tasks/triton/elementwise/add/fp32_1d/task.yaml"
    spec = load_task_spec(path, task_id="triton/elementwise/add/fp32_1d", target_language="asm")
    assert require_supported_backend(spec, CAPABILITIES) == "asm"
    with pytest.raises(ForgeRunError, match="has no assembly backend for asm"):
        require_supported_backend(spec, {"backends": ["triton"]})


def test_forge_optimizes_only_asm_candidates_with_assembly_sources(tmp_path):
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "kernel.py").write_text("import aka_asm\n")
    (solution / "vendored.co").write_bytes(b"\x7fELF")
    context = SimpleNamespace(workspace=tmp_path)
    with pytest.raises(ForgeRunError, match="optimizes existing .s sources"):
        require_editable_assembly(context)
    (solution / "kernel.s").write_text("s_endpgm\n")
    require_editable_assembly(context)


@pytest.mark.parametrize("language", RUN_TARGET_LANGUAGES)
def test_every_v3_task_has_an_explicit_forge_route(language):
    paths = sorted((ROOT / "tasks").rglob("task.yaml"))
    assert paths, "No schema-v3 tasks found"
    for path in paths:
        task_id = path.parent.relative_to(ROOT / "tasks").as_posix()
        # No exception-catching skips: malformed packages fail here.
        spec = load_task_spec(path, task_id=task_id, target_language=language)
        context = _context(spec)
        assert require_supported_backend(spec, CAPABILITIES) == language
        assert choose_workflow(context, target_verified=True) == "optimize"
        if spec.candidate.initial_state == "implemented":
            continue
        if language == "asm":
            with pytest.raises(ForgeRunError, match="cannot initialize an asm candidate"):
                choose_workflow(context, target_verified=False, capabilities=CAPABILITIES)
        elif language != "flydsl":
            assert choose_workflow(context, target_verified=False, capabilities=CAPABILITIES) == "initialize"
        elif context.baseline_language in CAPABILITIES["rewrite_source_languages"]:
            assert choose_workflow(context, target_verified=False, capabilities=CAPABILITIES) == "rewrite"
        else:
            with pytest.raises(ForgeRunError, match="FlyDSL rewrite reads"):
                choose_workflow(context, target_verified=False, capabilities=CAPABILITIES)


def test_schema_v2_serving_tasks_are_explicitly_outside_forge():
    paths = sorted((ROOT / "tasks" / "e2e").rglob("config.yaml"))
    for path in paths:
        spec = load_task_spec(path, task_id=path.parent.relative_to(ROOT / "tasks").as_posix())
        with pytest.raises(ValueError, match="schema-v3"):
            require_kernel_task(spec)


@pytest.mark.parametrize("language", ["tilelang", "gpu"])
def test_capability_check_uses_reported_registry_without_aliases(language):
    spec = TaskSpec.from_mapping({"schema_version": 2,
                                  "candidate": {"language": language, "editable": ["kernel.py"]},
                                  "evaluation": {"runner": ["python3", "evaluate.py"]}},
                                 task_id="arbitrary/operator")
    with pytest.raises(ForgeRunError, match=f"has no {language} backend"):
        require_supported_backend(spec, {"backends": ["flydsl", "triton"]})
    # If a future upstream actually supplies it, preserve its name.
    assert require_supported_backend(spec, {"backends": [language]}) == language
