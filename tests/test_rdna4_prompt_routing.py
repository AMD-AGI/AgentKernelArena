"""Exercise architecture/language selection through the real prompt builder."""

import logging
from pathlib import Path

import pytest

from src.prompt_builder import _load_cheatsheet, prompt_builder


ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "src/prompts/cheatsheet"
LOGGER = logging.getLogger(__name__)


def expected_guides(*names):
    return "\n\n---\n\n".join((GUIDES / name).read_text() for name in names)


@pytest.mark.parametrize("model", ["RDNA4", "rdna4"])
@pytest.mark.parametrize("task_type,language", [
    ("hip2hip", "hip"),
    ("torch2hip", "hip"),
    ("cuda2hip", "hip"),
    ("triton2triton", "triton"),
    ("instruction2triton", "triton"),
])
def test_rdna4_selects_destination_language_guide(model, task_type, language):
    text, arch = _load_cheatsheet(task_type, model, ROOT, {}, LOGGER)

    assert arch == "gfx1201"
    assert text == expected_guides(
        "RDNA4_architecture.md", f"{language}_rdna_cheatsheet.md"
    )


@pytest.mark.parametrize("model,arch,architecture_file", [
    ("MI300", "gfx942", "MI300X_architecture.md"),
    ("MI300X", "gfx942", "MI300X_architecture.md"),
    ("MI325", "gfx942", "MI300X_architecture.md"),
    ("MI325X", "gfx942", "MI300X_architecture.md"),
    ("MI355X", "gfx950", "MI355X_architecture.md"),
])
@pytest.mark.parametrize("language", ["hip", "triton"])
def test_cdna_keeps_default_language_guide(model, arch, architecture_file, language):
    text, actual_arch = _load_cheatsheet(
        f"{language}2{language}", model, ROOT, {}, LOGGER
    )

    assert actual_arch == arch
    assert text == expected_guides(architecture_file, f"{language}_cheatsheet.md")


@pytest.mark.parametrize("task_type", ["repository", "image_kernel"])
@pytest.mark.parametrize("language,guide", [
    ("hip", "hip_rdna_cheatsheet.md"),
    ("triton", "triton_rdna_cheatsheet.md"),
    ("flydsl", "flydsl_cheatsheet.md"),
    ("tilelang", "tilelang_cheatsheet.md"),
])
def test_repository_language_uses_rdna_override_or_default(task_type, language, guide):
    text, arch = _load_cheatsheet(
        task_type, "RDNA4", ROOT, {"repository_language": language}, LOGGER
    )

    assert arch == "gfx1201"
    assert text == expected_guides("RDNA4_architecture.md", guide)


@pytest.mark.parametrize("task_type", ["torch2flydsl", "triton2flydsl", "flydsl2flydsl"])
def test_conversion_uses_target_language_not_input_language(task_type):
    text, arch = _load_cheatsheet(task_type, "RDNA4", ROOT, {}, LOGGER)

    assert arch == "gfx1201"
    assert text == expected_guides("RDNA4_architecture.md", "flydsl_cheatsheet.md")


@pytest.mark.parametrize("override", ["Task-owned context", ""])
def test_task_override_replaces_both_architecture_and_language(override):
    result = _load_cheatsheet(
        "hip2hip", "RDNA4", ROOT, {"prompt": {"cheatsheet": override}}, LOGGER
    )

    assert result == (override, None)


def test_unknown_gpu_does_not_infer_rdna_support():
    text, arch = _load_cheatsheet("hip2hip", "UNKNOWN_GPU", ROOT, {}, LOGGER)

    assert arch is None
    assert text == expected_guides("hip_cheatsheet.md")


@pytest.mark.parametrize("task_path,language", [
    ("hip2hip/gpumode/GELU", "hip"),
    ("triton2triton/vllm/triton_rms_norm", "triton"),
])
def test_complete_task_prompt_includes_rdna_context_and_harness_rules(
    task_path, language, tmp_path
):
    text = prompt_builder(
        str(ROOT / "tasks" / task_path / "config.yaml"),
        tmp_path,
        {"target_gpu_model": "RDNA4"},
        LOGGER,
    )

    assert expected_guides(
        "RDNA4_architecture.md", f"{language}_rdna_cheatsheet.md"
    ) in text
    assert "architecture token: `gfx1201`" in text
    assert "### Protected Harness / Test Files" in text
    assert "DO NOT write task_result.yaml" in text
