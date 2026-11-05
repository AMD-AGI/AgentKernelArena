"""Exercise architecture/language selection through the real prompt builder."""

import logging
from pathlib import Path
import shutil

import pytest

from src.prompt_builder import _load_cheatsheet, prompt_builder
from src.task_contract import render_workspace


ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "src/prompts/cheatsheet"
LOGGER = logging.getLogger(__name__)
PILOT = ROOT / "tasks/normalization/rms_norm/bf16_h8192"
PILOT_ID = "normalization/rms_norm/bf16_h8192"


def expected_guides(*names):
    return "\n\n---\n\n".join((GUIDES / name).read_text() for name in names)


@pytest.mark.parametrize("model", ["RDNA4", "rdna4"])
@pytest.mark.parametrize("language", ["hip", "triton"])
def test_rdna4_selects_its_language_guide(model, language):
    text, arch = _load_cheatsheet(language, model, ROOT, LOGGER)

    assert arch == "gfx1201"
    assert text == expected_guides(
        "RDNA4_architecture.md", f"{language}_rdna_cheatsheet.md"
    )


@pytest.mark.parametrize("model,arch,architecture_files", [
    ("MI300", "gfx942", ["CDNA3_gfx942_architecture.md", "MI300_series_profile.md"]),
    ("MI300X", "gfx942", ["CDNA3_gfx942_architecture.md", "MI300X_profile.md"]),
    ("MI325", "gfx942", ["CDNA3_gfx942_architecture.md", "MI325X_profile.md"]),
    ("MI325X", "gfx942", ["CDNA3_gfx942_architecture.md", "MI325X_profile.md"]),
    ("MI300A", "gfx942", ["CDNA3_gfx942_architecture.md", "MI300A_profile.md"]),
    ("MI355X", "gfx950", ["MI355X_architecture.md"]),
])
@pytest.mark.parametrize("language", ["hip", "triton"])
def test_cdna_keeps_default_language_guide(model, arch, architecture_files, language):
    text, actual_arch = _load_cheatsheet(language, model, ROOT, LOGGER)

    assert actual_arch == arch
    assert text == expected_guides(*architecture_files, f"{language}_cheatsheet.md")


@pytest.mark.parametrize("language,guide", [
    ("flydsl", "flydsl_cheatsheet.md"),
    ("gluon", "gluon_cheatsheet.md"),
    ("gpu", "gpu_cheatsheet.md"),
])
def test_rdna4_uses_default_guides_for_languages_it_does_not_override(language, guide):
    text, arch = _load_cheatsheet(language, "RDNA4", ROOT, LOGGER)

    assert arch == "gfx1201"
    assert text == expected_guides("RDNA4_architecture.md", guide)


def test_unknown_gpu_does_not_infer_rdna_support():
    text, arch = _load_cheatsheet("hip", "UNKNOWN_GPU", ROOT, LOGGER)

    assert arch is None
    assert text == expected_guides("hip_cheatsheet.md")


@pytest.mark.parametrize("language", ["hip", "triton"])
def test_complete_v3_prompt_includes_rdna_context_and_harness_rules(language, tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(PILOT, workspace, ignore=shutil.ignore_patterns("__pycache__"))
    render_workspace(PILOT, workspace, task_id=PILOT_ID, target_language=language)
    text = prompt_builder(
        str(PILOT / "task.yaml"),
        workspace,
        {"target_gpu_model": "RDNA4", "_task_id": PILOT_ID},
        LOGGER,
    )

    assert expected_guides(
        "RDNA4_architecture.md", f"{language}_rdna_cheatsheet.md"
    ) in text
    assert f"Required final implementation backend: {language}" in text
    assert "architecture token: `gfx1201`" in text
    assert "Keep task config, input generators, case data, reference/comparison code, harnesses, and timing policy unchanged" in text
    assert "Do not write task_result.yaml" in text
    assert "do not rewrite protected files" in text
