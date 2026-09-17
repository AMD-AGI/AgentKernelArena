"""Every public task language must have usable prompt knowledge.

Knowledge availability is independent of any one agent's backend support:
TileLang requires a real TileLang cheatsheet even when Forge cannot run it.
"""

from pathlib import Path

import pytest
import yaml
from src.task_spec import load_task_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHEATSHEET_CONFIG = PROJECT_ROOT / "src/prompts/cheatsheet/default_cheatsheet.yaml"


def _config() -> dict:
    return yaml.safe_load(CHEATSHEET_CONFIG.read_text()) or {}


def _declared_languages() -> dict[str, list[str]]:
    """Inspect all validated v2 configs, including generation and conversion."""
    languages: dict[str, list[str]] = {}
    paths = sorted((PROJECT_ROOT / "tasks").rglob("config.yaml"))
    assert paths, "No task configs found"
    for path in paths:
        task_id = path.parent.relative_to(PROJECT_ROOT / "tasks").as_posix()
        candidate = load_task_spec(path, task_id=task_id).candidate
        rel = str(path.relative_to(PROJECT_ROOT))
        for language in {candidate.language, candidate.initial_language} - {None}:
            languages.setdefault(language, []).append(rel)
    assert len({path for tasks in languages.values() for path in tasks}) == len(paths)
    return languages


def _referenced_cheatsheets(config: dict) -> dict[str, str]:
    """Return every configured cheatsheet path without collapsing overrides."""
    referenced = {
        f"knowledge:{language}": rel
        for language, rel in (config.get("knowledge") or {}).items()
    }
    for arch_name, arch in (config.get("architecture") or {}).items():
        arch = arch or {}
        if arch.get("file"):
            referenced[f"architecture:{arch_name}:file"] = arch["file"]
        for language, rel in (arch.get("knowledge_override") or {}).items():
            referenced[f"architecture:{arch_name}:knowledge_override:{language}"] = rel
    return referenced


def test_every_task_language_has_a_knowledge_entry():
    knowledge = _config().get("knowledge", {})
    missing = {
        lang: tasks
        for lang, tasks in _declared_languages().items()
        if lang not in knowledge
    }
    assert not missing, (
        "these tasks declare a language with no cheatsheet, so the agent "
        f"will crash before it starts: {missing}. Known keys: {sorted(knowledge)}"
    )


def test_every_knowledge_cheatsheet_file_exists():
    missing = {
        key: rel
        for key, rel in _referenced_cheatsheets(_config()).items()
        if not (PROJECT_ROOT / rel).is_file()
    }
    assert not missing, f"cheatsheet files referenced but not present: {missing}"


def test_reference_collection_keeps_defaults_and_architecture_overrides():
    referenced = _referenced_cheatsheets(
        {
            "knowledge": {"hip": "default-hip.md"},
            "architecture": {
                "RDNA4": {
                    "file": "rdna4.md",
                    "knowledge_override": {"hip": "rdna-hip.md"},
                }
            },
        }
    )

    assert referenced == {
        "knowledge:hip": "default-hip.md",
        "architecture:RDNA4:file": "rdna4.md",
        "architecture:RDNA4:knowledge_override:hip": "rdna-hip.md",
    }


def test_tilelang_resolves_for_the_mhc_task():
    """The exact task that failed in production must now build a prompt cheatsheet."""
    task = (
        PROJECT_ROOT
        / "tasks/image_kernel/mi355x_vllm_tilelang_mhc_fused_post_pre/config.yaml"
    )
    if not task.is_file():
        pytest.skip("tilelang mHC task not present in this checkout")

    spec = load_task_spec(task, task_id="image_kernel/mi355x_vllm_tilelang_mhc_fused_post_pre")
    language = spec.candidate.language
    assert language == "tilelang"
    knowledge = _config().get("knowledge", {})

    assert language in knowledge, f"{language} still unregistered"
    body = (PROJECT_ROOT / knowledge[language]).read_text(encoding="utf-8")
    assert len(body) > 2000, "a stub cheatsheet is not useful guidance for the agent"
