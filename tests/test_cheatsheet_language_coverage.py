"""Every language a task declares must resolve to a cheatsheet.

An `image_kernel` task whose `repository_language` is missing from
`default_cheatsheet.yaml` does not degrade — `_load_cheatsheet` raises and the agent
never starts, so the run falls back to the baseline and reports a speedup of exactly
1.0 that looks like a real measurement. That is how the tilelang mHC task silently
produced a meaningless daily-CI row. These tests turn that class of failure into a
test failure instead of a wasted GPU-day.
"""
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHEATSHEET_CONFIG = PROJECT_ROOT / "src/prompts/cheatsheet/default_cheatsheet.yaml"


def _config() -> dict:
    return yaml.safe_load(CHEATSHEET_CONFIG.read_text()) or {}


def _declared_languages() -> dict[str, list[str]]:
    """Map each declared repository_language to the tasks that declare it."""
    languages: dict[str, list[str]] = {}
    for path in (PROJECT_ROOT / "tasks").rglob("config.yaml"):
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(cfg, dict):
            continue
        if cfg.get("task_type") not in ("repository", "image_kernel"):
            continue
        raw = cfg.get("repository_language")
        if raw is None or str(raw).strip() == "":
            continue
        rel = str(path.relative_to(PROJECT_ROOT))
        languages.setdefault(str(raw).lower().strip(), []).append(rel)
    return languages


def test_every_task_language_has_a_knowledge_entry():
    knowledge = _config().get("knowledge", {})
    missing = {
        lang: tasks for lang, tasks in _declared_languages().items()
        if lang not in knowledge
    }
    assert not missing, (
        "these tasks declare a repository_language with no cheatsheet, so the agent "
        f"will crash before it starts: {missing}. Known keys: {sorted(knowledge)}"
    )


def test_every_knowledge_cheatsheet_file_exists():
    config = _config()
    referenced = dict(config.get("knowledge", {}))
    for arch in (config.get("architecture") or {}).values():
        referenced.update((arch or {}).get("knowledge_override", {}) or {})
        if (arch or {}).get("file"):
            referenced[f"arch:{arch['file']}"] = arch["file"]

    missing = {
        key: rel for key, rel in referenced.items()
        if not (PROJECT_ROOT / rel).is_file()
    }
    assert not missing, f"cheatsheet files referenced but not present: {missing}"


def test_tilelang_resolves_for_the_mhc_task():
    """The exact task that failed in production must now build a prompt cheatsheet."""
    task = PROJECT_ROOT / "tasks/image_kernel/mi355x_vllm_tilelang_mhc_fused_post_pre/config.yaml"
    if not task.is_file():
        pytest.skip("tilelang mHC task not present in this checkout")

    cfg = yaml.safe_load(task.read_text(encoding="utf-8")) or {}
    language = str(cfg["repository_language"]).lower().strip()
    knowledge = _config().get("knowledge", {})

    assert language in knowledge, f"{language} still unregistered"
    body = (PROJECT_ROOT / knowledge[language]).read_text(encoding="utf-8")
    assert len(body) > 2000, "a stub cheatsheet is not useful guidance for the agent"
