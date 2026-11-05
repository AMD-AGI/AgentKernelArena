"""Explicit input, provenance and campaign limits.

The builder emits schema-v3 task packages, which name no candidate language:
the run configuration that later optimizes a task selects it. Seeds, timing
and the default error bound belong to the framework's task contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REVIEW_BACKENDS = ("codex", "claude_code", "none")


@dataclass
class Config:
    input_dir: str
    output_dir: str = "tasks/SIKL-task"
    artifact_root: str = "sikl_task_builder_runs"
    target_gpu_model: str = "MI355X"
    # SPDX identifier of the bundled baseline code when its solution records
    # carry no license. Never a placeholder; a non-Apache license also needs
    # the bundle's LICENSE text.
    license: str | None = None
    generator: dict = field(default_factory=lambda: {"backend": "codex", "model": None})
    # The QA auditor's model review; "none" runs only its deterministic checks.
    validator: dict = field(default_factory=lambda: {"backend": "codex", "model": None})
    max_repair_attempts: int = 3
    agent_timeout: int = 1800
    command_timeout: int = 1800
    max_task_seconds: int = 14400
    selections: dict = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name in ("input_dir", "output_dir", "artifact_root", "target_gpu_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.license is not None and (not isinstance(self.license, str) or not self.license.strip()):
            raise ValueError("license must be a nonempty SPDX identifier or null")
        for name in ("generator", "validator"):
            settings = getattr(self, name)
            if not isinstance(settings, dict) or set(settings) - {"backend", "model", "effort"}:
                raise ValueError(f"{name} accepts backend, model and effort")
            if any(v is not None and not isinstance(v, str) for v in settings.values()):
                raise ValueError(f"{name} settings must be strings or null")
        if not isinstance(self.tasks, list) or any(not isinstance(t, str) for t in self.tasks):
            raise ValueError("tasks must be a list of definition names")
        if not isinstance(self.selections, dict) or any(
            not isinstance(v, dict) or set(v) - {"baseline", "reference"}
            or any(not isinstance(n, str) for n in v.values()) for v in self.selections.values()
        ):
            raise ValueError("selections maps definition names to baseline/reference solution names")
        if self.generator.get("backend", "codex") != "codex":
            raise ValueError("Generator backend must be codex")
        if self.validator.get("backend", "codex") not in REVIEW_BACKENDS:
            raise ValueError(f"Validator (QA review) backend must be one of {REVIEW_BACKENDS}")
        if type(self.max_repair_attempts) is not int or not 0 <= self.max_repair_attempts <= 20:
            raise ValueError("max_repair_attempts must be between 0 and 20")
        for name in ("agent_timeout", "command_timeout", "max_task_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def mapping(self) -> dict:
        return asdict(self)

    def review_backend(self) -> str:
        return self.validator.get("backend") or "codex"

    def arena_task_id(self, definition: str) -> str:
        repo = Path(__file__).resolve().parents[2]
        output = (repo / self.output_dir).resolve()
        return (output.relative_to(repo / "tasks") / definition).as_posix()


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    return Config(**raw)
