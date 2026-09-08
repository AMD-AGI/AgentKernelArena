"""Explicit input, numerical policy, and campaign limits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    input_dir: str
    output_dir: str = "tasks/SIKL-task"
    artifact_root: str = "sikl_task_builder_runs"
    target_gpu_model: str = "MI355X"
    target_language: str = "triton"
    generator: dict = field(default_factory=lambda: {"backend": "codex", "model": None})
    validator: dict = field(default_factory=lambda: {"backend": "codex", "model": None})
    max_repair_attempts: int = 3
    agent_timeout: int = 1800
    command_timeout: int = 1800
    max_task_seconds: int = 14400
    policy: dict = field(default_factory=lambda: {
        "version": 1, "seed": 29, "rtol": 0.02, "atol": 0.02,
        "warmup": 20, "repetition": 100, "target_ms": 1.0,
    })
    selections: dict = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name in ("input_dir", "output_dir", "artifact_root", "target_gpu_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
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
        if self.target_language != "triton":
            raise ValueError("First release supports target_language: triton")
        if self.generator.get("backend", "codex") != "codex":
            raise ValueError("Generator backend must be codex")
        if self.validator.get("backend", "codex") not in {"codex", "claude_code"}:
            raise ValueError("Validator backend must be codex or claude_code")
        if type(self.max_repair_attempts) is not int or not 0 <= self.max_repair_attempts <= 20:
            raise ValueError("max_repair_attempts must be between 0 and 20")
        for name in ("agent_timeout", "command_timeout", "max_task_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        expected = {"version", "seed", "rtol", "atol", "warmup", "repetition", "target_ms"}
        if not isinstance(self.policy, dict) or set(self.policy) != expected or self.policy["version"] != 1:
            raise ValueError(f"policy requires version 1 and fields {sorted(expected)}")
        for name in ("seed", "warmup", "repetition"):
            if type(self.policy[name]) is not int or self.policy[name] < (1 if name != "seed" else 0):
                raise ValueError(f"policy.{name} must be a valid integer")
        for name in ("rtol", "atol", "target_ms"):
            value = self.policy[name]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"policy.{name} must be finite and nonnegative")
        if self.policy["target_ms"] <= 0:
            raise ValueError("policy.target_ms must be positive")

    def mapping(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    return Config(**raw)
