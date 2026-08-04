"""Configuration parsing and reproducible evaluation-plan fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    EvaluationPlan,
    EvaluationPolicy,
    SourceEvidence,
    TaskProfile,
    ToolName,
    ToolPlan,
)


PLAN_SCHEMA_VERSION = 1

# These values are discovered and attested by the selected sidecar image.  They
# are framework state, not user adapter options: accepting them from YAML would
# let the plan describe one runtime while the worker actually uses another.
RUNTIME_OPTION_KEYS: dict[str, frozenset[str]] = {
    "gpu_asan": frozenset(
        {
            "asan_runtime_dir",
            "hip_asan_runtime",
            "host_asan_preload",
            "host_asan_lib_dir",
            "normal_rocm_lib_dir",
        }
    ),
    "rocjitsu": frozenset({"rocjitsu_binary", "config_path"}),
    "hip_fpsan": frozenset({"include_dir", "public_header"}),
}


def reserved_option_keys(tool: str) -> frozenset[str]:
    return frozenset({"positive_control_required"}) | RUNTIME_OPTION_KEYS.get(
        _tool_name(tool), frozenset()
    )


def _tool_name(value: Any) -> str:
    name = value.value if isinstance(value, ToolName) else str(value)
    name = name.strip().lower().replace("-", "_")
    if not name:
        raise ValueError("evaluation tool name cannot be empty")
    return name


def _canonical(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, EvaluationPolicy):
        return value.value
    return value


@dataclass(frozen=True)
class ToolConfig:
    name: str
    runtime_ref: str | None = None
    timeout_s: int = 3600
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _tool_name(self.name))
        if int(self.timeout_s) <= 0:
            raise ValueError(f"timeout_s for {self.name} must be positive")
        object.__setattr__(self, "timeout_s", int(self.timeout_s))
        object.__setattr__(self, "options", dict(self.options or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runtime_ref": self.runtime_ref,
            "timeout_s": self.timeout_s,
            "options": _canonical(self.options),
        }


@dataclass(frozen=True)
class EvalToolsConfig:
    tools: tuple[ToolConfig, ...] = ()
    policy: EvaluationPolicy = EvaluationPolicy.ADVISORY
    runtime_profile: str | None = None
    positive_control_required: bool = True
    schema_version: int = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools or ()))
        policy = self.policy if isinstance(self.policy, EvaluationPolicy) else EvaluationPolicy(
            str(self.policy).strip().lower()
        )
        object.__setattr__(self, "policy", policy)
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError(f"evaluation_tools.enabled contains duplicates: {names}")

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    @classmethod
    def disabled(cls) -> "EvalToolsConfig":
        return cls()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "EvalToolsConfig":
        """Parse either a full run config or its ``evaluation_tools`` section."""

        if config is None:
            return cls.disabled()
        if not isinstance(config, Mapping):
            raise ValueError("evaluation tools config must be a mapping")
        section: Any = config.get("evaluation_tools", config)
        if section in (None, False):
            return cls.disabled()
        if not isinstance(section, Mapping):
            raise ValueError("evaluation_tools must be a mapping")

        policy = EvaluationPolicy(str(section.get("policy", "advisory")).strip().lower())
        runtime_profile = section.get("runtime_profile")
        if runtime_profile is not None:
            runtime_profile = str(runtime_profile).strip() or None

        raw_positive = section.get("positive_control", "required")
        if isinstance(raw_positive, bool):
            positive_required = raw_positive
        else:
            normalized = str(raw_positive).strip().lower()
            if normalized not in {"required", "optional", "disabled"}:
                raise ValueError(
                    "evaluation_tools.positive_control must be required, optional, disabled, or bool"
                )
            positive_required = normalized == "required"

        raw_enabled = section.get("enabled", ())
        # The host runner may intentionally override the configured subset.  It
        # publishes the exact normalized sidecars it started so the scoring plan
        # cannot silently expect a different set of sockets.
        selected_by_host = os.environ.get("AKA_EVAL_TOOLS_SELECTED")
        if selected_by_host is not None:
            raw_enabled = [
                item
                for item in selected_by_host.replace(",", " ").split()
                if item
            ]
        if raw_enabled is True:
            enabled = [tool.value for tool in ToolName]
        elif raw_enabled in (False, None):
            enabled = []
        elif isinstance(raw_enabled, str):
            enabled = [raw_enabled]
        elif isinstance(raw_enabled, (list, tuple)):
            enabled = list(raw_enabled)
        else:
            raise ValueError("evaluation_tools.enabled must be a list of tool names")

        raw_tool_config = section.get("tools") or {}
        if not isinstance(raw_tool_config, Mapping):
            raise ValueError("evaluation_tools.tools must be a mapping")
        default_timeout = int(section.get("timeout_s", 3600))
        if default_timeout <= 0:
            raise ValueError("evaluation_tools.timeout_s must be positive")

        tools: list[ToolConfig] = []
        for raw_name in enabled:
            name = _tool_name(raw_name)
            item = raw_tool_config.get(name) or {}
            if not isinstance(item, Mapping):
                raise ValueError(f"evaluation_tools.tools.{name} must be a mapping")
            runtime_ref = item.get("runtime_ref", item.get("image_digest"))
            if runtime_ref is not None:
                runtime_ref = str(runtime_ref).strip() or None
            # The host runner resolves the selected tag to the immutable local
            # image ID and passes that identity to both the scoring container
            # and the sidecar.  Folding it into the parsed config makes the
            # actual runtime part of the plan fingerprint even when users omit
            # ``runtime_ref``.  An explicit assertion must match rather than
            # silently describing a different image than the one that ran.
            runtime_env_name = f"AKA_EVAL_TOOL_RUNTIME_REF_{name.upper()}"
            resolved_runtime_ref = os.environ.get(runtime_env_name)
            if resolved_runtime_ref:
                resolved_runtime_ref = resolved_runtime_ref.strip() or None
            if runtime_ref and resolved_runtime_ref and runtime_ref != resolved_runtime_ref:
                raise ValueError(
                    f"evaluation_tools.tools.{name}.runtime_ref {runtime_ref!r} "
                    f"does not match selected sidecar image ID {resolved_runtime_ref!r}"
                )
            runtime_ref = runtime_ref or resolved_runtime_ref
            options = item.get("options") or {}
            if not isinstance(options, Mapping):
                raise ValueError(f"evaluation_tools.tools.{name}.options must be a mapping")
            attempted_reserved = reserved_option_keys(name).intersection(
                str(key) for key in options
            )
            if attempted_reserved:
                raise ValueError(
                    f"evaluation_tools.tools.{name}.options contains reserved framework "
                    f"keys: {sorted(attempted_reserved)}"
                )
            # The positive-control requirement is part of every invocation plan
            # and therefore of the plan fingerprint.
            merged_options = dict(options)
            merged_options["positive_control_required"] = positive_required
            tools.append(
                ToolConfig(
                    name=name,
                    runtime_ref=runtime_ref,
                    timeout_s=int(item.get("timeout_s", default_timeout)),
                    options=merged_options,
                )
            )

        return cls(
            tools=tuple(tools),
            policy=policy,
            runtime_profile=runtime_profile,
            positive_control_required=positive_required,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.value,
            "runtime_profile": self.runtime_profile,
            "positive_control_required": self.positive_control_required,
            "tools": [tool.to_dict() for tool in self.tools],
        }


def plan_fingerprint(
    *,
    config: EvalToolsConfig,
    profile: TaskProfile,
    plugin_versions: Mapping[str, str],
    source_evidence: SourceEvidence | Mapping[str, Any] | None = None,
) -> str:
    """Hash every input that can materially change a tool evaluation.

    ``runtime_ref`` is expected to be an immutable image digest in production.
    It is included through ``config``.  Plugin versions and original/candidate
    source identities are explicit so resume cannot silently reuse stale output.
    """

    evidence = SourceEvidence.from_value(source_evidence)
    payload = {
        "schema_version": config.schema_version,
        "config": config.to_dict(),
        "profile": profile.to_dict(),
        "plugin_versions": {
            name: str(plugin_versions[name]) for name in sorted(plugin_versions)
        },
        "source_evidence": evidence.to_dict(),
    }
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_evaluation_plan(
    *,
    config: EvalToolsConfig,
    profile: TaskProfile,
    plugin_versions: Mapping[str, str],
    source_evidence: SourceEvidence | Mapping[str, Any] | None = None,
) -> EvaluationPlan:
    evidence = SourceEvidence.from_value(source_evidence)
    missing_versions = [tool.name for tool in config.tools if tool.name not in plugin_versions]
    if missing_versions:
        raise KeyError(f"missing plugin versions for enabled tools: {missing_versions}")
    tools = tuple(
        ToolPlan(
            tool=item.name,
            runtime_ref=item.runtime_ref or config.runtime_profile,
            plugin_version=str(plugin_versions[item.name]),
            timeout_s=item.timeout_s,
            options=item.options,
        )
        for item in config.tools
    )
    fingerprint = plan_fingerprint(
        config=config,
        profile=profile,
        plugin_versions=plugin_versions,
        source_evidence=evidence,
    )
    return EvaluationPlan(
        schema_version=config.schema_version,
        policy=config.policy,
        profile=profile,
        tools=tools,
        fingerprint=fingerprint,
        source_evidence=evidence,
    )


def merge_task_tool_config(
    config: EvalToolsConfig,
    task_config: Mapping[str, Any],
) -> EvalToolsConfig:
    """Merge per-task adapter options without changing run-level policy/images.

    A heterogeneous run cannot use one sanitizer argv for every task.  Tasks may
    therefore supply ``evaluation_tools.tools.<id>.options`` and a shorter
    timeout.  They cannot enable tools, weaken policy/positive controls, select a
    different runtime image, or increase the run-level timeout.
    """

    section = task_config.get("evaluation_tools") or {}
    if not section:
        return config
    if not isinstance(section, Mapping):
        raise ValueError("task evaluation_tools must be a mapping")
    forbidden = set(section) - {"tools"}
    if forbidden:
        raise ValueError(
            "task evaluation_tools may only define per-tool adapter options; "
            f"forbidden fields={sorted(forbidden)}"
        )
    raw_tools = section.get("tools") or {}
    if not isinstance(raw_tools, Mapping):
        raise ValueError("task evaluation_tools.tools must be a mapping")
    unknown = set(str(name) for name in raw_tools) - set(config.enabled)
    if unknown:
        raise ValueError(
            "task config contains options for tools not enabled by the run: "
            f"{sorted(unknown)}"
        )

    merged: list[ToolConfig] = []
    for base in config.tools:
        override = raw_tools.get(base.name) or {}
        if not isinstance(override, Mapping):
            raise ValueError(
                f"task evaluation_tools.tools.{base.name} must be a mapping"
            )
        forbidden_tool = set(override) - {"options", "timeout_s"}
        if forbidden_tool:
            raise ValueError(
                f"task tool {base.name} cannot override {sorted(forbidden_tool)}"
            )
        raw_options = override.get("options") or {}
        if not isinstance(raw_options, Mapping):
            raise ValueError(
                f"task evaluation_tools.tools.{base.name}.options must be a mapping"
            )
        reserved_options = reserved_option_keys(base.name)
        attempted_reserved = reserved_options.intersection(str(key) for key in raw_options)
        if attempted_reserved:
            raise ValueError(
                f"task tool {base.name} cannot override reserved options "
                f"{sorted(attempted_reserved)}"
            )
        timeout = int(override.get("timeout_s", base.timeout_s))
        if timeout <= 0 or timeout > base.timeout_s:
            raise ValueError(
                f"task timeout for {base.name} must be in [1, {base.timeout_s}]"
            )
        merged.append(
            ToolConfig(
                name=base.name,
                runtime_ref=base.runtime_ref,
                timeout_s=timeout,
                options={**dict(base.options), **dict(raw_options)},
            )
        )
    return EvalToolsConfig(
        tools=tuple(merged),
        policy=config.policy,
        runtime_profile=config.runtime_profile,
        positive_control_required=config.positive_control_required,
        schema_version=config.schema_version,
    )


__all__ = [
    "EvalToolsConfig",
    "PLAN_SCHEMA_VERSION",
    "ToolConfig",
    "build_evaluation_plan",
    "merge_task_tool_config",
    "plan_fingerprint",
]
