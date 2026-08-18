"""Registry for deterministic evaluation-tool plugins."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .contracts import ToolName, ToolPlugin


def normalize_tool_name(value: str | ToolName) -> str:
    name = value.value if isinstance(value, ToolName) else str(value)
    name = name.strip().lower().replace("-", "_")
    if not name:
        raise ValueError("tool plugin name cannot be empty")
    return name


class ToolRegistry:
    """Small explicit registry; importing this module has no plugin side effects."""

    def __init__(self) -> None:
        self._plugins: dict[str, ToolPlugin] = {}

    @staticmethod
    def _validate_plugin(plugin: Any) -> None:
        missing = [
            name
            for name in ("name", "version", "assess", "build_invocation", "parse")
            if not hasattr(plugin, name)
        ]
        if missing:
            raise TypeError(f"invalid tool plugin; missing attributes: {missing}")
        for name in ("assess", "build_invocation", "parse"):
            if not callable(getattr(plugin, name)):
                raise TypeError(f"tool plugin attribute {name!r} must be callable")
        if not str(plugin.version).strip():
            raise ValueError("tool plugin version cannot be empty")

    def register(self, plugin: ToolPlugin, *, replace: bool = False) -> ToolPlugin:
        self._validate_plugin(plugin)
        name = normalize_tool_name(plugin.name)
        if name in self._plugins and not replace:
            raise ValueError(f"tool plugin already registered: {name}")
        self._plugins[name] = plugin
        return plugin

    def unregister(self, name: str | ToolName) -> ToolPlugin:
        normalized = normalize_tool_name(name)
        try:
            return self._plugins.pop(normalized)
        except KeyError as exc:
            raise KeyError(f"tool plugin not registered: {normalized}") from exc

    def get(self, name: str | ToolName) -> ToolPlugin:
        normalized = normalize_tool_name(name)
        try:
            return self._plugins[normalized]
        except KeyError as exc:
            raise KeyError(
                f"tool plugin not registered: {normalized}; available={list(self._plugins)}"
            ) from exc

    def versions(self, names: tuple[str, ...] | list[str] | None = None) -> dict[str, str]:
        selected = list(self._plugins) if names is None else [normalize_tool_name(n) for n in names]
        return {name: str(self.get(name).version) for name in selected}

    def __contains__(self, name: object) -> bool:
        try:
            normalized = normalize_tool_name(str(name))
        except (TypeError, ValueError):
            return False
        return normalized in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self) -> Iterator[str]:
        return iter(self._plugins)


DEFAULT_TOOL_REGISTRY = ToolRegistry()


def register_tool(plugin: ToolPlugin, *, replace: bool = False) -> ToolPlugin:
    """Register against the process-local default registry."""

    return DEFAULT_TOOL_REGISTRY.register(plugin, replace=replace)


__all__ = [
    "DEFAULT_TOOL_REGISTRY",
    "ToolRegistry",
    "normalize_tool_name",
    "register_tool",
]
