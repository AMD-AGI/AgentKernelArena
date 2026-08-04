"""Built-in eval-tool plugin registry."""

from __future__ import annotations

from typing import Iterable

from .gpu_asan import GpuAsanPlugin
from .hip_fpsan import HipFpSanPlugin
from .rocjitsu import RocJitsuPlugin
from .triton_fpsan import TritonFpSanPlugin


_PLUGINS = {
    plugin.name: plugin
    for plugin in (
        TritonFpSanPlugin(),
        GpuAsanPlugin(),
        RocJitsuPlugin(),
        HipFpSanPlugin(),
    )
}


def get_plugin(name: object):
    key = str(getattr(name, "value", name)).strip().lower().replace("-", "_")
    try:
        return _PLUGINS[key]
    except KeyError as exc:
        raise KeyError(f"unknown eval-tool plugin {name!r}; available={sorted(_PLUGINS)}") from exc


def iter_plugins() -> Iterable[object]:
    return tuple(_PLUGINS[name] for name in sorted(_PLUGINS))


def plugin_ids() -> tuple[str, ...]:
    return tuple(sorted(_PLUGINS))


def register_builtin_plugins(registry=None, *, replace: bool = False):
    """Register all built-ins in the core :class:`ToolRegistry`.

    Registration is explicit so importing plugin parsers in a sidecar does not
    mutate the process-global manager registry.
    """

    from ..registry import DEFAULT_TOOL_REGISTRY

    # ``ToolRegistry`` is falsey while empty because it implements ``__len__``.
    # An explicitly supplied empty registry is still the requested target.
    target = DEFAULT_TOOL_REGISTRY if registry is None else registry
    for plugin in iter_plugins():
        target.register(plugin, replace=replace)
    return target
