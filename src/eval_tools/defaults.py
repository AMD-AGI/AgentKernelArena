"""Default evaluation-tool selection: the single source of truth.

A run configuration without an ``evaluation_tools`` section runs the default
sanitizers. The Docker runner starts ``DEFAULT_ENABLED`` on a default
architecture and publishes the started set as ``AKA_EVAL_TOOLS_SELECTED``;
the scoring process then plans that set under ``DEFAULT_POLICY`` with each
tool's default role. ``evaluation_tools: false`` (or ``enabled: []``) turns
the stage off, and an explicit section keeps full control.

GPU AddressSanitizer is supported but not a default: its runtime needs XNACK
host-memory access, which virtualized GPU hosts do not provide. Enable it on
bare-metal hosts with ``enabled: [..., gpu_asan]``.

This module has no package imports so the host-side runner can execute it as
a script before any container starts::

    python3 src/eval_tools/defaults.py select CONFIG.yaml GPU_ARCH
"""

from __future__ import annotations

import sys

# Every built-in tool, in the order the runner starts them.
ALL_TOOLS = (
    "triton_fpsan",
    "gpu_asan",
    "rocjitsu",
    "rocjitsu_waitcheck",
    "rocjitsu_consan",
    "hip_fpsan",
    "page_fault_guard",
    "csan",
    "tilelens",
)
DEFAULT_ENABLED = (
    "page_fault_guard",
    "csan",
    "tilelens",
    "triton_fpsan",
    "rocjitsu_waitcheck",
    "rocjitsu",
)
# Semantic comparisons report a changed floating-point expression, which a
# correct rewrite (another reduction order or precision) also produces.
SIGNAL_TOOLS = frozenset({"triton_fpsan", "hip_fpsan"})
DEFAULT_POLICY = "gated"
# Per-tool deadline on the default path; a timeout is recorded as unverified.
DEFAULT_TIMEOUT_S = 1200
DEFAULT_ARCHITECTURES = frozenset({"gfx950"})


def default_role(tool: str) -> str:
    return "signal" if tool in SIGNAL_TOOLS else "gate"


def runner_selection(config: object, gpu_arch: str) -> tuple[str, ...]:
    """The tools the runner starts for a parsed run configuration."""
    if not isinstance(config, dict) or "evaluation_tools" not in config:
        return DEFAULT_ENABLED if gpu_arch in DEFAULT_ARCHITECTURES else ()
    section = config["evaluation_tools"]
    if section in (None, False):
        return ()
    if not isinstance(section, dict):
        raise ValueError("evaluation_tools must be a mapping or false")
    enabled = section.get("enabled", ())
    if enabled is True:
        return ALL_TOOLS
    if enabled in (None, False):
        return ()
    if isinstance(enabled, str):
        return (enabled,)
    if not isinstance(enabled, (list, tuple)):
        raise ValueError("evaluation_tools.enabled must be a list of tool names")
    return tuple(str(item) for item in enabled)


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "select":
        print("usage: defaults.py select CONFIG.yaml GPU_ARCH", file=sys.stderr)
        return 2
    import yaml

    with open(argv[1], encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    print(" ".join(runner_selection(config, argv[2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
