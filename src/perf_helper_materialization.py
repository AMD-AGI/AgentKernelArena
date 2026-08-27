# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Materialize canonical performance helpers into task workspaces.

Committed tasks keep small stubs/imports.  A copied run workspace receives the
self-contained helpers beside its performance entrypoints, so it never imports
AgentKernelArena's ``src`` package at runtime.
"""

from __future__ import annotations

import glob
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # Minimal perf-helper audit CI intentionally has no PyYAML.
    yaml = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "src" / "tools" / "perf"

AKA_HELPER_FILE_NAME = "_aka_benchmark.py"
NATIVE_HIP_DRIVER = Path("scripts/native/benchmark_driver.hip")
NATIVE_HIP_INCLUDE = '#include "hip_graph_benchmark.hpp"'
NATIVE_HIP_MATERIALIZED = Path("scripts/native/hip_graph_benchmark.hpp")

MARK_START = (
    "# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - "
    "edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>"
)
OLD_MARK_START = (
    "# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - "
    "edit tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>"
)
LEGACY_MARK_START = (
    "# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers \u2014 "
    "edit tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>"
)
MARK_STARTS = (MARK_START, OLD_MARK_START, LEGACY_MARK_START)
MARK_END = "# <<< AKA-GENERATED <<<"
VLLM_BLOCK_ANCHOR = "import sys as _aka_sys"
VLLM_HELPER_SYMBOLS = ("_measure_cuda_event_fallback", "_benchmark_cuda_graph_or_events")

ROCMBENCH_HELPER_STUB = '''"""Generated at workspace setup from src/tools/perf/performance_utils_pytest.py.

This task-source file is intentionally a stub. AgentKernelArena replaces it
with the canonical helper inside each run workspace before compile, correctness,
and performance commands execute.
"""

raise RuntimeError(
    "performance_utils_pytest.py is a generated stub in task sources. "
    "Run the task through AgentKernelArena so setup_workspace() can materialize "
    "src/tools/perf/performance_utils_pytest.py into the workspace."
)
'''

VLLM_HELPER_STUB_BLOCK = '''def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
'''


def rocmbench_targets(root: Path = ROOT) -> list[Path]:
    """Return committed ROCmBench helper stub targets under ``tasks/``."""

    return [
        Path(path)
        for path in sorted(
            glob.glob(
                str(root / "tasks/*/rocmbench/**/performance_utils_pytest.py"),
                recursive=True,
            )
        )
    ]


def vllm_targets(root: Path = ROOT) -> list[Path]:
    """Return committed vLLM runners with generated helper regions."""

    return [
        Path(path)
        for path in sorted(
            glob.glob(str(root / "tasks/triton2triton/vllm/*/scripts/task_runner.py"))
        )
    ]


def _marker_filtered_targets(root: Path, pattern: str) -> list[Path]:
    targets = []
    for path_string in sorted(glob.glob(str(root / pattern))):
        path = Path(path_string)
        text = path.read_text()
        if any(marker in text for marker in MARK_STARTS) or MARK_END in text:
            targets.append(path)
    return targets


def image_kernel_targets(root: Path = ROOT) -> list[Path]:
    """Return image task runners that carry a generated helper region."""

    return _marker_filtered_targets(root, "tasks/image_kernel/*/scripts/task_runner.py")


def canonical_rocmbench_helper(root: Path = ROOT) -> str:
    return (root / "src/tools/perf/performance_utils_pytest.py").read_text()


def canonical_aka_helper(root: Path = ROOT) -> str:
    return (root / "src/tools/perf/aka_benchmark.py").read_text()


def canonical_vllm_block(root: Path = ROOT) -> str:
    text = (root / "src/tools/perf/vllm_cuda_graph_block.py").read_text()
    return text[text.index(VLLM_BLOCK_ANCHOR) :]


def replace_marked_region(current: str, block: str) -> str | None:
    """Replace an AKA-GENERATED region, or return ``None`` for bad markers."""

    start = next((marker for marker in MARK_STARTS if marker in current), None)
    if start is None or MARK_END not in current:
        return None
    pre = current[: current.index(start)]
    post = current[current.index(MARK_END) + len(MARK_END) :]
    return pre + MARK_START + "\n" + block + MARK_END + post


def _workspace_uses_rocmbench_helper(workspace: Path) -> bool:
    helper = workspace / "performance_utils_pytest.py"
    if helper.exists():
        return True
    for source in workspace.glob("*.py"):
        try:
            if "performance_utils_pytest" in source.read_text():
                return True
        except UnicodeDecodeError:
            continue
    return False


_TOP_LEVEL_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:")
_SEQUENCE_ITEM = re.compile(r"^(\s*)-\s+(.*)$")


def _yaml_scalar(value: str) -> str:
    """Decode the small YAML string subset used by benchmark path fields."""

    value = value.strip()
    if not value:
        return ""
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]

    # In a plain YAML scalar a hash starts a comment only when separated from
    # the value by whitespace.  Hashes embedded in shell/Python tokens survive.
    value = re.split(r"\s+#", value, maxsplit=1)[0]
    return value.rstrip()


def _yaml_flow_sequence(value: str) -> list[str] | None:
    """Parse a top-level YAML flow sequence of scalar strings."""

    if not (value.startswith("[") and value.endswith("]")):
        return None
    fields: list[str] = []
    start = 1
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value[1:-1], start=1):
        if quote is not None:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == ",":
            fields.append(_yaml_scalar(value[start:index]))
            start = index + 1
    fields.append(_yaml_scalar(value[start:-1]))
    return [field for field in fields if field]


def _yaml_sequence(lines: list[str]) -> list[str]:
    """Parse a block sequence, including YAML's folded continuation lines."""

    items: list[tuple[str, list[str]]] = []
    for line in lines:
        match = _SEQUENCE_ITEM.match(line)
        if match:
            items.append((match.group(2).strip(), []))
        elif items and (not line or line[0].isspace()):
            items[-1][1].append(line.strip())

    parsed: list[str] = []
    for first, continuation in items:
        if first in {"|", "|-", "|+"}:
            value = "\n".join(continuation)
        elif first in {">", ">-", ">+"}:
            value = " ".join(part for part in continuation if part)
        else:
            value = " ".join(part for part in [first, *continuation] if part)
        parsed.append(_yaml_scalar(value))
    return parsed


def _dependency_free_config_fields(text: str) -> dict[str, Any]:
    """Read only the YAML fields needed to locate benchmark harnesses.

    This intentionally is not a general YAML loader.  It supports scalar and
    block/flow-sequence forms for ``performance_command`` plus scalar
    ``harness_path``.  Keeping this narrow path in the standard library lets the
    perf-helper source audit run in a clean Python installation; full task config
    consumers continue to use PyYAML.
    """

    lines = text.splitlines()
    config: dict[str, Any] = {}
    for index, line in enumerate(lines):
        if line[:1].isspace() or line.startswith("#") or ":" not in line:
            continue
        key, inline = line.split(":", 1)
        key = key.strip()
        if key not in {"harness_path", "performance_command"}:
            continue

        inline = inline.strip()
        if key == "harness_path":
            config[key] = _yaml_scalar(inline)
            continue
        if inline:
            flow_sequence = _yaml_flow_sequence(inline)
            config[key] = (
                flow_sequence if flow_sequence is not None else _yaml_scalar(inline)
            )
            continue

        end = index + 1
        while end < len(lines):
            candidate = lines[end]
            if candidate and not candidate[0].isspace() and _TOP_LEVEL_KEY.match(candidate):
                break
            end += 1
        config[key] = _yaml_sequence(lines[index + 1 : end])
    return config


def _load_workspace_config(workspace: Path) -> dict[str, Any]:
    for name in ("config.yaml", "config.yml"):
        path = workspace / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except OSError:
            return {}
        if yaml is None:
            return _dependency_free_config_fields(text)
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _resolved_workspace_file(workspace: Path, token: str) -> Path | None:
    token = token.split("::", 1)[0].strip("'\"")
    if not token or token in {".", ".."}:
        return None
    candidate = Path(token)
    if candidate.is_absolute():
        return None
    path = (workspace / candidate).resolve()
    try:
        path.relative_to(workspace.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _python_entrypoint(tokens: list[str], index: int, workspace: Path) -> Path | None:
    position = index + 1
    while position < len(tokens):
        token = tokens[position]
        if token in {"&&", "||", ";", "|"}:
            return None
        if token == "-c":
            return None
        if token == "-m":
            # ``python -m pytest path.py``: protect the selected pytest file.
            position += 2
            while position < len(tokens):
                path = _resolved_workspace_file(workspace, tokens[position])
                if path is not None:
                    return path
                position += 1
            return None
        if not token.startswith("-"):
            return _resolved_workspace_file(workspace, token)
        position += 1
    return None


def configured_performance_entrypoints(workspace: Path) -> set[Path]:
    """Resolve source files directly invoked by ``performance_command``.

    The parser deliberately selects executable scripts, not arbitrary file-valued
    arguments such as ``--hip_file source.hip``.  This protects benchmark harnesses
    without accidentally making the kernel source immutable.
    """

    workspace = Path(workspace)
    config = _load_workspace_config(workspace)
    entrypoints: set[Path] = set()

    harness_path = config.get("harness_path")
    if isinstance(harness_path, str):
        path = _resolved_workspace_file(workspace, harness_path)
        if path is not None:
            entrypoints.add(path)

    commands = config.get("performance_command") or []
    if isinstance(commands, str):
        commands = [commands]
    for command in commands:
        if not isinstance(command, str):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        for index, token in enumerate(tokens):
            executable = Path(token).name
            path: Path | None = None
            if executable.startswith("python") or executable in {"pypy", "pypy3"}:
                path = _python_entrypoint(tokens, index, workspace)
            elif executable in {"bash", "sh"} and index + 1 < len(tokens):
                path = _resolved_workspace_file(workspace, tokens[index + 1])
            elif executable.startswith("pytest"):
                for candidate in tokens[index + 1 :]:
                    path = _resolved_workspace_file(workspace, candidate)
                    if path is not None:
                        break
            elif index == 0:
                path = _resolved_workspace_file(workspace, token)
            if path is not None:
                entrypoints.add(path)
    return entrypoints


def _aka_importing_entrypoints(workspace: Path) -> set[Path]:
    candidates = configured_performance_entrypoints(workspace)
    candidates.update(workspace.glob("*.py"))
    for directory in (workspace / "scripts", workspace / "eval_tools"):
        if directory.is_dir():
            candidates.update(directory.glob("*.py"))

    importers: set[Path] = set()
    for path in candidates:
        if not path.is_file() or path.name == AKA_HELPER_FILE_NAME:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if "_aka_benchmark" in text:
            importers.add(path)
    return importers


def _materialize_file(path: Path, content: str, materialized: list[Path]) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    materialized.append(path)


def _materialize_native_helper(
    workspace: Path,
    root: Path,
    materialized: list[Path],
) -> None:
    driver = workspace / NATIVE_HIP_DRIVER
    if not driver.is_file() or NATIVE_HIP_INCLUDE not in driver.read_text():
        return
    canonical = root / "src/tools/perf/native_hip_graph_benchmark.hpp"
    if not canonical.is_file():
        raise RuntimeError(
            f"Native graph benchmark requested by {driver}, but canonical helper is missing: {canonical}"
        )
    _materialize_file(workspace / NATIVE_HIP_MATERIALIZED, canonical.read_text(), materialized)


def materialize_perf_helpers_in_workspace(
    workspace: Path,
    logger: logging.Logger | None = None,
    root: Path = ROOT,
) -> list[Path]:
    """Materialize canonical helpers in a copied task workspace.

    The operation is idempotent.  Python helpers are copied only beside files
    that import ``_aka_benchmark``; native HIP support is copied only when the
    canonical include is requested by the protected benchmark driver.
    """

    log = logger or logging.getLogger(__name__)
    workspace = Path(workspace)
    materialized: list[Path] = []

    if _workspace_uses_rocmbench_helper(workspace):
        _materialize_file(
            workspace / "performance_utils_pytest.py",
            canonical_rocmbench_helper(root),
            materialized,
        )

    runner = workspace / "scripts/task_runner.py"
    if runner.exists():
        current = runner.read_text()
        has_marker = any(marker in current for marker in MARK_STARTS) or MARK_END in current
        if has_marker:
            new_text = replace_marked_region(current, canonical_vllm_block(root))
            if new_text is None:
                raise RuntimeError(f"Invalid AKA-GENERATED helper markers in workspace file: {runner}")
            if new_text != current:
                runner.write_text(new_text)
                materialized.append(runner)
        elif any(symbol in current for symbol in VLLM_HELPER_SYMBOLS):
            raise RuntimeError(f"Missing AKA-GENERATED helper markers in workspace file: {runner}")

    canonical_python = canonical_aka_helper(root)
    helper_targets = {
        entrypoint.parent / AKA_HELPER_FILE_NAME
        for entrypoint in _aka_importing_entrypoints(workspace)
    }
    for helper in sorted(helper_targets):
        _materialize_file(helper, canonical_python, materialized)

    _materialize_native_helper(workspace, root, materialized)

    if materialized:
        log.info(
            "Materialized canonical perf helper(s) in workspace: %s",
            [str(path.relative_to(workspace)) for path in materialized],
        )
    return materialized
