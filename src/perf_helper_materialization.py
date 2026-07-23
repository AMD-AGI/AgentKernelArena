# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Materialize canonical perf helpers into task workspaces.

Task sources keep small stubs so generated helper code does not pollute normal
development diffs. Before a task runs, the framework replaces those stubs with
the canonical timing helpers from src/tools/perf/.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "src" / "tools" / "perf"

MARK_START = (
    "# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - "
    "edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>"
)
# Keep old marker forms readable so existing branches can be upgraded by sync.
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
FUNC_ANCHOR = "def _measure_cuda_event_fallback(fn, repetition):"
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
    """Return committed rocmbench helper stub targets under tasks/."""
    return [
        Path(p)
        for p in sorted(
            glob.glob(str(root / "tasks/*/rocmbench/**/performance_utils_pytest.py"), recursive=True)
        )
    ]


def vllm_targets(root: Path = ROOT) -> list[Path]:
    """Return committed vLLM task runners with generated helper regions."""
    return [
        Path(p)
        for p in sorted(glob.glob(str(root / "tasks/triton2triton/vllm/*/scripts/task_runner.py")))
    ]


def _marker_filtered_targets(root: Path, pattern: str) -> list[Path]:
    """Return runner files matching ``pattern`` that carry a generated helper region.

    Unlike ``vllm_targets`` (where every runner is expected to embed the markers),
    image_kernel tasks adopt the shared CUDA-graph helper task-by-task. A file only
    participates once it contains the AKA-GENERATED markers; runners still using a
    bespoke timer are left untouched.
    """
    targets = []
    for p in sorted(glob.glob(str(root / pattern))):
        text = Path(p).read_text()
        if any(marker in text for marker in MARK_STARTS) or MARK_END in text:
            targets.append(Path(p))
    return targets


def image_kernel_targets(root: Path = ROOT) -> list[Path]:
    """Return committed image_kernel task runners that carry a generated helper region."""
    return _marker_filtered_targets(root, "tasks/image_kernel/*/scripts/task_runner.py")


def canonical_rocmbench_helper(root: Path = ROOT) -> str:
    return (root / "src" / "tools" / "perf" / "performance_utils_pytest.py").read_text()


def canonical_vllm_block(root: Path = ROOT) -> str:
    text = (root / "src" / "tools" / "perf" / "vllm_cuda_graph_block.py").read_text()
    return text[text.index(FUNC_ANCHOR):]


def replace_marked_region(current: str, block: str) -> str | None:
    """Replace the generated vLLM region, returning None if markers are invalid."""
    start = next((marker for marker in MARK_STARTS if marker in current), None)
    if start is None or MARK_END not in current:
        return None
    pre = current[: current.index(start)]
    post = current[current.index(MARK_END) + len(MARK_END):]
    return pre + MARK_START + "\n" + block + MARK_END + post


def _workspace_uses_rocmbench_helper(workspace: Path) -> bool:
    """Return true when a copied task imports the rocmbench pytest helper."""
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


def materialize_perf_helpers_in_workspace(
    workspace: Path,
    logger: logging.Logger | None = None,
    root: Path = ROOT,
) -> list[Path]:
    """Replace committed stubs in a copied task workspace with canonical helpers.

    The function is safe to call more than once. It only touches
    performance_utils_pytest.py files and marked vLLM helper regions.
    """
    log = logger or logging.getLogger(__name__)
    workspace = Path(workspace)
    materialized: list[Path] = []

    rocmbench_helper = canonical_rocmbench_helper(root)
    helper = workspace / "performance_utils_pytest.py"
    if _workspace_uses_rocmbench_helper(workspace):
        if not helper.exists() or helper.read_text() != rocmbench_helper:
            helper.write_text(rocmbench_helper)
            materialized.append(helper)

    # Inline helper region lives in the task's scripts/task_runner.py (vLLM tasks
    # and image_kernel tasks), delimited by the AKA-GENERATED marker block.
    vllm_block = canonical_vllm_block(root)
    runner = workspace / "scripts" / "task_runner.py"
    if runner.exists():
        current = runner.read_text()
        has_generated_marker = any(marker in current for marker in MARK_STARTS) or MARK_END in current
        if has_generated_marker:
            new_text = replace_marked_region(current, vllm_block)
            if new_text is None:
                raise RuntimeError(f"Invalid AKA-GENERATED helper markers in workspace file: {runner}")
            if new_text != current:
                runner.write_text(new_text)
                materialized.append(runner)
        elif any(symbol in current for symbol in VLLM_HELPER_SYMBOLS):
            raise RuntimeError(f"Missing AKA-GENERATED helper markers in workspace file: {runner}")

    if materialized:
        log.info(
            "Materialized canonical perf helper(s) in workspace: %s",
            [str(p.relative_to(workspace)) for p in materialized],
        )
    return materialized
