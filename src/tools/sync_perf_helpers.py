#!/usr/bin/env python3
"""Propagate committed perf-helper stubs into every task source.

Tasks are executed inside per-run workspaces (a copy of the task folder).
Committed task sources keep small stubs so generated helper code does not pollute
normal development diffs. During setup_workspace(), the framework replaces those
stubs with the canonical helpers under src/tools/perf/.

Run this script after adding tasks or changing marker/stub structure:

    python src/tools/sync_perf_helpers.py            # apply
    python src/tools/sync_perf_helpers.py --check     # verify source stubs (CI-friendly)

Four supported benchmark entrypoint families:
  1. Every */rocmbench/**/performance_utils_pytest.py should be the committed
     stub from src.perf_helper_materialization. setup_workspace() replaces it.
  2. Every triton2triton/vllm/*/scripts/task_runner.py should contain the
     committed stub block between AKA-GENERATED markers. setup_workspace()
     replaces that block.
  3. Migrated Python entrypoints import ``_aka_benchmark``; setup_workspace()
     copies the canonical module beside the configured entrypoint.
  4. Native HIP benchmark drivers include ``hip_graph_benchmark.hpp``;
     setup_workspace() copies the canonical header beside the driver.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perf_helper_materialization import (  # noqa: E402
    MARK_STARTS,
    NATIVE_HIP_DRIVER,
    NATIVE_HIP_INCLUDE,
    ROCMBENCH_HELPER_STUB,
    VLLM_HELPER_STUB_BLOCK,
    configured_performance_entrypoints,
    image_kernel_targets,
    replace_marked_region,
    rocmbench_targets,
    vllm_targets,
)


def audit_task_benchmark_entrypoints(root: pathlib.Path) -> tuple[dict[str, int], list[str]]:
    """Classify every task config's performance path or report it as unsupported."""

    counts: dict[str, int] = {}
    problems: list[str] = []
    task_dirs = sorted(path.parent for path in (root / "tasks").rglob("config.yaml"))
    for task in task_dirs:
        entrypoints = configured_performance_entrypoints(task)
        family = None

        native_driver = task / NATIVE_HIP_DRIVER
        if native_driver.is_file() and NATIVE_HIP_INCLUDE in native_driver.read_text():
            family = "native_graph_driver"

        if family is None:
            for entrypoint in sorted(entrypoints):
                try:
                    text = entrypoint.read_text()
                except (OSError, UnicodeDecodeError):
                    continue
                if "_aka_benchmark" in text:
                    family = "canonical_python"
                    break
                if any(marker in text for marker in MARK_STARTS):
                    family = "vllm_adapter"
                    break

        if family is None and (task / "performance_utils_pytest.py").is_file():
            family = "rocmbench_adapter"

        if family is None:
            # An entrypoint may delegate its timing to one of the task's own
            # modules rather than call the canonical helper itself, which is how
            # a task keeps a single timing implementation shared by its harness
            # and any other driver it ships. setup_workspace() already
            # materializes the helper beside whichever file imports it, so the
            # task is on the canonical family wherever that import lives.
            if any(
                "_aka_benchmark" in module.read_text(errors="replace")
                for module in sorted(task.rglob("*.py"))
            ):
                family = "canonical_python"

        if family is None:
            relative = task.relative_to(root)
            resolved = [str(path.relative_to(task)) for path in sorted(entrypoints)]
            problems.append(
                f"{relative}: unrecognized performance entrypoint(s): {resolved or ['<none>']}"
            )
            continue
        counts[family] = counts.get(family, 0) + 1

    # Materialized helpers belong only in copied workspaces, never committed
    # into task sources where they would drift from the canonical implementation.
    for generated in sorted((root / "tasks").rglob("_aka_benchmark.py")):
        problems.append(f"{generated.relative_to(root)}: generated Python helper is committed")
    for generated in sorted((root / "tasks").rglob("hip_graph_benchmark.hpp")):
        problems.append(f"{generated.relative_to(root)}: generated native helper is committed")

    return counts, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift without writing (exit 1 if any)")
    args = ap.parse_args()

    drift = []
    wrote = 0

    audit_counts, audit_problems = audit_task_benchmark_entrypoints(ROOT)
    if audit_problems:
        print(
            f"ERROR: benchmark entrypoint audit failed ({len(audit_problems)} problem(s))",
            file=sys.stderr,
        )
        for problem in audit_problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    for p in rocmbench_targets(ROOT):
        if p.read_text() != ROCMBENCH_HELPER_STUB:
            drift.append(str(p))
            if not args.check:
                p.write_text(ROCMBENCH_HELPER_STUB)
                wrote += 1

    inline_targets = list(vllm_targets(ROOT)) + list(image_kernel_targets(ROOT))
    for p in inline_targets:
        cur = p.read_text()
        new = replace_marked_region(cur, VLLM_HELPER_STUB_BLOCK)
        if new is None:
            print(f"ERROR: missing AKA-GENERATED markers in {p}", file=sys.stderr)
            return 2
        if new != cur:
            drift.append(str(p))
            if not args.check:
                p.write_text(new)
                wrote += 1

    if args.check:
        if drift:
            print(f"OUT OF SYNC ({len(drift)} files). Run: python src/tools/sync_perf_helpers.py")
            for t in drift:
                print(f"  {t}")
            return 1
        print(f"perf helper stubs in sync; benchmark entrypoints={audit_counts}")
        return 0

    print(f"synced {wrote} file(s) "
          f"({len(rocmbench_targets(ROOT))} rocmbench + {len(vllm_targets(ROOT))} vllm "
          f"+ {len(image_kernel_targets(ROOT))} image_kernel checked); "
          f"benchmark entrypoints={audit_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
