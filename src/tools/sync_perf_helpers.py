#!/usr/bin/env python3
"""Audit committed benchmark entrypoints against the canonical helpers.

Schema-v3 kernel tasks (``task.yaml``) carry no benchmark code: the framework
writes the evaluator and its timing helper (``src/tools/perf/aka_benchmark.py``)
into each run workspace. The only schema-v2 tasks are the end-to-end serving
tasks, whose measurement code also stays in ``src/tools/perf/``. This audit
fails when a committed task would not receive that code:

  1. Every ``config.yaml`` task is either a serving recipe (an exact bootstrap
     stub plus a hash-pinned workload; host preparation supplies the client) or
     a serving client whose ``scripts/evaluate.py`` imports ``_aka_serving``
     (``src.perf_helper_materialization`` copies the helpers beside it).
  2. No generated helper copy is committed under ``tasks/``.

It audits task sources only; it is not a GPU qualification. Run it with
``make check-perf-helpers``.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perf_helper_materialization import (  # noqa: E402
    GENERATED_HELPER_NAMES,
    SERVING_RECIPE_RUNNER_STUB,
    SERVING_RUNNER,
)
from src.serving_task_preparation import recipe_identity  # noqa: E402

# Files the framework writes into run workspaces; never task sources.
GENERATED_NAMES = GENERATED_HELPER_NAMES | {"_aka_benchmark.py"}
GENERATED_DIRECTORIES = frozenset({"_aka_eval"})


def _serving_recipe(task: pathlib.Path, root: pathlib.Path) -> bool:
    """Audit a deferred serving entrypoint, without calling it GPU-qualified."""
    try:
        task_id = task.relative_to(root / 'tasks').as_posix()
        recipe_identity(task, task_id)
        config = (task / 'config.yaml').read_text()
        return ((task / SERVING_RUNNER).read_text() == SERVING_RECIPE_RUNNER_STUB
                and re.search(r'^schema_version: 2\s*$', config, re.MULTILINE) is not None
                and re.search(r'^\s+kind: serving\s*$', config, re.MULTILINE) is not None
                and re.search(r'^\s+recipe: recipe.json\s*$', config, re.MULTILINE) is not None)
    except (OSError, ValueError, TypeError):
        return False


def _serving_client(task: pathlib.Path) -> bool:
    try:
        return "_aka_serving" in (task / SERVING_RUNNER).read_text()
    except (OSError, UnicodeDecodeError):
        return False


def audit_task_benchmark_entrypoints(root: pathlib.Path) -> tuple[dict[str, int], list[str]]:
    """Classify every committed task's benchmark path or report it as unsupported."""

    counts: dict[str, int] = {}
    problems: list[str] = []
    tasks = root / "tasks"
    if kernel_tasks := sum(1 for _ in tasks.rglob("task.yaml")):
        counts["kernel_v3"] = kernel_tasks
    for task in sorted(path.parent for path in tasks.rglob("config.yaml")):
        if _serving_recipe(task, root):
            family = "serving_recipe"
        elif _serving_client(task):
            family = "serving_client"
        else:
            problems.append(f"{task.relative_to(root)}: schema-v2 task is neither a serving recipe "
                            f"nor a serving client importing _aka_serving from {SERVING_RUNNER}")
            continue
        counts[family] = counts.get(family, 0) + 1

    for path in sorted(tasks.rglob("*")):
        if path.name in GENERATED_NAMES or (path.is_dir() and path.name in GENERATED_DIRECTORIES):
            problems.append(f"{path.relative_to(root)}: generated helper is committed")
    return counts, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    counts, problems = audit_task_benchmark_entrypoints(ROOT)
    if problems:
        print(f"ERROR: benchmark entrypoint audit failed ({len(problems)} problem(s))", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"benchmark entrypoints audited: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
