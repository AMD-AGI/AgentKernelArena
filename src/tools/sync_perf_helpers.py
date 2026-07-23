#!/usr/bin/env python3
"""Propagate committed perf-helper stubs into every task source.

Tasks are executed inside per-run workspaces (a copy of the task folder).
Committed task sources keep small stubs so generated helper code does not pollute
normal development diffs. During setup_workspace(), the framework replaces those
stubs with the canonical helpers under src/tools/perf/.

Run this script after adding tasks or changing marker/stub structure:

    python src/tools/sync_perf_helpers.py            # apply
    python src/tools/sync_perf_helpers.py --check     # verify source stubs (CI-friendly)

Two helper families:
  1. Every */rocmbench/**/performance_utils_pytest.py should be the committed
     stub from src.perf_helper_materialization. setup_workspace() replaces it.
  2. Every triton2triton/vllm/*/scripts/task_runner.py should contain the
     committed stub block between AKA-GENERATED markers. setup_workspace()
     replaces that block.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perf_helper_materialization import (  # noqa: E402
    ROCMBENCH_HELPER_STUB,
    VLLM_HELPER_STUB_BLOCK,
    image_kernel_targets,
    replace_marked_region,
    rocmbench_targets,
    vllm_targets,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift without writing (exit 1 if any)")
    args = ap.parse_args()

    drift = []
    wrote = 0

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
        print("perf helper stubs in sync")
        return 0

    print(f"synced {wrote} file(s) "
          f"({len(rocmbench_targets(ROOT))} rocmbench + {len(vllm_targets(ROOT))} vllm "
          f"+ {len(image_kernel_targets(ROOT))} image_kernel checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
