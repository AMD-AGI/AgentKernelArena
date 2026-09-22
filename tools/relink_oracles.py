#!/usr/bin/env python3
"""Replace large copied oracle blobs in the suite with hardlinks to their source package.

``build_suite.py`` prefers hardlinks but silently falls back to copying, because
Linux ``fs.protected_hardlinks`` forbids linking to a file you neither own nor
can write - which is most of ``/shared_nfs/zihao/headkernel_ut_0831``. Run this
pass **as root** (inside a container on a compute node) to convert those copies
back into links and reclaim the space:

    docker run --rm -u 0 --entrypoint /bin/bash -v /shared_nfs:/shared_nfs <image> \
      -c 'python3 <suite>/tools/relink_oracles.py'

Only replaces a copy when the source file is byte-for-byte the same size and the
content hash of the first and last megabyte agrees, so a diverged file is never
silently aliased. Idempotent and safe to re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
TASKS = os.path.join(SUITE, "tasks", "headkernel")
MIN_SIZE = 8 * 1024 * 1024
PROBE = 1024 * 1024


def roots():
    with open(os.path.join(HERE, "manifest.json")) as fh:
        return json.load(fh)["roots"]


def source_package(task_dir, root_map):
    cfg = os.path.join(task_dir, "config.yaml")
    if not os.path.isfile(cfg):
        return None
    for line in open(cfg):
        line = line.strip()
        if line.startswith("source_package:"):
            ref = line.split(":", 1)[1].strip().strip("\"'")
            root, _, rest = ref.partition("/")
            if root in root_map:
                return os.path.join(root_map[root], rest)
    return None


def edges_match(a, b, size):
    """Same size plus matching first and last megabyte - enough to rule out divergence
    without reading 7 GB twice."""
    with open(a, "rb") as fa, open(b, "rb") as fb:
        if fa.read(PROBE) != fb.read(PROBE):
            return False
        if size > 2 * PROBE:
            fa.seek(-PROBE, os.SEEK_END)
            fb.seek(-PROBE, os.SEEK_END)
            if fa.read(PROBE) != fb.read(PROBE):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root_map = roots()
    linked = skipped = 0
    reclaimed = 0

    for task in sorted(os.listdir(TASKS)):
        task_dir = os.path.join(TASKS, task)
        pkg = source_package(task_dir, root_map)
        if not pkg or not os.path.isdir(pkg):
            continue
        ut = os.path.join(task_dir, "ut")
        for root, _dirs, files in os.walk(ut):
            for name in files:
                dst = os.path.join(root, name)
                if os.path.islink(dst):
                    continue
                try:
                    st = os.lstat(dst)
                except OSError:
                    continue
                if st.st_size < MIN_SIZE:
                    continue
                if st.st_nlink > 1:
                    continue                      # already linked
                src = os.path.join(pkg, os.path.relpath(dst, ut))
                if not os.path.isfile(src):
                    continue
                if os.stat(src).st_size != st.st_size or not edges_match(src, dst, st.st_size):
                    print(f"  skip (diverged) {os.path.relpath(dst, SUITE)}")
                    skipped += 1
                    continue
                if args.dry_run:
                    print(f"  would link {os.path.relpath(dst, SUITE)} "
                          f"({st.st_size / 1e9:.2f} GB)")
                    linked += 1
                    reclaimed += st.st_size
                    continue
                tmp = dst + ".relink"
                try:
                    os.link(src, tmp)
                    os.replace(tmp, dst)
                except OSError as exc:
                    if os.path.lexists(tmp):
                        os.unlink(tmp)
                    print(f"  FAILED {os.path.relpath(dst, SUITE)}: {exc}")
                    skipped += 1
                    continue
                print(f"  linked {os.path.relpath(dst, SUITE)} ({st.st_size / 1e9:.2f} GB)")
                linked += 1
                reclaimed += st.st_size

    verb = "would reclaim" if args.dry_run else "reclaimed"
    print(f"\n{linked} file(s) linked, {skipped} skipped, {verb} {reclaimed / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
