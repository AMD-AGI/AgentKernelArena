#!/usr/bin/env python3
"""Static + (optional) dynamic validity checker for an extracted GPU-kernel benchmark task.

Designed for AgentKernelArena ``tasks/.../<kernel>/`` directories that follow the
``scripts/task_runner.py compile|correctness|performance`` harness convention, but
works on any task with that layout.

It answers the questions a benchmark reviewer must answer:

  1. Does the task have the expected structure (config.yaml, runner, sources)?
  2. Are the ``target_kernel_functions`` actually present in the source, and is the
     kernel real editable code (NOT just a wrapper that shells out to a binary)?
  3. Does correctness actually compute + compare against a reference (vs a no-op /
     determinism-only fallback)?
  4. Is there optimization headroom (real kernel source, perf cases that stress HBM)?
  5. What junk is shipped that shouldn't be (hipify residue, build artifacts,
     profiler dumps, __pycache__, bloated regenerable test data)?
  6. Coverage gaps: are the perf/large shapes correctness-checked? do captured
     cases have internally consistent shapes (num_seqs vs query_rows)?

STATIC checks need no GPU. Pass ``--run`` to also execute compile/correctness/
performance via the task's own config.yaml commands (needs a GPU).

Usage:
    python3 validate_task.py <task_dir>            # static report only
    python3 validate_task.py <task_dir> --run      # also run the harness
    python3 validate_task.py <task_dir> --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

# Files that are regenerable build/run output and should NOT ship in a clean task.
ARTIFACT_DIRS = ("build", ".rocprofv3", "__pycache__", ".pytest_cache")
ARTIFACT_GLOBS = ("roofline*.csv", "roofline*.tsv", "*.so", "*.o", "*.ninja_log",
                  "*.ninja_deps")
# Hipify residue: torch.cpp_extension regenerates these from the .cu/.cuh source;
# the runner deletes them at build time, so any copy in the tree is stale clutter.
RESIDUE_GLOBS = ("**/*.hip", "**/*_hip.*")
# Source extensions actually compiled by the runner glob.
SRC_EXTS = ("cu", "cpp", "cc", "cxx")


def _section(title):
    return f"\n{'='*4} {title} {'='*4}"


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def load_yaml(path):
    """Tiny YAML reader for the flat config.yaml the harness uses (avoids a pyyaml
    dependency for the static path). Falls back to pyyaml if available/needed."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        pass
    # Minimal hand parser: top-level ``key:`` then ``- item`` lists or scalars.
    cfg, cur = {}, None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if re.match(r"^\S.*:\s*$", line):
                cur = line.split(":")[0].strip(); cfg[cur] = []
            elif re.match(r"^\S.*:\s*\S", line):
                k, _, v = line.partition(":"); cfg[k.strip()] = v.strip(); cur = None
            elif line.lstrip().startswith("- ") and cur:
                cfg[cur].append(line.lstrip()[2:].strip())
    return cfg


def check_structure(task, F):
    F.append(_section("1. STRUCTURE"))
    ok = True
    for req in ("config.yaml", "scripts/task_runner.py"):
        p = os.path.join(task, req)
        present = os.path.isfile(p)
        ok &= present
        F.append(f"  [{'ok' if present else 'MISSING'}] {req}")
    cfg = {}
    cpath = os.path.join(task, "config.yaml")
    if os.path.isfile(cpath):
        cfg = load_yaml(cpath)
        for key in ("compile_command", "correctness_command", "performance_command"):
            F.append(f"  config.{key}: {cfg.get(key)}")
    return cfg


def check_sources(task, cfg, F):
    F.append(_section("2. SOURCES & TARGET KERNELS"))
    srcdir = os.path.join(task, "src")
    sources = []
    for ext in SRC_EXTS:
        sources += glob.glob(os.path.join(srcdir, "**", f"*.{ext}"), recursive=True)
    sources = sorted(sources)
    F.append(f"  compiled sources ({len(sources)}):")
    total_lines = 0
    for s in sources:
        try:
            n = sum(1 for _ in open(s, errors="ignore"))
        except Exception:
            n = 0
        total_lines += n
        F.append(f"    {os.path.relpath(s, task)}  ({n} lines)")

    # Target functions present?
    targets = cfg.get("target_kernel_functions") or []
    blob = ""
    for s in sources:
        try:
            blob += open(s, errors="ignore").read()
        except Exception:
            pass
    F.append("  target_kernel_functions:")
    for t in targets:
        found = t in blob
        F.append(f"    [{'found' if found else 'NOT FOUND'}] {t}")

    # Binary-call smell test: a benchmark must be optimizable source, not a thin
    # wrapper that shells out / dlopens a prebuilt .so (no optimization headroom).
    smells = re.findall(r"\b(system|popen|exec[lv]?[ep]*|dlopen|dlsym)\s*\(", blob)
    F.append(f"  binary-call smell (system/popen/exec/dlopen): "
             f"{'NONE - good (real kernel source)' if not smells else sorted(set(smells))}")
    F.append(f"  total kernel source lines: {total_lines} "
             f"({'plenty of optimization surface' if total_lines > 200 else 'WARN: very small - little to optimize?'})")
    return sources


def check_junk(task, F):
    F.append(_section("3. SHIPPED JUNK (should be removed)"))
    found_any = False
    for d in ARTIFACT_DIRS:
        for hit in glob.glob(os.path.join(task, "**", d), recursive=True):
            if os.path.isdir(hit):
                sz = sum(os.path.getsize(os.path.join(r, f))
                         for r, _, fs in os.walk(hit) for f in fs)
                F.append(f"  [artifact dir] {os.path.relpath(hit, task)}  ({_human(sz)})")
                found_any = True
    for g in ARTIFACT_GLOBS:
        for hit in glob.glob(os.path.join(task, "**", g), recursive=True):
            if os.path.isfile(hit) and ".rocprofv3" not in hit and "/build/" not in hit:
                F.append(f"  [artifact file] {os.path.relpath(hit, task)}  ({_human(os.path.getsize(hit))})")
                found_any = True
    for g in RESIDUE_GLOBS:
        for hit in glob.glob(os.path.join(task, "src", g), recursive=True):
            F.append(f"  [hipify residue] {os.path.relpath(hit, task)} "
                     f"(regenerated at build; stale copy is clutter)")
            found_any = True
    if not found_any:
        F.append("  none - clean")


def check_test_cases(task, F):
    F.append(_section("4. TEST CASES"))
    tc = os.path.join(task, "test_cases.json")
    if not os.path.isfile(tc):
        F.append("  no test_cases.json (compile-only task?)")
        return None
    sz = os.path.getsize(tc)
    F.append(f"  file size: {_human(sz)}" + ("  <-- WARN: large; is the data regenerable?" if sz > 20*1024*1024 else ""))
    try:
        cases = json.load(open(tc))
    except Exception as e:
        F.append(f"  FAILED to parse: {e}")
        return None
    perf = [c for c in cases if c.get("perf_only")]
    cap = [c for c in cases if not c.get("perf_only")]
    F.append(f"  cases: {len(cases)} total = {len(cap)} captured + {len(perf)} perf_only")

    # Bytes attributable to baked tensor "data" (often deterministic / regenerable).
    baked = 0
    for c in cases:
        for s in c.get("args_sig", []):
            if isinstance(s, dict) and s.get("data") is not None:
                baked += len(json.dumps(s["data"]))
    if baked > 5*1024*1024:
        F.append(f"  baked tensor 'data' ~{_human(baked)} - if it's deterministic "
                 f"(arange/disjoint block tables) store gen params instead")

    # Internal-consistency check for paged-attention-style cases: a captured case
    # whose query has N rows but block_tables/seq_lens declare num_seqs<N means the
    # reference & kernel only compute num_seqs rows; the rest compare trivially.
    F.append("  consistency (query_rows vs num_seqs) for captured cases:")
    thin = 0
    for c in cap:
        names = [n.lower() for n in (c.get("args_names") or [])]
        a = c.get("args_sig", [])
        def shp(nm):
            if nm in names:
                s = a[names.index(nm)]
                return s.get("shape")
            return None
        q = shp("query")
        bt = shp("block_tables")
        if q and bt and len(q) >= 1 and len(bt) >= 1:
            qr, ns = q[0], bt[0]
            flag = ""
            if ns < qr:
                flag = f"  <-- only {ns}/{qr} rows actually validated"
                thin += 1
            F.append(f"    {c.get('test_case_id')}: query_rows={qr} num_seqs={ns}{flag}")
    if thin:
        F.append(f"  WARN: {thin} captured case(s) validate far fewer rows than query has.")
    return {"n": len(cases), "captured": len(cap), "perf_only": len(perf)}


def check_correctness_wiring(task, F):
    """Does correctness actually compare against a reference, or fall back to a
    no-op / determinism check? Inspect the runner + _runtime references table."""
    F.append(_section("5. CORRECTNESS WIRING"))
    runner = os.path.join(task, "scripts", "task_runner.py")
    rt = os.path.join(task, "scripts", "_runtime.py")
    op = src = None
    if os.path.isfile(runner):
        txt = open(runner).read()
        m = re.search(r'OP_NAME\s*=\s*["\']([^"\']+)', txt)
        op = m.group(1) if m else None
        m = re.search(r'REF_SOURCE\s*=\s*["\']([^"\']+)', txt)
        src = m.group(1) if m else None
        skips_perf = "perf_only" in txt and "run_correctness" in txt
        F.append(f"  OP_NAME={op}  REF_SOURCE={src}")
        F.append(f"  correctness skips perf_only cases: {skips_perf}"
                 + ("  <-- the large/optimized shapes are NOT correctness-checked" if skips_perf else ""))
    has_ref = False
    if os.path.isfile(rt) and op:
        rtxt = open(rt).read()
        keys = [f"{src}:{op}", op]
        for k in keys:
            if f'"{k}"' in rtxt or f"'{k}'" in rtxt:
                has_ref = True
                F.append(f"  reference registered for '{k}': YES (real compute+compare)")
                break
        if not has_ref:
            F.append(f"  reference for op '{op}': NONE -> falls back to DETERMINISM-ONLY "
                     f"check (does not verify numerical correctness!)")
    return op, src, has_ref


def run_harness(task, cfg, F):
    F.append(_section("6. DYNAMIC RUN (compile / correctness / performance)"))
    for key in ("compile_command", "correctness_command", "performance_command"):
        cmds = cfg.get(key) or []
        cmd = cmds[0] if isinstance(cmds, list) and cmds else cmds
        if not cmd:
            F.append(f"  {key}: (none)")
            continue
        F.append(f"  $ {cmd}")
        try:
            p = subprocess.run(cmd, shell=True, cwd=task, capture_output=True,
                               text=True, timeout=1800)
            tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-12:])
            F.append(f"    exit={p.returncode}")
            for ln in tail.splitlines():
                F.append(f"    | {ln}")
        except subprocess.TimeoutExpired:
            F.append("    TIMEOUT (>1800s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--run", action="store_true", help="also execute the harness (needs GPU)")
    ap.add_argument("--json", help="write a machine-readable summary here")
    args = ap.parse_args()
    task = os.path.abspath(args.task_dir)
    F = [f"BENCHMARK TASK VALIDITY REPORT\ntask: {task}"]

    cfg = check_structure(task, F)
    check_sources(task, cfg, F)
    check_junk(task, F)
    tc = check_test_cases(task, F)
    check_correctness_wiring(task, F)
    if args.run:
        run_harness(task, cfg, F)

    report = "\n".join(F)
    print(report)
    if args.json:
        json.dump({"task": task, "test_cases": tc, "report": report},
                  open(args.json, "w"), indent=2)
        print(f"\n[wrote {args.json}]")


if __name__ == "__main__":
    main()
