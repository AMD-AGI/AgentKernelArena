#!/usr/bin/env python3
"""Validate every task in this suite against the AgentKernelArena task standard.

Implements the static half of the ten checks in
``AgentKernelArena/agents/task_validator/README.md``. Checks 4/5/6/9 need a GPU
and the model's own image, so they are reported as SKIP here and are the job of
``tools/run_on_gpu.sh``.

    python3 tools/validate_suite.py             # table + per-task verdicts
    python3 tools/validate_suite.py --json r.json
    python3 tools/validate_suite.py --task <id> --verbose

Note on the arena's own ``benchmark-task-validator/scripts/validate_task.py``:
its source scan globs ``<task>/src/**/*.{cu,cpp,cc,cxx}``, so for any Python-kernel
task -- including every ``triton2triton`` task already shipped in the arena -- it
reports zero sources and "NOT FOUND" for every target symbol. That is a limitation
of the HIP-oriented script, not a property of the task; this checker resolves
symbols in the Python sources the same way the task_validator agent is told to
(``def <name>`` / ``@triton.jit``).
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
TASKS = os.path.join(SUITE, "tasks", "headkernel")

VALID_TASK_TYPES = {"hip2hip", "cuda2hip", "triton2triton", "torch2hip",
                    "instruction2triton", "rocprim"}
REQUIRED = ("source_file_path", "target_kernel_functions", "compile_command",
            "correctness_command", "task_type")
JUNK_DIRS = ("build", ".rocprofv3", "__pycache__", ".pytest_cache")
JUNK_GLOBS = ("*.so", "*.o", "*.pyc", "*.ninja_log", "roofline*.csv", "roofline*.tsv")
# Names that would mean the task ships a previous answer.
LEAK_NAMES = ("_candidate_best", "accepted_overlay", "_cand_overlay", "seam_patch",
              "final.patch", "overlay.tar.gz")


def load_yaml(path):
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh)
    except ImportError:
        raise SystemExit("pyyaml required: pip install pyyaml")


def _ok_json(path):
    """True when <path> is a JSON verdict file that AFFIRMATIVELY says it passed.

    Packages spell the verdict three ways, so all three are accepted -- but the
    absence of a negative is not a pass. An earlier version fell back to "every
    top-level dict value has ok=True", which is vacuously true for a schema whose
    top-level values are all lists: a selection_validation.json with
    ``"ok": false`` and every ``process_verdicts[*].ok`` false still gated PASS.
    """
    if not os.path.isfile(path):
        return False
    try:
        blob = json.load(open(path))
    except Exception:
        return False
    if not isinstance(blob, dict):
        return False
    # Any negative anywhere in the document loses, top-level aggregate or not: a
    # file whose per-case verdicts all failed must not pass because the summary
    # line still says ok.
    for k, v in _walk(blob):
        if k == "ok" and v is False:
            return False
        if k == "status" and isinstance(v, str) and v.upper() not in ("PASS", "OK", "IN_PROGRESS"):
            return False
    if blob.get("ok") is True or str(blob.get("status", "")).upper() in ("PASS", "OK"):
        return True
    # No top-level verdict: require at least one per-case verdict, all of them good.
    candidates = [v for _, v in _walk(blob)
                  if isinstance(v, dict) and ("ok" in v or "status" in v)]
    return bool(candidates)


def _walk(node):
    """Every (key, value) pair anywhere in a nested JSON document."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _rejects_corruption(path):
    """True when the package's negative control records that it caught a corruption.

    Existence alone proves nothing: an empty or failing negative_check.json would
    otherwise gate a task to PASS while its oracle cannot detect a wrong answer.

    The key is never plain "rejected" -- the packages name what was corrupted
    (``output_corruption_rejected``, ``normed_corruption_rejected``,
    ``pre_norm_sum_corruption_rejected``) and may nest it one level under a
    per-case dict. So walk the whole document, collect every *_rejected flag, and
    require at least one and all of them true.
    """
    if not os.path.isfile(path):
        return False
    try:
        blob = json.load(open(path))
    except Exception:
        return False
    flags = [v for k, v in _walk(blob)
             if k.endswith("rejected") and isinstance(v, bool)]
    statuses = [str(v).upper() for k, v in _walk(blob) if k == "status"]
    if any(s in ("FAIL", "ERROR") for s in statuses):
        return False
    return bool(flags) and all(flags)


def is_task_dir(path):
    """The arena only treats a directory with a config.yaml (or an explicit
    NOT_BUILT marker) as a task. Anything else under tasks/headkernel/ is stray
    output - the callable UTs used to drop a reports/ledger/ tree there - and
    must not be scanned, or one stray write FAILs the whole suite."""
    return os.path.isdir(path) and (os.path.isfile(os.path.join(path, "config.yaml"))
                                    or os.path.isfile(os.path.join(path, "NOT_BUILT")))


def defined_symbols(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError as exc:
        return None, f"syntax error line {exc.lineno}: {exc.msg}"
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.setdefault(node.name, node.lineno)
    return names, None


def check_task(task_dir):
    """Returns (per-check status dict, list of detail strings)."""
    C, D = {}, []
    name = os.path.basename(task_dir)
    cfg_path = os.path.join(task_dir, "config.yaml")

    if os.path.isfile(os.path.join(task_dir, "NOT_BUILT")):
        return {"_notbuilt": True}, ["NOT_BUILT placeholder - no config.yaml, not scanned as a task"]

    # ---- 1 config schema
    if not os.path.isfile(cfg_path):
        return {"config_schema": "FAIL"}, ["config.yaml missing"]
    cfg = load_yaml(cfg_path)
    problems = []
    for key in REQUIRED:
        if key not in cfg:
            problems.append(f"missing {key}")
        elif key == "task_type":
            if cfg[key] not in VALID_TASK_TYPES:
                problems.append(f"task_type {cfg[key]!r} not in {sorted(VALID_TASK_TYPES)}")
        elif not isinstance(cfg[key], list) or not cfg[key]:
            problems.append(f"{key} must be a non-empty list")
    if "performance_command" in cfg and not isinstance(cfg["performance_command"], list):
        problems.append("performance_command must be a list")
    C["config_schema"] = "PASS" if not problems else "FAIL"
    D += [f"config_schema: {p}" for p in problems]

    # ---- 2 source files exist
    sources = cfg.get("source_file_path") or []
    missing = [s for s in sources if not os.path.isfile(os.path.join(task_dir, s))]
    C["source_files_exist"] = "PASS" if not missing else "FAIL"
    D += [f"source_files_exist: missing {m}" for m in missing]

    # ---- 3 target symbols defined in those sources
    targets = cfg.get("target_kernel_functions") or []
    table, syntax_errors = {}, []
    for s in sources:
        p = os.path.join(task_dir, s)
        if not os.path.isfile(p) or not p.endswith(".py"):
            continue
        names, err = defined_symbols(p)
        if err:
            syntax_errors.append(f"{s}: {err}")
            continue
        for n, line in names.items():
            table.setdefault(n, f"{s}:{line}")
    not_found = [t for t in targets if t not in table]
    C["target_symbols_found"] = "PASS" if not (not_found or syntax_errors) else "FAIL"
    D += [f"target_symbols_found: {t} NOT DEFINED in source_file_path" for t in not_found]
    D += [f"target_symbols_found: {e}" for e in syntax_errors]

    # ---- 4/5/6/9 need a GPU
    for k in ("compilation", "correctness", "performance", "gpu_hang_check"):
        C[k] = "SKIP"

    # ---- 7 correctness implementation review (static)
    ut = os.path.join(task_dir, "ut")
    has_ut = os.path.isfile(os.path.join(ut, "unittest.py"))
    has_oracle = os.path.isfile(os.path.join(ut, "reference_io.pt"))
    meta_p = os.path.join(ut, "meta.json")
    wl_p = os.path.join(ut, "workload.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {}
    wl = json.load(open(wl_p)) if os.path.isfile(wl_p) else {}
    # Packages spell the GEOMETRY list in one of four places depending on which
    # generation of the capture tool produced them. Deliberately NOT counted:
    # meta.ledger_ids and meta.case_contracts, which are per-evidence-row ids -
    # three ledger rows can all describe one shape (the MoE package gates stage 1
    # and stage 2 of the same M=64 call separately), and counting them would
    # promote a single-geometry task to PASS.
    ncases = max(len(meta.get("cases") or []),
                 len(meta.get("case_specs") or []),
                 len(wl.get("cases") or []),
                 len((meta.get("workload") or {}).get("cases") or []))
    # A package may legitimately carry no reference_io.pt: the 2026-09-13
    # callable UTs regenerate a deterministic frozen baseline at runtime instead
    # of freezing hundreds of MB of tensors. That is a real oracle when the
    # package also proves the two legs are distinct (selection_validation) and
    # that a corrupted output is rejected (negative_check).
    runtime_oracle = bool(meta.get("oracle_policy")) and not meta.get("reference_io_sha256")
    gated = (_ok_json(os.path.join(ut, "selection_validation.json"))
             and _rejects_corruption(os.path.join(ut, "negative_check.json")))
    if not has_ut:
        C["correctness_implementation_review"] = "FAIL"
        D.append("correctness_implementation_review: no ut/unittest.py - nothing compares anything")
    elif has_oracle and ncases >= 2:
        C["correctness_implementation_review"] = "PASS"
        D.append(f"correctness_implementation_review: frozen live-capture oracle + "
                 f"{ncases} case geometries, tol={meta.get('tol')}")
    elif runtime_oracle and gated and ncases >= 2:
        C["correctness_implementation_review"] = "PASS"
        D.append(f"correctness_implementation_review: runtime frozen-baseline oracle "
                 f"({meta.get('oracle_policy', '')[:60]}...) over {ncases} case geometries, "
                 f"with a passing selection_validation and a corruption negative control")
    elif has_oracle:
        C["correctness_implementation_review"] = "WARN"
        D.append(f"correctness_implementation_review: frozen oracle present but meta.cases has "
                 f"{ncases} entry - the standard asks for 2-3 representative shapes")
    else:
        C["correctness_implementation_review"] = "WARN"
        D.append("correctness_implementation_review: no frozen reference_io.pt - "
                 "read the package README for what the oracle actually is")

    # ---- 7b the files the harness says it needs must actually be there
    # Three MiniMax tasks declare `geometry_file: timing_geometry.pt` (and record
    # its sha256) and their cases.py torch.load()s it unconditionally -- but the
    # blob exists in no upstream package, so the task cannot run at all while
    # every static check passes. A benchmark that validates green and dies on the
    # GPU is worse than one that says up front that it is incomplete.
    declared = []
    for key, val in (meta or {}).items():
        if not isinstance(val, str) or "/" in val or not val:
            continue
        if val.endswith((".pt", ".json", ".py", ".csv", ".safetensors")):
            if not os.path.isfile(os.path.join(ut, val)):
                declared.append(f"{key} -> ut/{val}")
    C["declared_files_present"] = "PASS" if not declared else "FAIL"
    D += [f"declared_files_present: ut/meta.json names {d}, which does not exist"
          for d in declared]

    # ---- 8 self-contained
    ext = []
    for s in sources:
        p = os.path.join(task_dir, s)
        if not os.path.isfile(p):
            continue
        for i, line in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
            if "/shared_nfs/" in line or "/sgl-workspace/" in line:
                if line.lstrip().startswith("#"):
                    continue                      # a comment, not a dependency
                ext.append(f"{s}:{i} hardcoded path")
    for root, dirs, files in os.walk(task_dir):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.islink(fp):
                tgt = os.readlink(fp)
                if os.path.isabs(tgt):
                    ext.append(f"absolute symlink {os.path.relpath(fp, task_dir)} -> {tgt}")
                elif not os.path.realpath(fp).startswith(os.path.realpath(task_dir) + os.sep):
                    ext.append(f"symlink escapes the task: {os.path.relpath(fp, task_dir)} -> {tgt}")
    C["self_contained"] = "PASS" if not ext else "FAIL"
    D += [f"self_contained: {e}" for e in ext[:8]]

    # ---- 10 result template compatibility
    runner = os.path.join(task_dir, "scripts", "task_runner.py")
    ok10 = os.path.isfile(runner)
    if ok10:
        txt = open(runner).read()
        ok10 = all(k in txt for k in ("compile_report.json", "correctness_report.json",
                                      "performance_report.json", "execution_time_ms"))
    C["result_template_compatibility"] = "PASS" if ok10 else "FAIL"
    if not ok10:
        D.append("result_template_compatibility: runner does not emit the three build/*.json reports")

    # ---- extra: shipped junk / leaked answers (arena validator section 3)
    junk = []
    for d in JUNK_DIRS:
        junk += [os.path.relpath(h, task_dir)
                 for h in glob.glob(os.path.join(task_dir, "**", d), recursive=True)
                 if os.path.isdir(h)]
    for g in JUNK_GLOBS:
        junk += [os.path.relpath(h, task_dir)
                 for h in glob.glob(os.path.join(task_dir, "**", g), recursive=True)
                 if os.path.isfile(h)]
    leaks = [os.path.relpath(h, task_dir)
             for n in LEAK_NAMES
             for h in glob.glob(os.path.join(task_dir, "**", n), recursive=True)]
    C["no_shipped_junk"] = "PASS" if not junk else "WARN"
    C["no_leaked_solution"] = "PASS" if not leaks else "FAIL"
    D += [f"no_shipped_junk: {j}" for j in junk[:6]]
    D += [f"no_leaked_solution: {l}" for l in leaks[:6]]

    return C, D


def verdict(C):
    if C.get("_notbuilt"):
        return "NOT_BUILT"
    vals = [v for k, v in C.items() if not k.startswith("_")]
    if "FAIL" in vals:
        return "FAIL"
    if "WARN" in vals:
        return "WARN"
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results = {}
    names = sorted(n for n in os.listdir(TASKS) if is_task_dir(os.path.join(TASKS, n)))
    stray = sorted(n for n in os.listdir(TASKS) if not is_task_dir(os.path.join(TASKS, n)))
    if args.task:
        names = [n for n in names if n == args.task]
    if not names:
        raise SystemExit(f"no task directories under {TASKS}"
                         + (f" (matching {args.task!r})" if args.task else ""))

    width = max(len(n) for n in names) + 2
    print(f"{'task':{width}} verdict   checks")
    print("-" * (width + 60))
    for n in names:
        C, D = check_task(os.path.join(TASKS, n))
        v = verdict(C)
        results[n] = {"verdict": v, "checks": C, "details": D}
        if v == "NOT_BUILT":
            print(f"{n:{width}} {'-':9} (placeholder, not a task)")
            continue
        summary = " ".join(f"{k.split('_')[0][:4]}={v2}" for k, v2 in C.items()
                           if v2 in ("FAIL", "WARN")) or "all static checks pass"
        print(f"{n:{width}} {v:9} {summary}")
        if args.verbose:
            for d in D:                            # every rationale, PASS included
                print(f"{'':{width}}   - {d}")
        elif v in ("FAIL", "WARN"):
            for d in D:
                if any(d.startswith(k) for k, s in C.items() if s in ("FAIL", "WARN")):
                    print(f"{'':{width}}   - {d}")

    built = {k: v for k, v in results.items() if v["verdict"] != "NOT_BUILT"}
    tally = {}
    for r in built.values():
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print(f"\n{len(built)} task(s): " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    print(f"{len(results) - len(built)} NOT_BUILT placeholder(s)")
    if stray:
        print(f"\nFAIL: {len(stray)} entr(y/ies) under tasks/headkernel are neither a task nor a "
              f"NOT_BUILT placeholder: {', '.join(stray)}")
        print("      A shipped suite must not have stray output in the directory the arena scans.")
        print("      Run tools/clean.sh - the UTs write a reports/ledger tree at run time.")
    print("\ncompilation / correctness / performance / gpu_hang are SKIP here - "
          "they need a GPU and the model's image (tools/run_on_gpu.sh).")

    if args.json:
        json.dump(results, open(args.json, "w"), indent=2)
        print(f"[wrote {args.json}]")
    return 1 if (stray or any(r["verdict"] == "FAIL" for r in built.values())) else 0


if __name__ == "__main__":
    sys.exit(main())
