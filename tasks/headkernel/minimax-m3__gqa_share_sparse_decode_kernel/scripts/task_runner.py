#!/usr/bin/env python3
"""AgentKernelArena task runner for a head-kernel benchmark task.

Shared by the eight candidate-bound non-Qwen tasks; task-specific details come
from ``config.yaml``. Three modes, the standard arena contract:

    python3 scripts/task_runner.py compile
    python3 scripts/task_runner.py correctness
    python3 scripts/task_runner.py performance

compile      AST-parses every ``source_file_path`` and asserts each
             ``target_kernel_functions`` entry is *defined* there (a real symbol
             table lookup, not a text search).
correctness  Delegates to the frozen GEAK op unittest under ``ut/`` -- captured
             oracle plus random-value parity against the live baseline leg, with
             whatever extra contracts that op needs (physical stride, graph
             replay, elementwise-median repair).
performance  Times the op with the arena's methodology: 10 warmup + 100 measured
             iterations, reported as the mean. Falls back to the GEAK
             interleaved median-of-3 when the op's oracle does not carry
             replayable argument records (recorded in the report either way).

Exit 0 pass, non-zero fail. Reports land in ``build/``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import runpy
import subprocess
import sys
import time
import uuid

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(TASK_DIR, "build")
UT_DIR = os.path.join(TASK_DIR, "ut")
CONFIG = os.path.join(TASK_DIR, "config.yaml")

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
EXIT_NO_RECORDS = 4


# --------------------------------------------------------------------------- config
def load_config():
    with open(CONFIG) as fh:
        text = fh.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        pass
    # Minimal fallback parser for the schema this suite uses, so the task still
    # runs in an image without pyyaml. It handles the top level plus ONE nested
    # level under `headkernel:`, because the runner reads preserve_symbols, tol
    # and oracle out of that block; skipping it used to silently disable the
    # preserve_symbols check and blank the correctness report's oracle fields.
    cfg, key, indent = {}, None, 0
    block, bkey, bindent = None, None, None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        m = re.match(r"^(\s*)([A-Za-z_][\w.]*):\s*(.*)$", raw)
        pad = len(raw) - len(raw.lstrip())

        if block is not None and (stripped.startswith("- ") or (m and pad > indent)):
            if stripped.startswith("- "):
                if bkey is not None and bindent is not None and pad > bindent:
                    block.setdefault(bkey, []).append(stripped[2:].strip().strip("'\""))
                continue
            name, val = m.group(2), m.group(3).strip()
            if bindent is not None and pad > bindent:
                continue                       # two levels deep - not read by the runner
            bkey, bindent = name, pad
            block[name] = val.strip("'\"") if val else []
            continue

        if stripped.startswith("- ") and key:
            cfg.setdefault(key, []).append(stripped[2:].strip().strip("'\""))
            continue
        if not m:
            continue
        name, val = m.group(2), m.group(3).strip()
        if pad > indent and key:
            continue                           # prompt: and other nested blocks
        indent, key = pad, name
        if name == "headkernel" and not val:
            block, bkey, bindent = {}, None, None
            cfg[name] = block
            continue
        block = None
        cfg[name] = val.strip("'\"") if val else []
    return cfg


def hk(cfg, field, default=None):
    """Read a field from the task's ``headkernel:`` provenance block."""
    block = cfg.get("headkernel")
    if isinstance(block, dict):
        return block.get(field, default)
    return default


def write_report(name, payload):
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(os.path.join(BUILD_DIR, name), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


# --------------------------------------------------------------------------- compile
def defined_symbols(path):
    """Top-level names bound by a Python module, via the AST -- not a grep."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read(), filename=path)
    except SyntaxError as exc:
        raise
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def run_compile(cfg):
    sources = cfg.get("source_file_path") or []
    targets = cfg.get("target_kernel_functions") or []
    found, missing, parsed = {}, [], []
    all_names = set()

    for rel in sources:
        path = os.path.join(TASK_DIR, rel)
        if not os.path.isfile(path):
            return False, f"source file missing: {rel}"
        try:
            names = defined_symbols(path)
        except SyntaxError as exc:
            return False, f"{rel}: syntax error at line {exc.lineno}: {exc.msg}"
        parsed.append(rel)
        all_names |= names
        for t in targets:
            if t in names:
                found.setdefault(t, rel)

    for t in targets:
        if t not in found:
            missing.append(t)

    # Some packages wire their two-leg UT through an extra symbol in the same
    # file (e.g. the fused-MoE overlay resolves its frozen baseline through
    # `baseline_callable`). Deleting it turns a working kernel into an opaque
    # correctness FAIL on the GPU, so catch it here, without one.
    preserve = hk(cfg, "preserve_symbols") or []
    dropped = [s for s in preserve if s not in all_names]

    # Some load-bearing lines are not a def or a class and so are invisible to the
    # symbol table -- e.g. the GLM-5.2 MoE package appends
    # `fused_moe_.__module__ = __name__`, without which the harness's leg-identity
    # probe cannot tell the two legs apart and refuses to measure at all.
    preserve_txt = hk(cfg, "preserve_text") or []
    if preserve_txt:
        blob = "".join(open(os.path.join(TASK_DIR, rel), encoding="utf-8", errors="ignore").read()
                       for rel in sources if os.path.isfile(os.path.join(TASK_DIR, rel)))
        dropped += [t for t in preserve_txt if t not in blob]

    write_report("compile_report.json", {
        "status": "ok" if not (missing or dropped) else "fail",
        "error": None if not (missing or dropped) else
                 f"target symbols not defined in source: {missing}"
                 if missing else
                 f"harness symbols removed from source: {dropped}",
        "parsed_sources": parsed,
        "symbols_found": found,
        "symbols_missing": missing,
        "harness_symbols_required": list(preserve) + list(preserve_txt),
        "harness_symbols_missing": dropped,
    })
    if missing:
        return False, f"target symbols not defined in source: {missing}"
    if dropped:
        return False, (f"these symbols must survive in source/ - the UT resolves its "
                       f"frozen baseline through them: {dropped}")
    return True, None


# --------------------------------------------------------------------------- correctness
def _ut_command(binding):
    return [sys.executable, "-u", os.path.abspath(__file__),
            "_correctness_child", json.dumps(binding)]


def _source_sha256(path):
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def _correctness_child(binding):
    """Attest the installed candidate, then run the immutable UT in this process.

    Python has already loaded the selected overlay's sitecustomize. Restore the
    original environment before imports or UT execution: baseline_random_outputs
    and measure_legs prepend their own overlays in fresh subprocesses and must
    not inherit the candidate path. The current interpreter keeps its installed
    candidate module/function and its import path.
    """
    baseline_path = binding["baseline_pythonpath"]
    if baseline_path:
        os.environ["PYTHONPATH"] = baseline_path
    else:
        os.environ.pop("PYTHONPATH", None)
    receipt = {"run_id": binding["run_id"], "status": "fail",
               "target": binding["target"], "source_sha256": binding["source_sha256"],
               "baseline_environment_restored": True}
    try:
        module, attr = binding["target"].split(":", 1)
        function = importlib.import_module(module)
        for part in attr.split("."):
            function = getattr(function, part)
        # torch.no_grad and similar transparent decorators retain __wrapped__.
        # Checking only __module__ would credit a stale/rebound production object.
        function = inspect.unwrap(function)
        code = getattr(function, "__code__", None)
        if code is None:
            raise RuntimeError("candidate target has no attestable Python implementation")
        resolved = os.path.realpath(code.co_filename)
        expected = os.path.realpath(binding["overlay_source"])
        if resolved != expected:
            raise RuntimeError(f"candidate target resolved to {resolved}, expected {expected}")
        if (_source_sha256(resolved) != binding["source_sha256"]
                or _source_sha256(binding["source"]) != binding["source_sha256"]):
            raise RuntimeError("candidate overlay/source differs from this run's source bytes")
        receipt.update(status="ok", resolved_source=resolved,
                       candidate_source=os.path.realpath(binding["source"]))
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_report("correctness_binding.json", receipt)
        print("Candidate binding: FAIL - " + receipt["error"], file=sys.stderr)
        return 3
    write_report("correctness_binding.json", receipt)
    # Match script execution while keeping this same attested interpreter. The
    # task removes ut/ before importing torch, avoiding its unittest.py name clash.
    sys.argv = [os.path.join(UT_DIR, "unittest.py")]
    sys.path.insert(0, UT_DIR)
    completion = {"run_id": binding["run_id"], "status": "fail",
                  "target": binding["target"], "source_sha256": binding["source_sha256"],
                  "unittest_sha256": binding["unittest_sha256"]}
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        # A kernel raising SystemExit(0) must not impersonate the UT's normal
        # sys.exit(main()). The originating frame must be the immutable UT.
        origin = exc.__traceback__
        while origin.tb_next is not None:
            origin = origin.tb_next
        if os.path.realpath(origin.tb_frame.f_code.co_filename) != os.path.realpath(os.path.join(UT_DIR, "unittest.py")):
            completion["error"] = "early SystemExit(0) did not originate from unittest.py"
            write_report("correctness_completion.json", completion)
            return 3
    if (_source_sha256(binding["source"]) != binding["source_sha256"]
            or _source_sha256(binding["overlay_source"]) != binding["source_sha256"]
            or _source_sha256(os.path.join(UT_DIR, "unittest.py")) != binding["unittest_sha256"]):
        completion["error"] = "candidate or unittest source changed during correctness"
        write_report("correctness_completion.json", completion)
        return 3
    # Written only after the real UT returns or exits successfully. os._exit(0)
    # bypasses this point and therefore cannot produce an accepted completion.
    completion["status"] = "complete"
    write_report("correctness_completion.json", completion)
    return 0


def run_ut(timeout):
    """Run the unchanged UT with an attested candidate and independent baseline."""
    t0 = time.monotonic()
    os.makedirs(BUILD_DIR, exist_ok=True)
    binding_report = os.path.join(BUILD_DIR, "correctness_binding.json")
    completion_report = os.path.join(BUILD_DIR, "correctness_completion.json")
    for report in (binding_report, completion_report):
        try:
            os.unlink(report)
        except FileNotFoundError:
            pass
    with open(os.path.join(UT_DIR, "meta.json")) as handle:
        meta = json.load(handle)
    bind = meta.get("candidate_bind") or {}
    if not bind.get("file") or bind.get("kind") not in ("module", "rebind"):
        raise ValueError("correctness requires a complete candidate_bind declaration")
    source = os.path.realpath(os.path.join(UT_DIR, bind["file"]))
    if os.path.commonpath([source, os.path.realpath(TASK_DIR)]) != os.path.realpath(TASK_DIR):
        raise ValueError("candidate source escapes the isolated task")
    source_hash = _source_sha256(source)
    cand = candidate_overlay()
    if not cand:
        raise RuntimeError("candidate overlay unavailable; refusing baseline-only correctness")
    overlay_source = (os.path.join(cand, "_patched", bind["module"] + ".py")
                      if bind["kind"] == "module"
                      else os.path.join(cand, os.path.basename(bind["file"])))
    if _source_sha256(overlay_source) != source_hash:
        raise RuntimeError("candidate overlay does not contain this run's source")
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Remove even a pre-existing copy of this candidate path before preserving
    # the caller's other imports for explicit baseline subprocesses.
    clean_path = os.pathsep.join(p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                if p and os.path.realpath(p) != os.path.realpath(cand))
    binding = {"run_id": uuid.uuid4().hex, "target": meta["target_callable"],
               "source": source, "source_sha256": source_hash,
               "unittest_sha256": _source_sha256(os.path.join(UT_DIR, "unittest.py")),
               "overlay_source": os.path.realpath(overlay_source),
               "baseline_pythonpath": clean_path}
    env["PYTHONPATH"] = os.pathsep.join([cand] + ([clean_path] if clean_path else []))
    remaining = timeout - (time.monotonic() - t0)
    if remaining <= 0:
        raise subprocess.TimeoutExpired(_ut_command(binding), timeout)
    proc = subprocess.run(_ut_command(binding), cwd=TASK_DIR, env=env, capture_output=True,
                          text=True, timeout=remaining)
    try:
        with open(binding_report) as handle:
            receipt = json.load(handle)
        valid = (receipt.get("status") == "ok" and receipt.get("run_id") == binding["run_id"]
                 and receipt.get("target") == binding["target"]
                 and receipt.get("source_sha256") == source_hash
                 and receipt.get("resolved_source") == binding["overlay_source"]
                 and receipt.get("baseline_environment_restored") is True
                 and _source_sha256(source) == source_hash)
    except (OSError, ValueError, AttributeError):
        receipt, valid = {"status": "fail", "error": "no valid candidate binding receipt"}, False
    if not valid:
        proc.returncode = proc.returncode or 3
        proc.stderr += "\nCandidate binding failed; no correctness result is valid.\n"
    proc.candidate_binding = receipt
    try:
        with open(completion_report) as handle:
            completion = json.load(handle)
        completed = (completion.get("status") == "complete"
                     and all(completion.get(key) == binding[key]
                             for key in ("run_id", "target", "source_sha256", "unittest_sha256"))
                     and _source_sha256(binding["overlay_source"]) == source_hash
                     and _source_sha256(os.path.join(UT_DIR, "unittest.py")) == binding["unittest_sha256"])
    except (OSError, ValueError, AttributeError):
        completion, completed = {"status": "missing", "error": "no valid post-UT completion receipt"}, False
    if proc.returncode == 0 and not completed:
        proc.returncode = 3
        proc.stderr += "\nUT completion failed; early exit cannot produce a correctness PASS.\n"
    proc.correctness_completion = completion
    return proc, time.monotonic() - t0


def run_correctness(cfg, timeout):
    if not os.path.isfile(os.path.join(UT_DIR, "unittest.py")):
        return False, "ut/unittest.py missing - this task has no correctness oracle"
    try:
        proc, secs = run_ut(timeout)
    except subprocess.TimeoutExpired:
        write_report("correctness_report.json",
                     {"status": "fail", "error": f"unittest.py timed out after {timeout}s"})
        return False, f"unittest.py timed out after {timeout}s"
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        error = f"candidate binding setup failed: {type(exc).__name__}: {exc}"
        write_report("correctness_report.json", {"status": "fail", "error": error,
                                                "candidate_binding": {"status": "fail"}})
        return False, error

    out = proc.stdout + proc.stderr
    # The GEAK driver's exit code IS the contract (0 pass, 1 correctness FAIL,
    # 2 environment, 3 harness incomplete). Do not additionally require a bare
    # "PASS" line: packages spell the verdict differently ("PASS",
    # "RESULT PASS (oracle=True random_parity=True)", or only the
    # GEAK_WEIGHTED_SPEEDUP block), and demanding one spelling reports a passing
    # kernel as a failure.
    ok = proc.returncode == 0
    # Per-case verdict lines. Packages spell these several ways: "[correct:...]",
    # "[oracle       ] FAIL err=...", "[sequence     ] PASS err=...", plus the
    # GEAK summary markers. Matching only "[correct:" meant a failing run recorded
    # zero checks and the report showed nothing but unrelated aiter chatter, which
    # made a real correctness FAIL impossible to diagnose from the archive.
    checks = [ln for ln in out.splitlines()
              if ln.startswith(("RESULT ", "GEAK_WEIGHTED_SPEEDUP", "GEAK_GEOMEAN_SPEEDUP",
                                "CORRECTNESS"))
              or (ln.startswith("[") and ("PASS" in ln or "FAIL" in ln or "err=" in ln))]
    verdict = next((ln for ln in reversed(out.splitlines())
                    if ln.strip() in ("PASS", "FAIL") or ln.startswith("RESULT ")), "")
    exit_meaning = {0: "pass", 1: "correctness FAIL", 2: "environment",
                    3: "harness incomplete (regenerate the UT)"}
    meaning = exit_meaning.get(proc.returncode, "killed by signal")
    # A UT that compared nothing did not fail a comparison. Some packages leave a
    # call outside their try/except (GLM-5.3's baseline_random_outputs is one), so
    # an environment problem escapes as a bare traceback and Python exits 1, which
    # the table above would label "correctness FAIL". Zero checks plus a traceback
    # is the signature of a run that never got as far as checking anything.
    if (proc.returncode == 1 and not checks
            and ("Traceback (most recent call last)" in out or "Error:" in out)):
        meaning = "environment (uncaught harness exception - the UT compared nothing)"
    write_report("correctness_report.json", {
        "status": "ok" if ok else "fail",
        "error": None if ok else
                 f"ut/unittest.py exit={proc.returncode} ({meaning})",
        "exit_code": proc.returncode,
        "exit_meaning": meaning,
        "candidate_binding": getattr(proc, "candidate_binding", None),
        "correctness_completion": getattr(proc, "correctness_completion", None),
        "ut_verdict_line": verdict,
        "duration_seconds": round(secs, 2),
        "oracle": hk(cfg, "oracle", "frozen live-capture (ut/reference_io.pt)"),
        "tolerance": hk(cfg, "tol"),
        "num_checks": len(checks),
        "checks": checks[:64],
        # Keep enough context to diagnose a failure from the archive alone. 20 lines
        # was routinely all aiter dispatch chatter with the verdict scrolled off.
        "stdout_tail": out.strip().splitlines()[-120:],
    })
    if not ok:
        return False, f"ut/unittest.py exit={proc.returncode}: {out.strip().splitlines()[-3:]}"
    return True, None


# --------------------------------------------------------------------------- performance
def candidate_overlay():
    """Build the GEAK candidate overlay so the timed callable resolves to source/.

    Returns the overlay dir, or None when this op has no rebind seam (then the
    live stack is timed directly, which is also what its own UT does).
    """
    meta_path = os.path.join(UT_DIR, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    meta = json.load(open(meta_path))
    if not (meta.get("candidate_bind") or {}).get("file"):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geak_harness_lib", os.path.join(UT_DIR, "harness_lib.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    _base, cand = h.build_candidate_overlay(UT_DIR, meta)
    return cand


def _benchmark_cases(raw):
    """Reject incomplete or invalid native reports before publishing any case."""
    if not isinstance(raw, dict):
        raise ValueError("benchmark report must be a mapping")
    if raw.get("timer") != "cuda_event":
        raise ValueError("benchmark report must use cuda_event timing")
    rows = raw.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark report has no cases")
    if any(type(raw.get(key)) is not int or raw[key] != expected
           for key, expected in (("warmup", WARMUP_ITERATIONS), ("iters", BENCHMARK_ITERATIONS))):
        raise ValueError("benchmark report has incorrect iteration counts")
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("sig"), str) or not row["sig"]
                or not isinstance(row.get("params", {}), dict)
                or any(type(row.get(key)) not in (int, float)
                       or not math.isfinite(row[key]) or row[key] <= 0
                       for key in ("mean_ms", "median_ms", "min_ms"))):
            raise ValueError("benchmark report has an invalid case or timing")
    return [{
        "test_case_id": row["sig"], "execution_time_ms": row["mean_ms"],
        "params": row.get("params", {}), "median_ms": row["median_ms"], "min_ms": row["min_ms"],
    } for row in rows]


def run_performance(cfg, timeout):
    bench = os.path.join(TASK_DIR, "scripts", "_bench.py")
    out_json = os.path.join(BUILD_DIR, "_bench_raw.json")
    os.makedirs(BUILD_DIR, exist_ok=True)
    try:
        os.unlink(out_json)
    except FileNotFoundError:
        pass

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    overlay_note = "none (timed against the live stack, as the op's own UT does)"
    meta_path = os.path.join(UT_DIR, "meta.json")
    wants_overlay = False
    if os.path.isfile(meta_path):
        try:
            wants_overlay = bool((json.load(open(meta_path)).get("candidate_bind") or {}).get("file"))
        except Exception:
            wants_overlay = False
    try:
        cand = candidate_overlay()
        if cand:
            env["PYTHONPATH"] = os.pathsep.join(
                [cand] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            overlay_note = os.path.relpath(cand, TASK_DIR)
    except Exception as exc:
        cand = None
        overlay_note = f"overlay build FAILED ({exc!r})"
    if wants_overlay and not cand:
        # This op is only reachable through a rebind. Without the overlay the
        # timed callable is the unmodified production stack, so any number we
        # printed would describe code the optimizer never touched.
        write_report("performance_report.json", {
            "status": "fail",
            "error": f"this task rebinds its seam through a candidate overlay, and the overlay "
                     f"could not be built ({overlay_note}). Timing would measure the production "
                     f"stack, not source/.",
            "candidate_overlay": overlay_note,
            "test_cases": [],
        })
        print("Performance: FAILED - candidate overlay unavailable, refusing to time the "
              "production stack")
        return []

    cmd = [sys.executable, "-u", bench, "--ut", UT_DIR, "--out", out_json,
           "--warmup", str(WARMUP_ITERATIONS), "--iters", str(BENCHMARK_ITERATIONS)]
    try:
        proc = subprocess.run(cmd, cwd=TASK_DIR, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc = None

    if proc is None:
        reason = f"_bench.py timed out after {timeout}s"
    else:
        reason = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:] or \
                 [f"_bench.py exit={proc.returncode}"]
    if proc is not None and proc.returncode == 0:
        try:
            with open(out_json) as fh:
                raw = json.load(fh)
            cases = _benchmark_cases(raw)
        except (OSError, ValueError, TypeError) as exc:
            reason = f"invalid benchmark report: {exc}"
        else:
            write_report("performance_report.json", {
                "status": "ok",
                "methodology": (f"{WARMUP_ITERATIONS} warmup + {BENCHMARK_ITERATIONS} measured "
                                f"iterations per case, cuda-event device time, reported as the mean"),
                "warmup_iterations": WARMUP_ITERATIONS,
                "benchmark_iterations": BENCHMARK_ITERATIONS,
                "candidate_overlay": overlay_note,
                "timer": raw.get("timer"),
                "test_cases": cases,
            })
            total = sum(c["execution_time_ms"] for c in cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(cases)} test case(s), total time: {total:.4f} ms")
            return cases

    # Only the explicit unsupported-replay result permits another timing path.
    # Execution failures, partial reports and timeouts must remain failures.
    if proc is not None and proc.returncode == EXIT_NO_RECORDS and not os.path.exists(out_json):
        return run_performance_via_ut(cfg, timeout, overlay_note, reason)
    write_report("performance_report.json", {
        "status": "fail", "error": "native benchmark failed; no fallback is valid",
        "failure_reason": reason, "candidate_overlay": overlay_note,
        "fallback_used": False, "test_cases": [],
    })
    print("Performance: FAILED - native benchmark did not produce a valid measurement")
    return []


def _per_case_from_stdout(out):
    """Harvest the GEAK harness's per-case timing block.

    Packages spell the marker two ways -- ``GEAK_PER_CASE=<json>`` and
    ``GEAK_PER_CASE <json>``. Most of the UTs in this suite use the space form,
    so an ``=``-only parse silently reported them as "no test cases measured"
    and failed the performance leg of a task whose UT had passed.
    """
    per_case = []
    for line in out.splitlines():
        head, sep, payload = line.partition("=")
        if head.strip() == "GEAK_PER_CASE" and sep:
            pass                                  # GEAK_PER_CASE=<json>
        elif line.startswith("GEAK_PER_CASE "):
            payload = line[len("GEAK_PER_CASE "):]   # GEAK_PER_CASE <json>
        else:
            continue                              # not this marker (GEAK_PER_CASE_TOTAL=...)
        try:
            parsed = json.loads(payload.strip())
        except Exception:
            continue
        if isinstance(parsed, list) and all(isinstance(r, dict) for r in parsed):
            per_case = parsed                     # a scalar or a string is not a case list
    return per_case


def _per_case_from_result_json(started_at):
    """The callable UTs write ut/result.json with the same per-case schema.

    ``started_at`` is the wall clock from just before this run's UT was invoked.
    The suite SHIPS each package's capture-time result.json, which records a full
    prior PASS including per-case baseline_ms/optimized_ms from the packager's own
    GPU run days ago. Reading it unconditionally would republish those numbers as
    this run's measurement whenever the UT died before rewriting the file -- a
    fabricated result, reported with status "ok". So only accept the file if this
    run actually rewrote it.
    """
    path = os.path.join(UT_DIR, "result.json")
    if not os.path.isfile(path):
        return []
    if os.path.getmtime(path) < started_at:
        print(f"[perf] ignoring {os.path.relpath(path, TASK_DIR)}: not rewritten by this run "
              f"(shipped from the upstream capture)", file=sys.stderr)
        return []
    try:
        blob = json.load(open(path))
    except Exception:
        return []
    timing = blob.get("timing") or {}
    for spot in (timing.get("per_case"),
                 (timing.get("aggregate") or {}).get("per_case")):
        if isinstance(spot, list) and spot:
            return spot
    return []


TIMING_LINE = re.compile(
    r"^timing:(?P<sig>\S+)\s+baseline_ms=(?P<baseline_ms>\S+)\s+"
    r"candidate_ms=(?P<optimized_ms>\S+)"
    r"(?:\s+speedup=(?P<speedup>\S+))?(?:\s+reps=(?P<reps>\S+))?$")


def _per_case_from_json_block(out):
    """Some UTs print no marker at all -- they pretty-print the whole result.

    The three DeepSeek UTs end with
    ``print(json.dumps({"per_case": [...], "geomean": ...}, indent=2))``, so the
    numbers are sitting in the captured output in a form none of the marker
    parsers can see. Scan for a balanced JSON object containing "per_case".
    """
    text, out_rows = out, []
    start = 0
    while True:
        i = text.find('"per_case"', start)
        if i < 0:
            return out_rows
        start = i + 1
        obj = text.rfind("{", 0, i)
        if obj < 0:
            continue
        depth, end = 0, None
        for j in range(obj, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            continue
        try:
            rows = json.loads(text[obj:end]).get("per_case")
        except Exception:
            continue
        if isinstance(rows, list) and all(isinstance(r, dict) for r in rows) and rows:
            out_rows = rows
    return out_rows


def _per_case_from_timing_lines(out):
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("timing:"):
            continue
        m = TIMING_LINE.fullmatch(line)
        if not m:
            raise ValueError(f"malformed fallback timing row: {line}")
        d = m.groupdict()
        try:
            baseline = float(d["baseline_ms"])
            candidate = float(d["optimized_ms"])
            reps = None if d["reps"] in (None, "None") else int(d["reps"])
        except ValueError as exc:
            raise ValueError(f"invalid fallback timing row: {line}") from exc
        if any(not math.isfinite(value) or value <= 0 for value in (baseline, candidate)):
            raise ValueError(f"nonpositive or nonfinite fallback timing row: {line}")
        rows.append({"sig": d["sig"],
                     "baseline_ms": baseline,
                     "optimized_ms": candidate,
                     "reps": reps})
    return rows


def run_performance_via_ut(cfg, timeout, overlay_note, reason):
    started_at = time.time()
    try:
        proc, secs = run_ut(timeout)
    except subprocess.TimeoutExpired:
        write_report("performance_report.json",
                     {"status": "fail", "error": f"timed out after {timeout}s", "test_cases": []})
        print("Performance: FAILED - no test cases measured")
        return []
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        write_report("performance_report.json", {
            "status": "fail", "error": f"candidate binding setup failed: {exc}",
            "candidate_overlay": overlay_note, "fallback_reason": reason,
            "test_cases": [],
        })
        print("Performance: FAILED - candidate correctness binding unavailable")
        return []

    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        # A UT that did not pass did not measure anything either. Reporting
        # numbers scraped from a failed run - or worse, from the shipped
        # capture-time result.json it never overwrote - would be fabrication.
        write_report("performance_report.json", {
            "status": "fail",
            "error": f"ut/unittest.py exit={proc.returncode}; no measurement is valid from a "
                     f"run that did not pass",
            "candidate_overlay": overlay_note,
            "fallback_reason": reason,
            "duration_seconds": round(secs, 2),
            # Keep enough context to diagnose a failure from the archive alone. 20 lines
        # was routinely all aiter dispatch chatter with the verdict scrolled off.
        "stdout_tail": out.strip().splitlines()[-120:],
            "test_cases": [],
        })
        print("Performance: FAILED - the unit test did not pass, so nothing was measured")
        return []

    try:
        # Validate every emitted timing row even when a structured representation
        # is available. An invalid row cannot disappear behind another format.
        timing_rows = _per_case_from_timing_lines(out)
        per_case, source = _per_case_from_stdout(out), "GEAK_PER_CASE marker"
        if not per_case:
            per_case, source = _per_case_from_json_block(out), "ut stdout per_case JSON block"
        if not per_case:
            per_case, source = timing_rows, "ut stdout timing: lines"
        if not per_case:
            per_case, source = (_per_case_from_result_json(started_at),
                                "ut/result.json timing.per_case (rewritten by this run)")
    except (ValueError, TypeError, AttributeError) as exc:
        write_report("performance_report.json", {
            "status": "fail", "error": f"invalid fallback timing report: {exc}",
            "candidate_overlay": overlay_note, "fallback_reason": reason,
            "duration_seconds": round(secs, 2), "test_cases": [],
        })
        print("Performance: FAILED - invalid fallback timing report")
        return []
    if not per_case:
        source = "none - the UT reported no per-case timing in any known form"
    # execution_time_ms is the CANDIDATE's time. Falling back to the baseline's
    # would credit an untimed candidate with the reference implementation's
    # performance. Invalid or missing candidate times invalidate the whole
    # attempted result; no row-level unsupported-measurement contract exists.
    cases, unusable = [], 0
    for c in per_case:
        t = c.get("optimized_ms") if isinstance(c, dict) else None
        if not (type(t) in (int, float) and math.isfinite(t) and t > 0):
            unusable += 1
            continue
        cases.append({
            "test_case_id": c.get("sig"),
            "execution_time_ms": t,
            "params": {"regime": c.get("regime"), "m": c.get("m")},
            "baseline_ms": c.get("baseline_ms"),
            "reps": c.get("reps"),
            "speedup_spread": c.get("speedup_spread"),
        })
    if unusable:
        cases = []

    write_report("performance_report.json", {
        "status": "ok" if cases else "fail",
        "error": "fallback contains invalid candidate measurements" if unusable else None,
        "methodology": ("GEAK measure_legs fallback: interleaved baseline/candidate pairs in fresh "
                        "subprocesses, median over up to 3 pairs, cuda-event device time with a "
                        "cache flush before each sample. Used because this op's oracle carries no "
                        "replayable argument records."),
        "methodology_is_arena_default": False,
        "fallback_reason": reason,
        "per_case_source": source,
        "candidate_overlay": overlay_note,
        "duration_seconds": round(secs, 2),
        "rows_without_a_candidate_time": unusable,
        "test_cases": cases,
    })
    if cases:
        total = sum(c["execution_time_ms"] for c in cases if c["execution_time_ms"] > 0)
        print(f"Performance: measured {len(cases)} test case(s), total time: {total:.4f} ms")
    else:
        print("Performance: FAILED - no test cases measured")
    return cases


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="AgentKernelArena head-kernel task runner")
    ap.add_argument("mode", choices=["compile", "correctness", "performance"])
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("HK_TASK_TIMEOUT", "1800")))
    args = ap.parse_args()

    os.makedirs(BUILD_DIR, exist_ok=True)
    cfg = load_config()

    if args.mode == "compile":
        ok, err = run_compile(cfg)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    if args.mode == "correctness":
        ok, err = run_correctness(cfg, args.timeout)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    cases = run_performance(cfg, args.timeout)
    sys.exit(0 if cases else 1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_correctness_child":
        sys.exit(_correctness_child(json.loads(sys.argv[2])))
    main()
