#!/usr/bin/env python3
"""AgentKernelArena task runner for a head-kernel benchmark task.

RMSNorm task runner. Three modes, the standard arena contract:

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
performance  Uses the canonical graph helper for 10 warmups + 100 measured
             replays per live shape; preserves every sample and validates both
             outputs of the measured graph. Graph failure is a failed run.

Exit 0 pass, non-zero fail. Reports land in ``build/``.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(TASK_DIR, "build")
UT_DIR = os.path.join(TASK_DIR, "ut")
CONFIG = os.path.join(TASK_DIR, "config.yaml")

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


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
def _ut_command():
    return [sys.executable, "-u", os.path.join(UT_DIR, "unittest.py")]


def run_ut(timeout):
    """Run the frozen GEAK unittest in its own directory.

    It manages its own overlays: the baseline leg must resolve to the live
    serving stack and the candidate leg to ``ut/kernel_src/`` (symlinked to
    ``source/``), so nothing is injected on PYTHONPATH here.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    t0 = time.time()
    proc = subprocess.run(_ut_command(), cwd=UT_DIR, env=env, capture_output=True,
                          text=True, timeout=timeout)
    return proc, time.time() - t0


def run_correctness(cfg, timeout):
    if not os.path.isfile(os.path.join(UT_DIR, "unittest.py")):
        return False, "ut/unittest.py missing - this task has no correctness oracle"
    try:
        proc, secs = run_ut(timeout)
    except subprocess.TimeoutExpired:
        write_report("correctness_report.json",
                     {"status": "fail", "error": f"unittest.py timed out after {timeout}s"})
        return False, f"unittest.py timed out after {timeout}s"

    out = proc.stdout + proc.stderr
    # The GEAK driver's exit code IS the contract (0 pass, 1 correctness FAIL,
    # 2 environment, 3 harness incomplete). Do not additionally require a bare
    # "PASS" line: packages spell the verdict differently ("PASS",
    # "RESULT PASS (oracle=True random_parity=True)", or only the
    # GEAK_WEIGHTED_SPEEDUP block), and demanding one spelling reports a passing
    # kernel as a failure.
    ok = proc.returncode == 0
    challenge_report, challenge_error = None, None
    if ok:
        try:
            challenge_report, challenge_out = run_cpu_truth_challenges(max(1, timeout - secs))
            out += "\n" + challenge_out
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            ok, challenge_error = False, str(exc)
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
        "error": None if ok else challenge_error or
                 f"ut/unittest.py exit={proc.returncode} ({meaning})",
        "cpu_truth_challenges": challenge_report,
        "exit_code": proc.returncode,
        "exit_meaning": meaning,
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
        return False, challenge_error or f"ut/unittest.py exit={proc.returncode}: {out.strip().splitlines()[-3:]}"
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


def run_cpu_truth_challenges(timeout):
    """Supplement the unchanged frozen UT with independent CPU-truth cases."""
    overlay = candidate_overlay()
    if not overlay:
        raise RuntimeError("CPU-truth correctness requires the candidate overlay")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([overlay] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    output = os.path.join(BUILD_DIR, "_cpu_truth_correctness.json")
    try:
        os.unlink(output)
    except FileNotFoundError:
        pass
    command = [sys.executable, "-u", os.path.join(TASK_DIR, "scripts", "_bench.py"),
               "--ut", UT_DIR, "--out", output, "--correctness-only"]
    proc = subprocess.run(command, cwd=TASK_DIR, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError("CPU-truth correctness failed: " + (proc.stderr or proc.stdout)[-2000:])
    with open(output) as fh:
        raw = json.load(fh)
    with open(os.path.join(UT_DIR, "meta.json")) as fh:
        meta = json.load(fh)
    fields = ("regime", "m", "n", "dtype", "eps", "x_shape", "x_stride",
              "residual_shape", "residual_stride", "weight_shape", "weight_stride")
    expected = {f"{case['sig']}|{case['regime']}": {key: case[key] for key in fields}
                for case in meta["workload"]["cases"]}
    rows = raw.get("cases", [])
    kinds = ("ordinary", "zero_residual", "zero_sum", "near_cancellation", "small_amplitude")
    if (raw.get("status") != "ok" or raw.get("mode") != "cpu_truth_correctness"
            or not isinstance(rows, list) or len(rows) != 2
            or any(not isinstance(row, dict) for row in rows)
            or {row.get("sig") for row in rows} != set(expected)
            or any(row.get("params") != expected[row["sig"]]
                   or row.get("checks") != [{"kind": kind, "passed": True} for kind in kinds]
                   or row.get("expected_device") != "cpu" or row.get("tolerance") != meta["tol"]
                   or type(row.get("checked_invocations")) is not int or row["checked_invocations"] != len(kinds)
                   or type(row.get("max_rel_err")) not in (int, float)
                   or not math.isfinite(row["max_rel_err"]) or row["max_rel_err"] < 0 for row in rows)):
        raise ValueError("incomplete CPU-truth correctness challenge report")
    return raw, proc.stdout


def _benchmark_cases(raw):
    """Require the complete graph protocol before publishing either live case."""
    if not isinstance(raw, dict) or raw.get("timer") != "cuda_graph":
        raise ValueError("benchmark report must use cuda_graph timing")
    if any(type(raw.get(key)) is not int or raw[key] != expected
           for key, expected in (("warmup", WARMUP_ITERATIONS), ("iters", BENCHMARK_ITERATIONS))):
        raise ValueError("benchmark report has incorrect iteration counts")
    with open(os.path.join(UT_DIR, "meta.json")) as fh:
        meta = json.load(fh)
    fields = ("regime", "m", "n", "dtype", "eps", "x_shape", "x_stride",
              "residual_shape", "residual_stride", "weight_shape", "weight_stride")
    expected = {f"{spec['sig']}|{spec['regime']}": {key: spec[key] for key in fields}
                for spec in meta["workload"]["cases"]}
    rows = raw.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected) or len(expected) != 2:
        raise ValueError("benchmark report must contain both live cases")
    seen, result = set(), []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sig"), str):
            raise ValueError("benchmark report has an invalid case")
        sig = row["sig"]
        if sig not in expected or sig in seen or row.get("params") != expected[sig]:
            raise ValueError("benchmark case identity/geometry is missing, duplicated or changed")
        seen.add(sig)
        if (row.get("benchmark_method") != "cuda_graph"
                or "benchmark_fallback_reason" in row
                or any(type(row.get(key)) is not int or row[key] != value
                       for key, value in (("benchmark_warmup", WARMUP_ITERATIONS),
                                          ("benchmark_samples", BENCHMARK_ITERATIONS),
                                          ("benchmark_effective_repeats", 1)))):
            raise ValueError("benchmark case has an invalid graph method or iteration counts")
        samples = row.get("samples_ms")
        if (not isinstance(samples, list) or len(samples) != BENCHMARK_ITERATIONS
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)):
            raise ValueError("benchmark case must retain every positive finite raw sample")
        for key, value in (("mean_ms", statistics.mean(samples)),
                           ("median_ms", statistics.median(samples)), ("min_ms", min(samples))):
            if (type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                    or not math.isclose(row[key], value, rel_tol=1e-12, abs_tol=0.0)):
                raise ValueError("benchmark summaries do not match raw samples")
        validation = row.get("validation", {})
        if (not isinstance(validation, dict)
                or validation.get("checked_invocations") != WARMUP_ITERATIONS + BENCHMARK_ITERATIONS + 8
                or validation.get("timing_checked_invocations") != WARMUP_ITERATIONS + BENCHMARK_ITERATIONS + 4
                or validation.get("fresh_input_sets") != WARMUP_ITERATIONS + BENCHMARK_ITERATIONS + 10
                or validation.get("expected_device") != "cpu"
                or validation.get("fresh_inputs_per_replay") is not True
                or validation.get("input_seed") != 31000 + list(expected).index(sig)
                or validation.get("correctness_challenges") != [
                    {"kind": kind, "passed": True} for kind in
                    ("zero_residual", "zero_sum", "near_cancellation", "small_amplitude")]
                or validation.get("tolerance") != meta["tol"]
                or validation.get("outputs") != ["normed", "pre_norm_sum"]
                or validation.get("inputs_varied") != ["x", "residual", "weight"]
                or validation.get("outputs_poisoned_before_replay") is not True
                or validation.get("measured_graph_validated") is not True
                or type(validation.get("max_rel_err")) not in (int, float)
                or not math.isfinite(validation["max_rel_err"]) or validation["max_rel_err"] < 0):
            raise ValueError("benchmark case has incomplete measured-graph validation")
        result.append({
            "test_case_id": sig, "execution_time_ms": row["mean_ms"],
            "params": row["params"], "median_ms": row["median_ms"], "min_ms": row["min_ms"],
            "samples_ms": samples, "validation": validation,
            **{key: value for key, value in row.items() if key.startswith("benchmark_")},
        })
    return result


def run_performance(cfg, timeout):
    bench = os.path.join(TASK_DIR, "scripts", "_bench.py")
    out_json = os.path.join(BUILD_DIR, "_bench_raw.json")
    os.makedirs(BUILD_DIR, exist_ok=True)
    try:
        os.unlink(out_json)
    except FileNotFoundError:
        pass

    # The configured entrypoint requests the canonical workspace helper used by
    # _bench.py. A source checkout without materialization cannot run timing.
    helper = importlib.util.find_spec("_aka_benchmark")
    expected_helper = os.path.join(TASK_DIR, "scripts", "_aka_benchmark.py")
    if helper is None or helper.origin != expected_helper:
        write_report("performance_report.json", {
            "status": "fail", "error": "canonical _aka_benchmark.py is not materialized beside the runner",
            "test_cases": [],
        })
        return []

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
                                f"graph replays per case, canonical device timing, all raw samples retained"),
                "warmup_iterations": WARMUP_ITERATIONS,
                "benchmark_iterations": BENCHMARK_ITERATIONS,
                "candidate_overlay": overlay_note,
                "timer": raw.get("timer"),
                "test_cases": cases,
            })
            total = sum(c["execution_time_ms"] for c in cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(cases)} test case(s), total time: {total:.4f} ms")
            return cases

    write_report("performance_report.json", {
        "status": "fail", "error": "native benchmark failed; no fallback is valid",
        "failure_reason": reason, "candidate_overlay": overlay_note,
        "fallback_used": False, "test_cases": [],
    })
    print("Performance: FAILED - native benchmark did not produce a valid measurement")
    return []


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
    main()
