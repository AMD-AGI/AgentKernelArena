#!/usr/bin/env python3
"""Summarize the archived GPU results in _results/ -- one line per built task.

run_on_gpu.sh archives each task's three build/*.json here the moment it
finishes, so this survives tools/clean.sh and shows progress across the several
spur reservations a full sweep usually spans.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
TASKS = os.path.join(SUITE, "tasks", "headkernel")
RESULTS = os.path.join(SUITE, "_results")

EXIT = {0: "pass", 1: "correctness FAIL", 2: "environment", 3: "harness incomplete"}


def load(task, name):
    p = os.path.join(RESULTS, task, name)
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    built = sorted(t for t in os.listdir(TASKS)
                   if os.path.isfile(os.path.join(TASKS, t, "config.yaml")))
    w = max(len(t) for t in built) + 1
    print(f"{'task':{w}} {'compile':9} {'correctness':22} performance")
    print("-" * (w + 60))
    tally = {"pass": 0, "fail": 0, "missing": 0}
    for t in built:
        c, k, p = (load(t, n) for n in ("compile_report.json",
                                        "correctness_report.json",
                                        "performance_report.json"))
        if not (c or k or p):
            print(f"{t:{w}} {'-':9} {'(not run yet)':22} -")
            tally["missing"] += 1
            continue
        cs = "PASS" if (c or {}).get("status") == "ok" else "FAIL" if c else "-"
        if k:
            code = k.get("exit_code")
            ks = "PASS" if k.get("status") == "ok" else f"FAIL exit={code} {EXIT.get(code,'signal')}"
        else:
            ks = "-"
        if p:
            tc = p.get("test_cases") or []
            n = len(tc)
            real = [c for c in tc
                    if isinstance(c.get("execution_time_ms"), (int, float))
                    and c["execution_time_ms"] > 0]
            if p.get("status") == "ok" and real and len(real) == n:
                ps = f"{n} case(s)"
            elif p.get("status") == "ok":
                # status ok but some/all rows carry no candidate time -- not a measurement
                ps = f"SUSPECT {len(real)}/{n} timed"
                p = dict(p, status="fail")
            else:
                ps = "FAIL 0 cases"
            if p.get("methodology_is_arena_default") is False:
                ps += " [GEAK fallback]"
        else:
            ps = "-"
        ok = cs == "PASS" and ks == "PASS" and p and p.get("status") == "ok"
        tally["pass" if ok else "fail"] += 1
        print(f"{t:{w}} {cs:9} {ks:22} {ps}")
    print(f"\n{tally['pass']} fully green, {tally['fail']} with a failing leg, "
          f"{tally['missing']} not run yet  (of {len(built)} built tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
