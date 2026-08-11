#!/usr/bin/env python3
"""AgentKernelArena task runner for hip2hip/paged_attention_decode.

Format-only adapter: wraps the local extracted baseline harness (pa_aiter_baseline.py,
which reuses pa_harness.py) as a subprocess and translates SNR / median_us
output into the arena contract. Kernel and harness are untouched.

The baseline harness runs the local pa_v1 extraction vs the torch golden for a single
(--num-seqs, --ctx-len) and prints SNR + median_us. We derive:
  compile     — import the local pa_v1 extraction.
  correctness — run num_seqs=64 ctx=2048; PASS iff SNR >= 30 dB (the pass bar).
  performance — run for each decode-shape point; median_us -> execution_time_ms.
"""
import argparse
import json
import os
import re
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "hip2hip/paged_attention_decode"
HARNESS = os.path.join(TASK_DIR, "pa_aiter_baseline.py")
BUILD_DIR = os.path.join(TASK_DIR, "build")

SNR_THRESH = 30.0
# Arena regime: decode paged-attention. concurrency B -> num_seqs (decode batch,
# q_len=1 inherent to paged_attention_v1); seqlen 1024 -> ctx_len. ids c2/c32/c64.
# (num_seqs, ctx_len, case_id)
PERF_SHAPES = [(2, 1024, "c2"), (32, 1024, "c32"), (64, 1024, "c64")]
# correctness uses the largest regime case (most heads/seqs exercised).
CORRECTNESS_SHAPE = (64, 1024)

SNR_RE = re.compile(r"SNR vs torch golden:\s*([0-9.\-]+)\s*dB")
US_RE = re.compile(r"median_us:\s*([0-9.]+)")


def _run(num_seqs, ctx_len, timeout=900, reps=100, warmup=10):
    r = subprocess.run(
        [sys.executable, HARNESS, "--num-seqs", str(num_seqs), "--ctx-len", str(ctx_len),
         "--reps", str(reps), "--warmup", str(warmup)],
        cwd=TASK_DIR, capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def run_compile():
    try:
        subprocess.run([sys.executable, "-c", "import pa_aiter_baseline"], cwd=TASK_DIR,
                       check=True, capture_output=True, text=True, timeout=300)
        return True, None
    except Exception as e:
        tail = getattr(e, "stderr", "") or str(e)
        return False, f"import local pa_v1 failed: {tail.strip()[-240:]}"


def run_correctness():
    try:
        rc, out, err = _run(*CORRECTNESS_SHAPE)
    except Exception as e:
        return False, str(e)
    sys.stdout.write(out)
    m = SNR_RE.search(out)
    if not m:
        return False, (err.strip()[-240:] or "no SNR reported")
    snr = float(m.group(1))
    if snr >= SNR_THRESH:
        return True, None
    return False, f"SNR {snr:.2f} dB < {SNR_THRESH} dB"


def run_performance():
    cases = []
    for (ns, ctx, cid) in PERF_SHAPES:
        try:
            rc, out, err = _run(ns, ctx)
            sys.stdout.write(out)
            m = US_RE.search(out)
            if m:
                cases.append({"test_case_id": cid,
                              "execution_time_ms": float(m.group(1)) / 1000.0,
                              "params": {"num_seqs": ns, "ctx_len": ctx}})
            else:
                cases.append({"test_case_id": cid, "execution_time_ms": -1.0,
                              "params": {"num_seqs": ns, "ctx_len": ctx,
                                         "stderr": err.strip()[-240:]}})
        except Exception as e:
            cases.append({"test_case_id": cid, "execution_time_ms": -1.0,
                          "params": {"num_seqs": ns, "ctx_len": ctx, "error": str(e)[:240]}})
    return cases


def main():
    p = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    p.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = p.parse_args()
    os.makedirs(BUILD_DIR, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(BUILD_DIR, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(BUILD_DIR, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        cases = run_performance()
        json.dump({"test_cases": cases},
                  open(os.path.join(BUILD_DIR, "performance_report.json"), "w"), indent=2)
        for c in cases:
            if c["execution_time_ms"] > 0:
                print(f"Performance: {c['execution_time_ms']:.5f} ms ({c['test_case_id']})")
            else:
                print(f"Performance: FAILED ({c['test_case_id']})")
        sys.exit(0)


if __name__ == "__main__":
    main()
