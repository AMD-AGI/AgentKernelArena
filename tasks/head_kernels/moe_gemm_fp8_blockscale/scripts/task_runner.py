#!/usr/bin/env python3
"""AgentKernelArena task runner for hip2hip/moe_gemm_fp8_blockscale.

Format-only adapter: wraps the existing baseline harness (asm_baseline.py, which
reuses moe_harness.py) as a subprocess and translates SNR / mean_us output
into the arena contract. Kernel and harness are untouched.

The baseline harness has no correctness/benchmark split; it runs the ASM kernel
vs the torch golden for a single --token and prints SNR + mean_us. We derive:
  compile     — import moe_harness (imports aiter).
  correctness — run --token 64; PASS iff SNR >= 25 dB (the production pass bar).
  performance — run for each token regime; mean_us -> execution_time_ms.
"""
import argparse
import json
import os
import re
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "hip2hip/moe_gemm_fp8_blockscale"
HARNESS = os.path.join(TASK_DIR, "asm_baseline.py")
BUILD_DIR = os.path.join(TASK_DIR, "build")

SNR_THRESH = 25.0   # production pass bar (RESULTS.md); a correct CK 2-stage candidate hits ~32.7 dB at all tokens
COS_THRESH = 0.01   # moe_harness.COS_THRESH — aiter's strict tune threshold

# WORKLOAD REGIME: seqlen=1024, concurrency B in {2,32,64}. This is a token-parallel
# MoE GEMM, so the regime maps onto the kernel's token (M) dim as M = B * 1024:
#   c2  -> token 2048, c32 -> token 32768, c64 -> token 65536.
# Model dims (model_dim=3072, inter_dim=768 [TP=2], E=256, topk=8, blockscale 128x128)
# are kept from the captured base case (moe_harness.SHAPE).
PERF_CASES = [("c2", 2 * 1024), ("c32", 32 * 1024), ("c64", 64 * 1024)]

# Correctness oracle runs across ALL real token regimes (decode -> big prefill),
# validating the EDITED candidate (asm_baseline.asm_call dispatch) against the
# independent torch fp32 golden. The candidate must clear the production bar
# (SNR >= 25 dB AND cosine_diff < 0.01) at EVERY regime. NOTE: the unmodified ASM
# .co is itself below 25 dB at token>=256 (RESULTS.md) and so will (correctly)
# FAIL these large regimes — the task's whole point is to replace it with the
# CK 2-stage path, which is accurate (~32.7 dB) at all tokens. Pinning the oracle
# to token=64 only existed to let the inaccurate baseline pass, masking the very
# defect the task asks the candidate to fix.
CORRECTNESS_TOKENS = [64, 256, 1024, 4096]

SNR_RE = re.compile(r"SNR vs torch golden:\s*([0-9.\-]+)\s*dB")
COS_RE = re.compile(r"cosine_diff vs torch golden:\s*([0-9.eE\-+]+)")
US_RE = re.compile(r"(?:mean_us|median_us):\s*([0-9.]+)")


def _run(token, timeout=1200):
    r = subprocess.run(
        [sys.executable, HARNESS, "--token", str(token)], cwd=TASK_DIR,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def run_compile():
    try:
        subprocess.run([sys.executable, "-c", "import moe_harness"], cwd=TASK_DIR,
                       check=True, capture_output=True, text=True, timeout=300)
        return True, None
    except Exception as e:
        tail = getattr(e, "stderr", "") or str(e)
        return False, f"import moe_harness failed: {tail.strip()[-240:]}"


def run_correctness():
    all_ok = True
    fails = []
    for tok in CORRECTNESS_TOKENS:
        try:
            rc, out, err = _run(tok)
        except Exception as e:
            return False, f"token={tok}: {e}"
        sys.stdout.write(out)
        m = SNR_RE.search(out)
        c = COS_RE.search(out)
        if not m or not c:
            return False, f"token={tok}: " + (err.strip()[-240:] or "no SNR/cosine reported")
        snr = float(m.group(1))
        cos = float(c.group(1))
        ok = (snr >= SNR_THRESH) and (cos < COS_THRESH)
        print(f"  token={tok}  SNR={snr:.2f}dB  cos_diff={cos:.4e}  "
              f"{'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
        if not ok:
            fails.append(f"token={tok} SNR={snr:.2f}dB cos_diff={cos:.2e}")
    if all_ok:
        print("ALL CORRECTNESS CHECKS PASSED")
        return True, None
    return False, "; ".join(fails)


def run_performance():
    cases = []
    for cid, tok in PERF_CASES:
        try:
            rc, out, err = _run(tok)
            sys.stdout.write(out)
            m = US_RE.search(out)
            if m:
                ms = float(m.group(1)) / 1000.0
                cases.append({"test_case_id": cid, "execution_time_ms": ms,
                              "params": {"token": tok, "concurrency": tok // 1024,
                                         "seqlen": 1024}})
                print(f"Performance: {ms:.4f} ms ({cid})")
            else:
                cases.append({"test_case_id": cid, "execution_time_ms": -1.0,
                              "params": {"token": tok, "stderr": err.strip()[-240:]}})
                print(f"Performance: FAIL ({cid})")
        except Exception as e:
            cases.append({"test_case_id": cid, "execution_time_ms": -1.0,
                          "params": {"token": tok, "error": str(e)[:240]}})
            print(f"Performance: FAIL ({cid})")
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
        ok = [c for c in cases if c["execution_time_ms"] > 0]
        print(f"Performance: measured {len(ok)}/{len(cases)} case(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
