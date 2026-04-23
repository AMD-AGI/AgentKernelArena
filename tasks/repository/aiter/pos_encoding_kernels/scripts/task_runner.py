#!/usr/bin/env python3
"""Task runner for repository/aiter/pos_encoding_kernels.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (aiter pip-installed environment):
  - Kernel source : /opt/venv/lib/python3.12/site-packages/aiter_meta/csrc/kernels/pos_encoding_kernels.cu
  - JIT module    : module_pos_encoding
  - JIT .so cache : /opt/venv/lib/python3.12/site-packages/aiter/jit/module_pos_encoding.so
  - JIT build dir : /opt/venv/lib/python3.12/site-packages/aiter/jit/build/module_pos_encoding/
  - Test file     : /workspace/aiter-src/op_tests/test_rope.py

CRITICAL NOTES:
  - Do NOT run test_rope.py without parameters — it compiles many modules (~570s total).
  - query/key must be 2D [s*b, h*d] format, NOT 3D (3D causes GPU memory crash).
  - The "leg" value in performance output is the legacy pos_encoding_kernels.cu timing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/pos_encoding_kernels"
MODULE_NAME = "module_pos_encoding"

# aiter pip-installed paths
AITER_JIT_DIR = Path("/opt/venv/lib/python3.12/site-packages/aiter/jit")
AITER_SRC_DIR = Path("/workspace/aiter-src")
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_rope.py"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _report_root() -> Path:
    d = _workspace_root() / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout_s: int = 600) -> tuple[bool, str]:
    """Run a command and return (success, combined_output)."""
    print(f"[RUN] {' '.join(cmd)}")
    sys.stdout.flush()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = proc.stdout + proc.stderr
        print(output, end="", flush=True)
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return False, str(e)


def _delete_jit_cache() -> None:
    """Remove JIT-compiled .so and build directory to force recompilation."""
    so_path = AITER_JIT_DIR / f"{MODULE_NAME}.so"
    build_dir = AITER_JIT_DIR / "build" / MODULE_NAME

    if so_path.exists():
        so_path.unlink()
        print(f"Deleted: {so_path}")
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print(f"Deleted: {build_dir}")


# ── compile ──────────────────────────────────────────────────────────────────

def run_compile() -> None:
    """Force JIT recompilation of module_pos_encoding by deleting cache and running a minimal test.

    Uses restricted args to compile only module_pos_encoding (~38s),
    avoiding the full test_rope.py run which compiles many modules (~570s).
    """
    _delete_jit_cache()

    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "-d", "fp16",
            "-b", "1",
            "-s", "1024",
            "-hs", "64",
            "-hd", "128",
            "-rs", "neox",
            "-rr", "0",
            "-t", "f",
        ],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    report = {"status": "ok" if ok else "fail", "output": out[-2000:]}
    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))

    if not ok:
        print("Compilation: FAIL")
        sys.exit(1)
    print("Compilation: PASS")


# ── correctness ──────────────────────────────────────────────────────────────

def run_correctness() -> None:
    """Run correctness tests using compare mode.

    Compares legacy pos_encoding_kernels.cu output against the new rope
    implementation. Uses checkAllclose with atol=0.001, rtol=0.001.

    Success indicator: 'passed~'
    Failure indicators: 'FAIL' or 'failed'
    """
    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "--compare", "--compare_check", "--no_check",
            "-d", "fp16",
            "-b", "1", "4",
            "-s", "1024",
            "-hs", "64",
            "-hd", "128",
            "-rs", "neox", "gptj",
            "-rr", "0", "3",
        ],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    passed = len(re.findall(r"passed~", out))
    failed = len(re.findall(r"(?:FAIL|failed)", out))

    report = {
        "status": "ok" if (ok and failed == 0) else "fail",
        "passed": passed,
        "failed": failed,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok:
        print("Correctness: FAIL")
        sys.exit(1)
    print("Correctness: PASS")


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks using compare mode and write performance_report.json.

    Output format from test_rope.py --compare:
      hip: 168.22   us. leg: 259.69   us. diff: 64.77889673179855%.

    The 'leg' (legacy) value is the pos_encoding_kernels.cu timing.
    execution_time_ms = leg_us / 1000.
    """
    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "--compare", "--no_check",
            "-d", "fp16",
            "-b", "1", "4",
            "-s", "1024",
            "-hs", "64",
            "-hd", "128",
            "-rs", "neox", "gptj",
            "-rr", "0", "3",
        ],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    # Strip ANSI escape codes before parsing to avoid regex interference
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    out_clean = ansi_escape.sub("", out)

    # Parse performance lines: "hip: <hip_us> us. leg: <leg_us> us. diff: ..."
    # Context from test_rope.py output looks like:
    #   dtype: torch.float16, rotate_style: 0, rpar: (1.0, True, False), (s,b,hx,hy,d): (1024, 1, 64, 64, 128), has_offsets: False
    #   dtype: torch.float16, ..., dim_input: ..., rotate_style: 0, ...
    #   hip: 57.95    us. leg: 81.47    us. diff: ...
    perf_pattern = re.compile(
        r"hip:\s*(\d+\.?\d*)\s*us\.\s*leg:\s*(\d+\.?\d*)\s*us\."
    )

    lines = out_clean.splitlines()

    results = []
    seen = set()

    for i, line in enumerate(lines):
        m = perf_pattern.search(line)
        if not m:
            continue

        hip_us = float(m.group(1))
        leg_us = float(m.group(2))

        # Search backwards for context parameters (up to 10 lines back).
        # Iterate in REVERSE so the closest lines take priority (setdefault).
        context = {}
        for j in range(i - 1, max(-1, i - 11), -1):
            ctx_line = lines[j]
            # Pattern: "(s,b,hx,hy,d): (1024, 1, 64, 64, 128)"
            sbhd_m = re.search(
                r"\(s,b,hx,hy,d\):\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
                ctx_line,
            )
            if sbhd_m:
                context.setdefault("seq", sbhd_m.group(1))
                context.setdefault("batch", sbhd_m.group(2))
                context.setdefault("heads", sbhd_m.group(3))
                context.setdefault("head_dim", sbhd_m.group(5))

            # Pattern: "rotate_style: 0" or "rotate_style: 1"
            rs_m = re.search(r"rotate_style:\s*(\d+)", ctx_line)
            if rs_m:
                rs_val = int(rs_m.group(1))
                context.setdefault("style", "neox" if rs_val == 0 else "gptj")

            # Pattern: "rpar: (1.0, True, False)" — first element is rotary percent
            rpar_m = re.search(r"rpar:\s*\(([0-9.]+),", ctx_line)
            if rpar_m:
                rpar_val = float(rpar_m.group(1))
                # encode as percentage integer (1.0 -> 100, 0.5 -> 50)
                context.setdefault("rpar", str(int(rpar_val * 100)))

            # Pattern: "has_offsets: True/False"
            off_m = re.search(r"has_offsets:\s*(True|False)", ctx_line)
            if off_m:
                context.setdefault("has_offsets", off_m.group(1))

            # Pattern: "dtype: torch.float16"
            dtype_m = re.search(r"dtype:\s*torch\.(\w+)", ctx_line)
            if dtype_m:
                context.setdefault("dtype", dtype_m.group(1))

        batch = context.get("batch", "?")
        seq = context.get("seq", "?")
        heads = context.get("heads", "?")
        head_dim = context.get("head_dim", "?")
        style = context.get("style", "?")
        rpar = context.get("rpar", "?")
        has_offsets = context.get("has_offsets", "?")
        dtype = context.get("dtype", "fp16")

        key = (batch, seq, heads, head_dim, style, rpar, has_offsets, dtype)
        if key in seen:
            continue
        seen.add(key)

        test_case_id = (
            f"pos_encoding_b{batch}_s{seq}_hs{heads}_hd{head_dim}"
            f"_{style}_rp{rpar}_off{has_offsets}_{dtype}"
        )

        results.append({
            "test_case_id": test_case_id,
            "execution_time_ms": leg_us / 1000.0,
            "metadata": {
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
                "style": style,
                "rotary_percent": rpar,
                "has_offsets": has_offsets,
                "dtype": dtype,
                "kernel": "legacy_pos_encoding",
                "hip_us": hip_us,
                "leg_us": leg_us,
                "unit_original": "us",
            },
        })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))

    if results:
        avg_us = sum(r["metadata"]["leg_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
        for r in results:
            print(
                f"  {r['test_case_id']}: leg={r['metadata']['leg_us']:.2f} us"
                f"  hip={r['metadata']['hip_us']:.2f} us"
            )

    else:
        print("Performance: no results parsed")
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    if args.mode == "compile":
        run_compile()
    elif args.mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
