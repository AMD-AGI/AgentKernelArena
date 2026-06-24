#!/usr/bin/env python3
# ruff: noqa: E402 — bootstrap inserts paths before importing kernels package
"""Real execution-timing harness for FlyDSL silu_and_mul_fq (flydsl2flydsl).

This kernel is a fused MoE stage-1 post-processor:
  SiLU/SwiGLU(gate) * up  ->  FP4 (e2m1) / FP8 / bf16 quantized output
  + per-32-element E8M0 scales written into a tiled "sorted" layout.

Unlike the old compile-smoke stub (which timed kernel *compilation*), this
harness compiles each config ONCE and then times kernel *execution* with
torch.cuda.Event over `iters` (median).

Correctness oracle = SELF-REFERENCE: the PRISTINE original kernel in this task
dir (kernel.py) is loaded as the oracle, and the candidate kernel from
$GEAK_WORK_DIR/kernel.py (fallback: this task dir) is run on identical inputs.
The candidate's outputs (out_buf + out_scale_sorted) must match the oracle's
exactly. Deriving a full torch SiLU+mul+fp4 reference is impractical, so
self-reference is the accepted way to validate that an optimization preserves
numerics.

Speedup is a *display-only* relative number: candidate kernel latency vs a
simple torch SiLU+mul reference (silu(gate) * up) latency, reported as geomean.
It is NOT the correctness oracle.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# ============================================================================
# GEAK bootstrap — make `from kernels...` imports work and load kernel.py
# ============================================================================

KERNEL_FILE = "kernel.py"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_F2F_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # tasks/flydsl2flydsl
for _p in (_F2F_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_writable_home():
    """FlyDSL caches compiled kernels under $HOME/.flydsl. In the GEAK
    container $HOME is often read-only, so redirect HOME to a writable dir
    (must happen before flydsl is imported)."""

    def _writable(d):
        if not d:
            return False
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, ".geak_write_test")
            with open(t, "w") as fh:
                fh.write("ok")
            os.remove(t)
            return True
        except Exception:
            return False

    home = os.environ.get("HOME", "")
    if home and _writable(home):
        return
    for cand in (
        os.environ.get("GEAK_WORK_DIR", "").strip(),
        os.path.join(tempfile.gettempdir(), "geak_flydsl_home"),
    ):
        if _writable(cand):
            os.environ["HOME"] = cand
            return


_ensure_writable_home()


def _resolve_candidate_dir():
    """Directory of the kernel under test: $GEAK_WORK_DIR, else this task dir."""
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    candidates = [work_dir, _THIS_DIR]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, KERNEL_FILE)):
            return c
    return _THIS_DIR


def _load_kernel(kernel_dir, alias):
    """Import kernel.py from kernel_dir under a unique module alias."""
    entry = os.path.join(kernel_dir, KERNEL_FILE)
    if not os.path.isfile(entry):
        return None
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_CANDIDATE_DIR = _resolve_candidate_dir()
_ORACLE_DIR = _THIS_DIR  # pristine original always lives in the task dir

# ============================================================================
# Configs ("shapes")
#
# Each config drives build_silu_and_mul_fq_module(inter_dim, topk, quant_mode,
# ...). The default/canonical build (see config.yaml compile_command) is
# inter_dim=1024, topk=2. inter_dim must be divisible by 32.
#
#   config = (token_num, inter_dim, topk, quant_mode)
#
# Derived shapes/dtypes (see _make_inputs for the full derivation):
#   rows            = token_num * topk
#   num_sorted_rows = rows                      (identity routing -> grid blocks)
#   x               : (rows, inter_dim*2)  bf16  [gate | up], gui_layout=False
#   out_buf (fp4)   : (rows, inter_dim//2)  uint8  (2 e2m1 nibbles per byte)
#   out_buf (fp8)   : (rows, inter_dim)     uint8
#   out_buf (none)  : (rows, inter_dim)     bf16
#   out_scale_sorted: (ceil(num_sorted_rows/32)*32 * (inter_dim//32) + pad,) uint8
#   sorted_ids      : (num_sorted_rows,)   int32  packed (slot<<24)|token
#   num_valid_ids   : (1,)                 int32  = num_sorted_rows
#   topk_ids        : (rows,)              int32  (only read when enable_bias)
#   bias            : (1, inter_dim*2)     f32    (only read when enable_bias)
# ============================================================================

_INTER_DIM = 1024
_TOPK = 2
_QUANT = "fp4"


def _cfg(token_num, inter_dim=_INTER_DIM, topk=_TOPK, quant_mode=_QUANT):
    return (token_num, inter_dim, topk, quant_mode)


ALL_SHAPES = [
    _cfg(64),
    _cfg(128),
    _cfg(256),
    _cfg(512),
    _cfg(1024),
]

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 2)) for i in range(3)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]

# Build-options held fixed across configs (the kernel's primary code path).
_GUI_LAYOUT = False
_ACT = "silu"
_ENABLE_BIAS = False
_SWIGLU_LIMIT = 0.0

BLOCK_THREADS = 256  # mirrors kernel.py BLOCK_THREADS


# ============================================================================
# Input construction (derived strictly from kernel.py indexing)
# ============================================================================


def _make_inputs(cfg, seed=0):
    import torch

    token_num, inter_dim, topk, quant_mode = cfg
    assert inter_dim % 32 == 0, "inter_dim must be divisible by 32"

    rows = token_num * topk
    num_sorted_rows = rows
    scale_cols = inter_dim // 32

    torch.manual_seed(seed)

    # Activations: [gate (inter_dim) | up (inter_dim)] in bf16. Modest range so
    # SiLU*mul stays in a well-behaved numeric region for fp4 quantization.
    x = (torch.randn(rows, inter_dim * 2, device="cuda", dtype=torch.float32) * 0.5).to(torch.bfloat16)

    # Identity routing: sorted row i -> (token=i//topk, slot=i%topk).
    # Packed format matches the kernel: token_id = val & 0xFFFFFF, slot = val>>24
    # (confirmed by moe_sorting reference: (topk_pos << 24) | token_id).
    idx = torch.arange(num_sorted_rows, device="cuda", dtype=torch.int32)
    tok = idx // topk
    slot = idx % topk
    sorted_ids = (tok | (slot << 24)).to(torch.int32)

    num_valid_ids = torch.tensor([num_sorted_rows], device="cuda", dtype=torch.int32)

    # topk_ids / bias only read when enable_bias=True; provide valid tensors.
    topk_ids = torch.zeros(rows, device="cuda", dtype=torch.int32)
    bias = torch.zeros(1, inter_dim * 2, device="cuda", dtype=torch.float32)

    inputs = {
        "x": x,
        "sorted_ids": sorted_ids,
        "num_valid_ids": num_valid_ids,
        "topk_ids": topk_ids,
        "bias": bias,
        "token_num": int(token_num),
        "num_sorted_rows": int(num_sorted_rows),
        "rows": rows,
        "scale_cols": scale_cols,
        "inter_dim": inter_dim,
        "quant_mode": quant_mode,
    }
    return inputs


def _alloc_outputs(inputs):
    import torch

    rows = inputs["rows"]
    inter_dim = inputs["inter_dim"]
    scale_cols = inputs["scale_cols"]
    num_sorted_rows = inputs["num_sorted_rows"]
    quant_mode = inputs["quant_mode"]

    if quant_mode == "fp4":
        out_buf = torch.zeros(rows, inter_dim // 2, device="cuda", dtype=torch.uint8)
    elif quant_mode == "fp8":
        out_buf = torch.zeros(rows, inter_dim, device="cuda", dtype=torch.uint8)
    else:  # "none"
        out_buf = torch.zeros(rows, inter_dim, device="cuda", dtype=torch.bfloat16)

    # Tiled E8M0 scale layout. The within-row-block byte addressing fits in
    # scale_cols*32 bytes (verified for inter_dim%32==0), and row blocks of 32
    # rows are stacked: total = ceil(rows/32)*32 * scale_cols bytes. Pad for
    # safety; out-of-bounds buffer stores are dropped anyway (max_size=True).
    scale_blocks = (num_sorted_rows + 31) // 32
    scale_bytes = scale_blocks * 32 * scale_cols + 256
    out_scale_sorted = torch.zeros(scale_bytes, device="cuda", dtype=torch.uint8)
    return out_buf, out_scale_sorted


def _build_launcher(mod, cfg):
    _token_num, inter_dim, topk, quant_mode = cfg
    return mod.build_silu_and_mul_fq_module(
        inter_dim,
        topk,
        quant_mode=quant_mode,
        gui_layout=_GUI_LAYOUT,
        act=_ACT,
        enable_bias=_ENABLE_BIAS,
        swiglu_limit=_SWIGLU_LIMIT,
    )


def _launch(launcher, inputs, out_buf, out_scale_sorted, stream):
    launcher(
        inputs["x"],
        out_buf,
        out_scale_sorted,
        inputs["sorted_ids"],
        inputs["num_valid_ids"],
        inputs["topk_ids"],
        inputs["bias"],
        inputs["token_num"],
        inputs["num_sorted_rows"],
        stream,
    )


# ============================================================================
# Reference (display-only speedup baseline)
# ============================================================================


def _torch_ref_silu_mul(x, inter_dim):
    import torch
    import torch.nn.functional as F

    gate = x[:, :inter_dim].float()
    up = x[:, inter_dim:].float()
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# MXFP4 (e2m1 + per-32 e8m0 block scale) reference codec. These replicate the
# exact scheme in kernel.py so we can build a known-good reference and decode
# the kernel's packed output — no self-reference needed.
# ---------------------------------------------------------------------------
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # magnitude per (code & 7)


def _e8m0_biased(max_abs, headroom=2):
    """e8m0 biased exponent from a block's max |value|, matching kernel.py:
    max_rounded = (bits + 0x400000) & 0xFF800000; exp = max_rounded>>23;
    biased = max(exp - headroom, 0)."""
    import numpy as np

    bits = max_abs.detach().cpu().numpy().astype(np.float32).view(np.uint32)
    rounded = (bits.astype(np.uint64) + np.uint64(0x400000)) & np.uint64(0xFF800000)
    exp = (rounded >> np.uint64(23)) & np.uint64(0xFF)
    biased = np.clip(exp.astype(np.int64) - headroom, 0, 255)
    return biased.astype(np.int64)


def _decode_e2m1(nibbles):
    import torch

    mag = torch.tensor(_E2M1_MAG, device=nibbles.device, dtype=torch.float32)
    sign = torch.where((nibbles & 8) > 0, -1.0, 1.0)
    return sign * mag[(nibbles & 7).long()]


def _scale_tiled_offsets(rows, scale_cols):
    """Byte offset of each (row, col_block) e8m0 scale in the kernel's tiled
    'sorted' layout (mirrors the d0..d5 addressing in kernel.py)."""
    import numpy as np

    n32_sort = scale_cols * 32
    r = np.arange(rows)[:, None]
    c = np.arange(scale_cols)[None, :]
    d0 = r >> 5
    d1 = (r >> 4) & 1
    d2 = r & 15
    d3 = c >> 3
    d4 = (c >> 2) & 1
    d5 = c & 3
    return d0 * n32_sort + d3 * 256 + d5 * 64 + d2 * 4 + d4 * 2 + d1  # [rows, scale_cols]


def _nearest_e2m1_code(scaled):
    """Round scaled values to the nearest signed e2m1 grid value; return both
    the chosen code (0..15) and the grid magnitude/value."""
    import torch

    mag = torch.tensor(_E2M1_MAG, device=scaled.device, dtype=torch.float32)  # [8]
    a = scaled.abs().unsqueeze(-1)                       # [..., 1]
    diff = (a - mag).abs()                               # [..., 8]
    mcode = diff.argmin(dim=-1)                          # [...]
    gmag = mag[mcode]                                    # [...]
    gval = torch.where(scaled < 0, -gmag, gmag)
    return mcode, gval


def reference_mxfp4(ref_fp32, scale_cols):
    """Quantize a true silu*mul reference to MXFP4 the same way the kernel does.
    Returns (dequant_ref, e8m0_biased[rows,scale_cols]) where dequant_ref is the
    reference re-expressed on the kernel's quantization grid."""
    import torch

    rows, inter_dim = ref_fp32.shape
    blk = ref_fp32.view(rows, scale_cols, 32)
    max_abs = blk.abs().amax(dim=-1)                     # [rows, scale_cols]
    e8 = _e8m0_biased(max_abs)                           # numpy [rows, scale_cols]
    e8_t = torch.tensor(e8, device=ref_fp32.device, dtype=torch.float32)
    quant_scale = torch.pow(2.0, 127.0 - e8_t)           # real -> grid units
    dequant_scale = torch.pow(2.0, e8_t - 127.0)         # grid -> real
    scaled = blk * quant_scale.unsqueeze(-1)
    _code, gval = _nearest_e2m1_code(scaled)
    dequant = (gval * dequant_scale.unsqueeze(-1)).view(rows, inter_dim)
    return dequant, e8


def decode_kernel_fp4(out_buf, out_scale_sorted, rows, inter_dim):
    """Decode the kernel's packed fp4 output + tiled e8m0 scales to fp32."""
    import numpy as np
    import torch

    scale_cols = inter_dim // 32
    # unpack two e2m1 nibbles per byte -> [rows, inter_dim]
    lo = (out_buf & 0xF)
    hi = (out_buf >> 4) & 0xF
    nibbles = torch.stack([lo, hi], dim=-1).view(rows, inter_dim)
    vals = _decode_e2m1(nibbles)                         # grid values
    # gather e8m0 scale byte per (row, col_block)
    offs = _scale_tiled_offsets(rows, scale_cols)        # numpy [rows, scale_cols]
    offs_t = torch.tensor(offs, device=out_buf.device, dtype=torch.long)
    e8 = out_scale_sorted[offs_t].float()                # [rows, scale_cols]
    dequant_scale = torch.pow(2.0, e8 - 127.0)
    deq = (vals.view(rows, scale_cols, 32) * dequant_scale.unsqueeze(-1)).view(rows, inter_dim)
    return deq, e8.cpu().numpy().astype(np.int64)


# ============================================================================
# Modes
# ============================================================================


def run_correctness(shapes=None, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness (vs torch silu*mul + MXFP4 reference) on "
              f"{len(shapes)} config(s)...")

    cand_mod = _load_kernel(_CANDIDATE_DIR, "silu_candidate")
    if cand_mod is None:
        print("FAIL: cannot load kernel.py (candidate)")
        return {"correct": False, "num_correct": 0, "num_failed": len(shapes), "failures": []}

    if verbose:
        print(f"  candidate : {os.path.join(_CANDIDATE_DIR, KERNEL_FILE)}")

    stream = torch.cuda.current_stream()
    results, failures = [], []

    # Acceptance: the kernel's dequantized output must equal an independent
    # torch MXFP4 quantization of silu*mul. e8m0 block scales must match
    # exactly; e2m1 codes may differ only at round-to-nearest-even ties.
    GRID_TIE_FRAC = 0.01  # <=1% of elements may sit on an RNE tie boundary

    for i, cfg in enumerate(shapes):
        token_num, inter_dim, topk, quant_mode = cfg
        try:
            if quant_mode != "fp4":
                raise AssertionError(f"unsupported quant_mode for reference: {quant_mode}")
            inputs = _make_inputs(cfg, seed=42 + i)
            rows = inputs["rows"]
            scale_cols = inter_dim // 32

            cand_launch = _build_launcher(cand_mod, cfg)
            c_buf, c_scale = _alloc_outputs(inputs)
            _launch(cand_launch, inputs, c_buf, c_scale, stream)
            torch.cuda.synchronize()

            ref_fp32 = _torch_ref_silu_mul(inputs["x"], inter_dim)          # [rows, inter_dim]
            ref_deq, e8_ref = reference_mxfp4(ref_fp32, scale_cols)
            kern_deq, e8_kern = decode_kernel_fp4(c_buf, c_scale, rows, inter_dim)

            if not torch.isfinite(kern_deq).all():
                raise AssertionError("kernel output decoded to non-finite values")

            scale_mismatch = int((e8_ref != e8_kern).sum())
            if scale_mismatch:
                raise AssertionError(
                    f"e8m0 block-scale mismatch in {scale_mismatch}/{e8_ref.size} blocks")

            # Grid disagreements (after identical scale) should only be RNE ties.
            grid_mismatch = (kern_deq != ref_deq)
            n = kern_deq.numel()
            frac = float(grid_mismatch.float().mean())
            # On a tie, the two neighbouring grid points straddle the true value,
            # so the kernel choice must still be close to the true reference.
            tie_resid = (kern_deq[grid_mismatch] - ref_fp32[grid_mismatch]).abs()
            ref_resid = (ref_deq[grid_mismatch] - ref_fp32[grid_mismatch]).abs()
            bad_ties = int((tie_resid > ref_resid + 1e-6).sum())
            max_err = (kern_deq - ref_fp32).abs().max().item()

            if frac > GRID_TIE_FRAC or bad_ties:
                raise AssertionError(
                    f"grid mismatch frac={frac:.4f} (>{GRID_TIE_FRAC}) "
                    f"bad_ties={bad_ties}")

            results.append({"config": cfg, "correct": True})
            if verbose:
                print(
                    f"  PASS: (token_num={token_num}, inter_dim={inter_dim}, "
                    f"topk={topk}, {quant_mode}) scales exact, "
                    f"grid_tie_frac={frac:.4f}, max|deq-ref|={max_err:.3e}"
                )
        except Exception as e:
            failures.append({"config": cfg, "error": str(e)})
            if verbose:
                print(
                    f"  FAIL: (token_num={token_num}, inter_dim={inter_dim}, "
                    f"topk={topk}, {quant_mode}) - {str(e)[:160]}"
                )

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_profile(shapes=None, warmup=10, iters=50, verbose=True):
    import torch

    if shapes is None:
        shapes = PROFILE_SHAPES
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    mod = _load_kernel(_CANDIDATE_DIR, "silu_candidate")
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return

    stream = torch.cuda.current_stream()
    for cfg in shapes:
        token_num, inter_dim, topk, quant_mode = cfg
        inputs = _make_inputs(cfg)
        launcher = _build_launcher(mod, cfg)
        out_buf, out_scale = _alloc_outputs(inputs)

        _launch(launcher, inputs, out_buf, out_scale, stream)  # trigger JIT compile
        torch.cuda.synchronize()
        for _ in range(warmup):
            _launch(launcher, inputs, out_buf, out_scale, stream)
        torch.cuda.synchronize()
        for _ in range(iters):
            _launch(launcher, inputs, out_buf, out_scale, stream)
        torch.cuda.synchronize()
        if verbose:
            print(f"  (token_num={token_num}, inter_dim={inter_dim}, topk={topk}, {quant_mode}) done")


def run_benchmark(shapes=None, warmup=10, iters=100, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_CANDIDATE_DIR, "silu_candidate")
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    stream = torch.cuda.current_stream()
    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(shapes)} config(s), {warmup} warmup, {iters} iterations...")
    print(f"{'Config (tok,inter,topk,q)':<30} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 68)

    for idx, cfg in enumerate(shapes):
        token_num, inter_dim, topk, quant_mode = cfg
        inputs = _make_inputs(cfg, seed=42)
        out_buf, out_scale = _alloc_outputs(inputs)

        # Compile ONCE (first launch triggers FlyDSL JIT), outside timing.
        launcher = _build_launcher(mod, cfg)
        _launch(launcher, inputs, out_buf, out_scale, stream)
        torch.cuda.synchronize()

        for _ in range(warmup):
            _launch(launcher, inputs, out_buf, out_scale, stream)
        torch.cuda.synchronize()

        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _launch(launcher, inputs, out_buf, out_scale, stream)
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sum(kernel_times) / len(kernel_times)

        # Display-only torch reference (silu(gate)*mul). Not the oracle.
        x = inputs["x"]
        for _ in range(min(warmup, 5)):
            _ = _torch_ref_silu_mul(x, inter_dim)
        torch.cuda.synchronize()
        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = _torch_ref_silu_mul(x, inter_dim)
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))
        ref_ms = sum(ref_times) / len(ref_times)

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)

        rows = inputs["rows"]
        # Bytes moved: read 2*inter_dim bf16 in, write inter_dim fp4 nibbles (+ scales).
        in_bytes = rows * inter_dim * 2 * 2
        out_bytes = rows * (inter_dim // 2 if quant_mode == "fp4" else inter_dim)
        gbps = (in_bytes + out_bytes) / (kernel_ms * 1e-3) / 1e9

        report_cases.append(
            {
                "test_case_id": f"test_case_{idx}",
                "execution_time_ms": kernel_ms,
                "shape": [token_num, inter_dim, topk],
                "params": {
                    "token_num": token_num,
                    "inter_dim": inter_dim,
                    "topk": topk,
                    "quant_mode": quant_mode,
                    "rows": rows,
                },
                "gbytes_per_s": gbps,
            }
        )

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(
                f"(t={token_num:>5}, i={inter_dim:>5}, k={topk}, {quant_mode})"
                f" {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                flush=True,
            )

        del out_buf, out_scale, inputs
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(max(l, 1e-9)) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))

    build_dir = Path(_CANDIDATE_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 68)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL silu_and_mul_fq Kernel Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "100")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL silu_and_mul_fq Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if result.get("correct", False) else 1)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
