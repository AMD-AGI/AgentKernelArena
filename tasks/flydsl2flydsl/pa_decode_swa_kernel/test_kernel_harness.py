#!/usr/bin/env python3
"""GEAK execution-timing harness for FlyDSL pa_decode_swa (sliding-window
paged-attention decode) on AMD MI300X (gfx942).

This replaces the old "compile-smoke" stub that timed kernel COMPILATION.
Here we compile ONCE (the kernel's compile_* entry points are lru_cached and
the returned launchers are @flyc.jit, so repeated calls reuse the compiled
artifact) and then time real kernel EXECUTION with torch.cuda.Event.

Pipeline (per the kernel's intended usage, both stages run):
  stage 1: launch_pa_decode_sw        -> exp_sums / max_logits / tmp_out
  stage 2: launch_pa_decode_sw_reduce -> final output

Oracle: SELF-REFERENCE. We load the PRISTINE kernel from this task dir as the
oracle and the candidate kernel from $GEAK_WORK_DIR (fallback: task dir). The
two kernels are fed identical inputs and their final outputs must match
tightly. A full torch sliding-window paged-attention reference is impractical
for this packed-FP8 layout, so self-reference vs the original FlyDSL kernel is
the accepted correctness oracle.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

# ============================================================================
# Bootstrap / path discipline
# ============================================================================
KERNEL_FILE = "kernel.py"
_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_FLYDSL2_DIR = os.path.abspath(os.path.join(_TASK_DIR, ".."))  # has `kernels` pkg

# Make `from kernels import ...` work for kernel.py imports.
if _FLYDSL2_DIR not in sys.path:
    sys.path.insert(0, _FLYDSL2_DIR)


def _candidate_kernel_dir():
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir and os.path.isfile(os.path.join(work_dir, KERNEL_FILE)):
        return work_dir
    return _TASK_DIR


def _load_kernel(kernel_dir, alias):
    entry = os.path.join(kernel_dir, KERNEL_FILE)
    if not os.path.isfile(entry):
        raise FileNotFoundError(f"kernel.py not found in {kernel_dir}")
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Geometry constants (fixed by the kernel)
# ============================================================================
HEAD_SIZE = 128
QUERY_GROUP_SIZE = 16
KV_BLOCK_SIZE = 1024  # physical page size
KV_COMPUTE_BLOCK = 256  # tile size
X = 16  # FP8 elems per 16-byte K group (HEAD_SIZE // X = 8 he-groups)
QUERY_LENGTH = 1  # plain decode (one query token per sequence)

# ============================================================================
# Configs: (num_seqs, context_len, num_kv_heads, sliding_window)
# Chosen to COMPILE+RUN fast on gfx942. context_len kept to <=2 physical
# blocks; sliding windows realistic for SWA decode.
# ============================================================================
ALL_SHAPES = [
    (1, 1024, 1, 256),
    (1, 2048, 1, 512),
    (2, 1024, 2, 256),
    (4, 1024, 2, 256),
    (8, 1024, 4, 512),
]
HARNESS_SHAPES = ALL_SHAPES
PROFILE_SHAPES = ALL_SHAPES[:3]

# Tolerance vs the torch reference. Driven by fp8 e4m3 KV quantization (the
# reference dequantizes the SAME stored fp8 values the kernel reads) plus the
# bf16 final-output rounding; the residual is small but non-zero.
ATOL = 3e-2

FP8_MAX = 240.0


# ============================================================================
# Input construction
# ============================================================================
def _cdiv(a, b):
    return (a + b - 1) // b


def _quantize_fp8(t):
    """Quantize a float tensor to e4m3fnuz; return (fp8_tensor, dequant_scale)."""
    import torch

    amax = t.abs().max().item()
    scale = FP8_MAX / max(amax, 1e-12)  # quantization scale (real -> fp8 units)
    q = (t.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fnuz)
    dequant = 1.0 / scale  # fp8 units -> real
    return q, dequant


def _create_inputs(num_seqs, context_len, num_kv_heads, sliding_window, seed=42):
    import torch

    device = "cuda"
    torch.manual_seed(seed)

    num_query_heads = num_kv_heads * QUERY_GROUP_SIZE
    num_blocks_per_seq = _cdiv(context_len, KV_BLOCK_SIZE)
    total_blocks = num_seqs * num_blocks_per_seq + 2  # a couple of spares
    max_blocks_per_seq = num_blocks_per_seq + 2

    # --- Query: [num_seqs, num_query_heads, HEAD_SIZE] bf16 (query_length=1) ---
    query = torch.randn(
        num_seqs, num_query_heads, HEAD_SIZE, dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0)

    # --- K cache "real" values then quantize+repack to kernel layout ---
    #   logical:  [block, kv_head, token, head_dim]
    #   stored :  [block, kv_head, head_dim//X, token, X]   (X innermost)
    k_real = torch.randn(
        total_blocks, num_kv_heads, KV_BLOCK_SIZE, HEAD_SIZE,
        dtype=torch.float32, device=device,
    ).uniform_(-1.0, 1.0)
    k_q, key_scale = _quantize_fp8(k_real)
    key_cache = (
        k_q.view(total_blocks, num_kv_heads, KV_BLOCK_SIZE, HEAD_SIZE // X, X)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )  # [block, kv_head, head_dim//X, token, X]

    # --- V cache "real" values then quantize+repack ---
    #   logical:  [block, kv_head, token, head_dim]
    #   stored :  [block, kv_head, head_dim, token]   (token innermost)
    v_real = torch.randn(
        total_blocks, num_kv_heads, KV_BLOCK_SIZE, HEAD_SIZE,
        dtype=torch.float32, device=device,
    ).uniform_(-1.0, 1.0)
    v_q, value_scale = _quantize_fp8(v_real)
    value_cache = v_q.permute(0, 1, 3, 2).contiguous()  # [block, kv_head, head_dim, token]

    # --- block_tables: [num_seqs, max_blocks_per_seq] i32 ---
    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(num_seqs):
        for i in range(num_blocks_per_seq):
            block_tables[b, i] = b * num_blocks_per_seq + i

    context_lengths = torch.full((num_seqs,), context_len, dtype=torch.int32, device=device)

    key_scale_t = torch.tensor([key_scale], dtype=torch.float32, device=device)
    value_scale_t = torch.tensor([value_scale], dtype=torch.float32, device=device)

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "context_lengths": context_lengths,
        "key_scale": key_scale_t,
        "value_scale": value_scale_t,
        "num_seqs": num_seqs,
        "num_kv_heads": num_kv_heads,
        "num_query_heads": num_query_heads,
        "num_blocks_per_seq": num_blocks_per_seq,
        "max_blocks_per_seq": max_blocks_per_seq,
        "total_blocks": total_blocks,
        "sliding_window": sliding_window,
    }


# ============================================================================
# Build the full decode call (stage1 + reduce) for one config / one module.
# Returns (run_fn, output_tensor).
# ============================================================================
def _make_decode(mod, data):
    import torch

    num_seqs = data["num_seqs"]
    num_kv_heads = data["num_kv_heads"]
    num_query_heads = data["num_query_heads"]
    sliding_window = data["sliding_window"]
    eqgs = QUERY_LENGTH * QUERY_GROUP_SIZE  # extended query-group size = 16

    max_parts = mod.get_sw_max_context_partition_num(
        sliding_window, KV_COMPUTE_BLOCK, QUERY_LENGTH
    )
    mtp_groups = _cdiv(QUERY_LENGTH * QUERY_GROUP_SIZE, 16)

    # --- compile both stages (lru_cached inside the kernel module) ---
    stage1 = mod.compile_pa_decode_sw(
        sliding_window=sliding_window,
        query_group_size=QUERY_GROUP_SIZE,
        per_token_kv=False,
        query_length=QUERY_LENGTH,
        query_input_dtype="bf16",
        fuse_partitions=False,
    )["launch"]
    reduce = mod.compile_pa_decode_sw_reduce(
        max_context_partition_num=max_parts,
        query_seq_len=QUERY_LENGTH,
        query_group_size=QUERY_GROUP_SIZE,
        head_size=HEAD_SIZE,
        output_dtype_str="bf16",
    )["launch"]

    # --- intermediate / output tensors ---
    exp_sums = torch.zeros(num_seqs, num_kv_heads, max_parts, eqgs,
                           dtype=torch.float32, device="cuda")
    max_logits = torch.zeros_like(exp_sums)
    tmp_out = torch.zeros(num_seqs, num_kv_heads, max_parts, eqgs, HEAD_SIZE,
                          dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(num_seqs, QUERY_LENGTH, num_kv_heads, QUERY_GROUP_SIZE, HEAD_SIZE,
                         dtype=torch.bfloat16, device="cuda")

    q = data["query"]
    kc = data["key_cache"]
    vc = data["value_cache"]
    bt = data["block_tables"]
    cl = data["context_lengths"]
    ks = data["key_scale"]
    vs = data["value_scale"]

    # --- strides (element counts, matching kernel addressing) ---
    # query [num_seqs, num_query_heads, HEAD_SIZE]
    s_q_seq = num_query_heads * HEAD_SIZE
    s_q_head = HEAD_SIZE
    # key_cache [block, kv_head, head_dim//X, token, X]  (fp8 bytes == elems)
    s_k_block = num_kv_heads * (HEAD_SIZE // X) * KV_BLOCK_SIZE * X
    s_k_head = (HEAD_SIZE // X) * KV_BLOCK_SIZE * X
    # value_cache [block, kv_head, head_dim, token]
    s_v_block = num_kv_heads * HEAD_SIZE * KV_BLOCK_SIZE
    s_v_head = HEAD_SIZE * KV_BLOCK_SIZE
    # exp_sums / max_logits [num_seqs, kv_heads, max_parts, eqgs]
    s_es_seq = num_kv_heads * max_parts * eqgs
    s_es_head = max_parts * eqgs
    s_es_part = eqgs
    # tmp_out [num_seqs, kv_heads, max_parts, eqgs, head_size]
    s_to_seq = num_kv_heads * max_parts * eqgs * HEAD_SIZE
    s_to_head = max_parts * eqgs * HEAD_SIZE
    s_to_part = eqgs * HEAD_SIZE
    s_to_group = HEAD_SIZE
    # output [num_seqs, query_length, kv_heads, query_group_size, head_size]
    s_out_bs = QUERY_LENGTH * num_kv_heads * QUERY_GROUP_SIZE * HEAD_SIZE
    s_out_len = num_kv_heads * QUERY_GROUP_SIZE * HEAD_SIZE
    s_out_kv_head = QUERY_GROUP_SIZE * HEAD_SIZE
    s_out_group_size = HEAD_SIZE
    # block_tables [num_seqs, max_blocks_per_seq]
    s_bt_seq = data["max_blocks_per_seq"]
    # per-token kv scale strides (unused for per_token_kv=False)
    s_ks_block = 0
    s_ks_head = 0

    # grid for stage1 = (batch, kv_heads * mtp_groups, max_parts)
    gx = num_seqs
    gy = num_kv_heads * mtp_groups
    gz = max_parts

    stream = torch.cuda.current_stream()

    def _run():
        stage1(
            exp_sums, max_logits, tmp_out, output,
            q, kc, vc, bt, cl, ks, vs,
            s_q_seq, s_q_head,
            s_k_block, s_k_head,
            s_v_block, s_v_head,
            s_es_seq, s_es_head, s_es_part,
            s_to_seq, s_to_head, s_to_part, s_to_group,
            s_out_bs, s_out_len, s_out_kv_head, s_out_group_size,
            s_bt_seq,
            s_ks_block, s_ks_head,
            gx, gy, gz,
            stream,
        )
        reduce(
            output, exp_sums, max_logits, tmp_out,
            s_out_bs, s_out_len, s_out_kv_head, s_out_group_size,
            s_es_seq, s_es_head, s_es_part,
            s_to_seq, s_to_head, s_to_part, s_to_group,
            num_seqs, num_kv_heads,
            stream,
        )

    return _run, output


# ============================================================================
# Independent torch reference: dequantize the fp8 paged KV cache and compute
# sliding-window GQA decode attention in fp32. This is a real reference (not a
# self-reference): it never calls the FlyDSL kernel.
# ============================================================================
def reference_swa_decode(data):
    import torch

    ns = data["num_seqs"]
    kvh = data["num_kv_heads"]
    sw = data["sliding_window"]
    group = QUERY_GROUP_SIZE
    H = HEAD_SIZE
    softmax_scale = 1.0 / math.sqrt(H)

    q = data["query"].float()                  # [ns, kvh*group, H]; qh = h*group + g
    bt = data["block_tables"]
    cl = data["context_lengths"]
    ks = data["key_scale"].item()              # fp8 -> real multiplier (dequant)
    vs = data["value_scale"].item()

    # Dequantize whole caches to [block, kv_head, token, head_dim].
    kc = data["key_cache"].float() * ks        # [blk, kvh, H//X, KVB, X]
    kc = kc.permute(0, 1, 3, 2, 4).reshape(kc.shape[0], kvh, KV_BLOCK_SIZE, H)
    vc = data["value_cache"].float() * vs      # [blk, kvh, H, KVB]
    vc = vc.permute(0, 1, 3, 2).contiguous()   # [blk, kvh, KVB, H]

    out = torch.zeros(ns, QUERY_LENGTH, kvh, group, H,
                      dtype=torch.bfloat16, device=q.device)
    for s in range(ns):
        ctx = int(cl[s].item())
        pos = torch.arange(ctx, device=q.device)
        blk_idx = bt[s, pos // KV_BLOCK_SIZE].long()
        w_idx = (pos % KV_BLOCK_SIZE).long()
        K = kc[blk_idx, :, w_idx, :]           # [ctx, kvh, H]
        V = vc[blk_idx, :, w_idx, :]           # [ctx, kvh, H]
        qg = q[s].view(kvh, group, H)          # [kvh, group, H]
        K_kh = K.permute(1, 0, 2)              # [kvh, ctx, H]
        V_kh = V.permute(1, 0, 2)              # [kvh, ctx, H]
        scores = torch.einsum("kgd,ktd->kgt", qg, K_kh) * softmax_scale
        # Sliding-window mask for the decode query at position ctx-1: keep keys
        # with (ctx-1 - t) <= sw (mirrors the kernel's pos_diff >= sw+1 masking).
        keep = pos >= (ctx - 1 - sw)
        scores = scores.masked_fill(~keep.view(1, 1, ctx), float("-inf"))
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("kgt,ktd->kgd", p, V_kh)  # [kvh, group, H]
        out[s, 0] = o.to(torch.bfloat16)
    return out


# ============================================================================
# Correctness: candidate kernel vs the independent torch reference above.
# ============================================================================
def run_correctness(shapes=None, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES

    cand_mod = _load_kernel(_candidate_kernel_dir(), "pa_swa_candidate")

    print(f"Running correctness on {len(shapes)} shapes (vs torch reference)...")
    failures = []
    for i, (num_seqs, ctx, kvh, sw) in enumerate(shapes):
        try:
            data = _create_inputs(num_seqs, ctx, kvh, sw, seed=42 + i)

            run_c, out_c = _make_decode(cand_mod, data)
            run_c()
            torch.cuda.synchronize()
            cand = out_c.clone()

            ref = reference_swa_decode(data)

            if not torch.isfinite(cand.float()).all():
                raise AssertionError("candidate output has non-finite values")
            max_err = (cand.float() - ref.float()).abs().max().item()
            if max_err > ATOL:
                raise AssertionError(f"max_err={max_err:.4e} > {ATOL}")
            if verbose:
                print(f"  PASS: (seqs={num_seqs}, ctx={ctx}, kv_heads={kvh}, "
                      f"sw={sw}) max_err={max_err:.4e}")
        except Exception as e:
            failures.append((num_seqs, ctx, kvh, sw))
            if verbose:
                print(f"  FAIL: (seqs={num_seqs}, ctx={ctx}, kv_heads={kvh}, "
                      f"sw={sw}) - {str(e)[:140]}")

    print("-" * 62)
    if failures:
        print(f"Status: FAILED ({len(failures)}/{len(shapes)})")
        return {"correct": False, "num_correct": len(shapes) - len(failures),
                "num_failed": len(failures)}
    print("Status: ALL PASS")
    return {"correct": True, "num_correct": len(shapes), "num_failed": 0}


# ============================================================================
# Benchmark (compile ONCE, time EXECUTION via cuda events, median over iters)
# ============================================================================
def run_benchmark(shapes=None, warmup=10, iters=100, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_candidate_kernel_dir(), "pa_swa_candidate")

    latencies, speedups, report_cases = [], [], []
    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, "
          f"{iters} iterations...")
    print(f"{'Config (seqs,ctx,kvh,sw)':<34} {'FlyDSL(ms)':>12} {'Speedup':>10}")
    print("-" * 62)

    for idx, (num_seqs, ctx, kvh, sw) in enumerate(shapes):
        try:
            data = _create_inputs(num_seqs, ctx, kvh, sw, seed=42)
            run_fn, _ = _make_decode(mod, data)

            # one trial launch to surface any error before timing
            run_fn()
            torch.cuda.synchronize()

            for _ in range(warmup):
                run_fn()
            torch.cuda.synchronize()

            times = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run_fn()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            kernel_ms = sum(times) / len(times)
            status = ""
        except Exception as ex:
            kernel_ms = float("nan")
            status = f"  [FAIL: {str(ex)[:60]}]"

        speedup = 1.0  # no torch SWA paged-attention reference; report latency
        if kernel_ms == kernel_ms:  # not nan
            latencies.append(kernel_ms)
            speedups.append(speedup)

        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "params": {"num_seqs": num_seqs, "context_len": ctx,
                       "num_kv_heads": kvh, "sliding_window": sw},
        })
        if verbose:
            print(f"(seqs={num_seqs:>2}, ctx={ctx:>5}, kvh={kvh:>2}, sw={sw:>4})"
                  f"        {kernel_ms:>10.4f}  {speedup:>8.2f}x{status}", flush=True)
        torch.cuda.empty_cache()

    if not latencies:
        print("FAIL: no successful timing")
        print("GEAK_RESULT_LATENCY_MS=-1", flush=True)
        print("GEAK_RESULT_GEOMEAN_SPEEDUP=-1", flush=True)
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_candidate_kernel_dir()) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 62)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


def run_profile(shapes=None, warmup=5, iters=10, verbose=True):
    import torch

    if shapes is None:
        shapes = PROFILE_SHAPES
    mod = _load_kernel(_candidate_kernel_dir(), "pa_swa_candidate")
    print(f"Profile: {len(shapes)} config(s)")
    for (num_seqs, ctx, kvh, sw) in shapes:
        try:
            data = _create_inputs(num_seqs, ctx, kvh, sw, seed=42)
            run_fn, _ = _make_decode(mod, data)
            for _ in range(warmup + iters):
                run_fn()
            torch.cuda.synchronize()
            print(f"  OK: (seqs={num_seqs}, ctx={ctx}, kv_heads={kvh}, sw={sw})")
        except Exception as e:
            print(f"  FAIL: (seqs={num_seqs}, ctx={ctx}, kv_heads={kvh}, sw={sw}) "
                  f"- {str(e)[:100]}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FlyDSL pa_decode_swa kernel harness")
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--full-benchmark", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int,
                    default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "100")))
    args = ap.parse_args()

    print("=" * 62)
    print("FlyDSL pa_decode_swa (sliding-window paged-attention decode)")
    print("=" * 62)

    if args.correctness:
        r = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if r.get("correct", False) else 1)
    elif args.profile:
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
