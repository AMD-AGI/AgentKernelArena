#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Test harness for MLA decode kernel
# Shapes source: /home/upandey/AIG-Eval/external_repos/aiter/op_tests/test_mla.py

import argparse
import json
import os
import sys
import math
from pathlib import Path

import torch
from _aka_benchmark import benchmark_cuda_graph_or_events_samples

# Newer aiter imports template-backed C++ interfaces eagerly and defaults their
# build cache to ~/.aiter. The Docker validator exposes a read-only HOME, so keep
# that cache local to this task before importing kernel.py/aiter.
os.environ.setdefault(
    "AITER_ROOT_DIR",
    str(Path(__file__).resolve().parent / "build" / "aiter_root"),
)


def benchmark_cuda_graph_or_events(*args, **kwargs):
    samples, metadata = benchmark_cuda_graph_or_events_samples(*args, **kwargs)
    values = sorted(samples)
    midpoint = len(values) // 2
    median_ms = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    return median_ms, metadata

# Ensure aiter is importable
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        os.path.dirname(os.path.abspath(__file__)),
    ),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kernel import decode_attention_fwd_grouped_rope

torch.set_default_device("cuda")

# --- Fixed constants ---
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# --- Config space (from test_mla.py defaults, decode path only) ---
# bf16/bf16 decode configs with supported nhead values
# The local Triton entrypoint is a decode-only, one-query-per-sequence operator.
CTX_LENS = [21, 64, 256, 512, 1200, 3200, 5200, 8192]
BATCH_SIZES = [1, 3, 5, 16, 32, 64, 128, 256]
NHEADS = [16, 128]

# Fixed params from test_mla.py defaults
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64

# Build ordered full case stream (same order as test_mla.py)
ALL_CONFIGS = []
for _nhead in NHEADS:
    for _ctx_len in CTX_LENS:
        for _batch_size in BATCH_SIZES:
            ALL_CONFIGS.append((_ctx_len, _batch_size, _nhead))


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def setup_inputs(ctx_len, batch_size, nhead, use_rope=False):
    """Set up one decode query per sequence for the local Triton wrapper."""
    torch.manual_seed(42)

    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    qk_head_dim = kv_lora_rank + qk_rope_head_dim  # 576
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim ** 0.5)
    num_kv_splits = 2

    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * ctx_len,
        ctx_len,
        dtype=torch.int32,
    )
    total_kv = batch_size * ctx_len
    kv_indices = torch.arange(total_kv, dtype=torch.int32)
    q = torch.randn((batch_size, nhead, qk_head_dim), dtype=torch.bfloat16)
    kv_cache = torch.randn((total_kv, qk_head_dim), dtype=torch.bfloat16)
    k_input = kv_cache.unsqueeze(1)
    v_input = kv_cache[:, :kv_lora_rank].contiguous().unsqueeze(1)
    # NaN sentinels make incomplete writes fail the finite-output check below.
    output = torch.full(
        (batch_size, nhead, v_head_dim), float("nan"), dtype=torch.bfloat16
    )
    attn_logits = torch.full(
        (batch_size, nhead, num_kv_splits, kv_lora_rank + 1),
        float("nan"),
        dtype=torch.bfloat16,
    )

    rotary_dim = qk_rope_head_dim
    if use_rope:
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )
        freqs = torch.outer(
            torch.arange(ctx_len + 1, dtype=torch.float32), inv_freq
        )
        cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(q.dtype)
        positions = torch.full((batch_size,), ctx_len, dtype=torch.int64)
        k_pe_tokens = torch.full(
            (batch_size, qk_rope_head_dim),
            float("nan"),
            dtype=kv_cache.dtype,
        )
    else:
        cos_sin_cache = None
        positions = None
        k_pe_tokens = None

    return {
        "q": q,
        "k_input": k_input,
        "v_input": v_input,
        "output": output,
        "attn_logits": attn_logits,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "num_kv_splits": num_kv_splits,
        "v_head_dim": v_head_dim,
        "sm_scale": sm_scale,
        "kv_lora_rank": kv_lora_rank,
        "rotary_dim": rotary_dim,
        "cos_sin_cache": cos_sin_cache,
        "positions": positions,
        "k_pe_tokens": k_pe_tokens,
        "use_rope": use_rope,
    }


def run_kernel(inputs):
    """Run the declared task-local MLA decode wrapper."""
    decode_attention_fwd_grouped_rope(
        inputs["q"],
        inputs["k_input"],
        inputs["v_input"],
        inputs["output"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["k_pe_tokens"],
        inputs["kv_lora_rank"],
        inputs["rotary_dim"] if inputs["use_rope"] else None,
        inputs["cos_sin_cache"],
        inputs["positions"],
        inputs["attn_logits"],
        inputs["num_kv_splits"],
        sm_scale=inputs["sm_scale"],
        logit_cap=0.0,
        use_rope=inputs["use_rope"],
    )
    return inputs["output"]


def apply_rope_ref(x, cos_sin_cache, positions, rotary_dim):
    """Apply GPT-J-style RoPE using an independent PyTorch implementation."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    while cos.ndim < x_rot.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    x_even = x_rot[..., ::2]
    x_odd = x_rot[..., 1::2]
    x_rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    result = x_rot * cos + x_rotated * sin
    return torch.cat((result, x_pass), dim=-1)


def run_ref(inputs):
    """Independent PyTorch grouped-attention reference."""
    q = inputs["q"]
    batch_size = q.shape[0]
    ctx_len = inputs["kv_indices"].numel() // batch_size
    token_ids = inputs["kv_indices"].long()
    keys = inputs["k_input"][token_ids, 0].view(batch_size, ctx_len, -1)
    values = inputs["v_input"][token_ids, 0].float().view(batch_size, ctx_len, -1)

    expected_k_pe_tokens = None
    if inputs["use_rope"]:
        rank = inputs["kv_lora_rank"]
        q = q.clone()
        keys = keys.clone()
        q[..., rank:] = apply_rope_ref(
            q[..., rank:],
            inputs["cos_sin_cache"],
            inputs["positions"],
            inputs["rotary_dim"],
        )
        expected_k_pe_tokens = apply_rope_ref(
            keys[:, -1, rank:],
            inputs["cos_sin_cache"],
            inputs["positions"],
            inputs["rotary_dim"],
        )
        keys[:, -1, rank:] = expected_k_pe_tokens

    scores = torch.einsum("bhd,btd->bht", q.float(), keys.float())
    probabilities = torch.softmax(scores * inputs["sm_scale"], dim=-1)
    output = torch.einsum("bht,btd->bhd", probabilities, values).to(
        inputs["output"].dtype
    )
    return output, expected_k_pe_tokens


def check_correctness_val(out_ref, out_asm):
    """Reject non-finite or unbounded errors while retaining the source tolerance."""
    finite = torch.isfinite(out_ref).all() & torch.isfinite(out_asm).all()
    is_close = torch.isclose(out_ref, out_asm, rtol=1e-2, atol=1e-2)
    err_ratio = (~is_close).float().mean().item()
    # A small fraction of BF16 reduction results can straddle the primary
    # tolerance boundary. Bound every such outlier so the ratio allowance can
    # never hide sparse garbage.
    within_error_bound = torch.isclose(out_ref, out_asm, rtol=2e-2, atol=2e-2)

    # Also compute cos_diff for reporting
    x, y = out_ref.double(), out_asm.double()
    if finite.item():
        cos_diff = 1 - 2 * (x * y).sum().item() / max(
            (x * x + y * y).sum().item(), 1e-12
        )
    else:
        cos_diff = float("inf")

    passed = bool(
        finite.item()
        and err_ratio <= 0.05
        and within_error_bound.all().item()
        and cos_diff <= 1e-3
    )
    return passed, err_ratio, cos_diff


def benchmark_kernel(inputs):
    """Benchmark the MLA decode kernel with graph replay when supported."""
    def fn():
        return run_kernel(inputs)

    return benchmark_cuda_graph_or_events(
        fn, warmup=WARMUP, repetition=ITERATIONS
    )


def config_str(cfg):
    ctx_len, batch_size, nhead = cfg
    return "ctx={} bs={} nhead={}".format(ctx_len, batch_size, nhead)


def mode_correctness(indices):
    print("Running correctness check on {} configs...".format(len(indices)))
    all_pass = True
    for case_number, idx in enumerate(indices):
        cfg = ALL_CONFIGS[idx]
        ctx_len, batch_size, nhead = cfg
        use_rope = case_number % 2 == 1
        label = "{} rope={}".format(config_str(cfg), use_rope)
        try:
            inputs = setup_inputs(ctx_len, batch_size, nhead, use_rope=use_rope)
            out_asm = run_kernel(inputs)
            out_ref, expected_k_pe_tokens = run_ref(inputs)
            passed, err_ratio, cos_diff = check_correctness_val(out_ref, out_asm)
            rope_err_ratio = None
            if use_rope:
                rope_passed, rope_err_ratio, _ = check_correctness_val(
                    expected_k_pe_tokens, inputs["k_pe_tokens"]
                )
                passed = passed and rope_passed
            rope_result = (
                " rope_err_ratio={:.4f}".format(rope_err_ratio)
                if rope_err_ratio is not None
                else ""
            )
            if passed:
                print("  [{}] {}  err_ratio={:.4f} cos_diff={:.2e}{}  PASS".format(
                    idx, label, err_ratio, cos_diff, rope_result))
            else:
                print("  [{}] {}  err_ratio={:.4f} cos_diff={:.2e}{}  FAIL".format(
                    idx, label, err_ratio, cos_diff, rope_result))
                all_pass = False
        except Exception as e:
            print("  [{}] {}  ERROR: {}".format(idx, label, e))
            all_pass = False
        finally:
            torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def mode_benchmark(indices):
    print("Running benchmark on {} configs...".format(len(indices)))
    latencies = []
    methods = []
    report_cases = []
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ctx_len, batch_size, nhead = cfg
        label = config_str(cfg)
        try:
            inputs = setup_inputs(ctx_len, batch_size, nhead)
            ms, metadata = benchmark_kernel(inputs)
            print("  {}  {:.4f}ms".format(label, ms))
            latencies.append(ms)
            methods.append(metadata["benchmark_method"])
            report_cases.append({
                "test_case_id": label,
                "params": {
                    "ctx_len": ctx_len,
                    "batch_size": batch_size,
                    "nhead": nhead,
                },
                "execution_time_ms": ms,
                **metadata,
            })
        except Exception as e:
            print("  {}  ERROR: {}".format(label, e))
            all_pass = False
        finally:
            torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))
    report_path = Path("build/performance_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_cases, indent=2))
    if latencies:
        geo_mean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
        print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo_mean))
        print("GEAK_BENCHMARK_METHOD={}".format(
            methods[0] if len(set(methods)) == 1 else "mixed:" + ",".join(sorted(set(methods)))
        ))
    else:
        print("No successful benchmarks")
        sys.exit(1)
    if not all_pass:
        print("BENCHMARK FAILED")
        sys.exit(1)


def mode_profile(indices):
    print("Running profile on {} configs...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ctx_len, batch_size, nhead = cfg
        label = config_str(cfg)
        try:
            inputs = setup_inputs(ctx_len, batch_size, nhead)
            out_asm = run_kernel(inputs)
            print("  {}  OK".format(label))
        except Exception as e:
            print("  {}  ERROR: {}".format(label, e))
        finally:
            torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))


def main():
    parser = argparse.ArgumentParser(description="MLA decode kernel test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=None, help="Number of benchmark iterations (overrides GEAK_BENCHMARK_ITERATIONS env var)")
    args = parser.parse_args()
    if args.iterations is not None:
        global ITERATIONS
        ITERATIONS = args.iterations

    total = len(ALL_CONFIGS)
    print("Total configs: {}".format(total))

    if args.correctness:
        # Cover both head-count branches and the full context/batch range without
        # turning an independent O(B*H*T*D) PyTorch oracle into an hour-long test.
        indices = _pick(ALL_CONFIGS, 16)
        mode_correctness(indices)
    elif args.benchmark:
        indices = list(range(total))  # use all configs so benchmark matches full-benchmark
        mode_benchmark(indices)
    elif args.full_benchmark:
        indices = list(range(total))
        mode_benchmark(indices)
    elif args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        mode_profile(indices)


if __name__ == "__main__":
    main()
