#!/usr/bin/env python3
"""
Test harness for fused_kv_cache kernel (fused_qk_rope_cat_and_cache_mla).
Modes: --correctness, --benchmark, --full-benchmark, --profile
"""

import os
import sys
import argparse
import math

# Ensure the repo root is on sys.path so op_tests and aiter are importable
REPO_ROOT = os.environ.get(
    "GEAK_REPO_ROOT",
    os.path.dirname(os.path.abspath(__file__)),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import triton

# ── imports from the repo ──────────────────────────────────────────────
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle
from op_tests.triton_tests.test_rope import generate_rope_inputs
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.utils._triton import arch_info

# ── constants ──────────────────────────────────────────────────────────
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ── full config list (matches test_fused_qk_rope_cat_and_cache_mla parametrize order) ──
# Parametrize order (outermost first -> innermost last):
#   dtype, cache_dtype, reuse_freqs_front_part, rotate_style,
#   num_kv_cahce_tokens, D_lora, D_q_nope, D, KH, QH_per_KH, T

_T_vals = [1, 2, 4, 2048]
_QH_per_KH_vals = [1, 16]
_KH_vals = [1, 8]
_D_vals = [128]
_D_q_nope_vals = [128]
_D_lora_vals = [512]
_num_kv_cache_tokens_vals = [16384]
_rotate_style_vals = [RotateStyle.GPTJ, RotateStyle.NEOX]
_reuse_freqs_front_part_vals = [False, True]
_cache_dtype_vals = [torch.bfloat16, torch.uint8]
_dtype_vals = [torch.bfloat16]


def _build_all_configs():
    """Build ordered config list matching pytest parametrize order."""
    configs = []
    for dtype in _dtype_vals:
        for cache_dtype in _cache_dtype_vals:
            for reuse_freqs_front_part in _reuse_freqs_front_part_vals:
                for rotate_style in _rotate_style_vals:
                    for num_kv_cache_tokens in _num_kv_cache_tokens_vals:
                        for D_lora in _D_lora_vals:
                            for D_q_nope in _D_q_nope_vals:
                                for D in _D_vals:
                                    for KH in _KH_vals:
                                        for QH_per_KH in _QH_per_KH_vals:
                                            for T in _T_vals:
                                                configs.append(
                                                    dict(
                                                        T=T,
                                                        QH_per_KH=QH_per_KH,
                                                        KH=KH,
                                                        D=D,
                                                        D_q_nope=D_q_nope,
                                                        D_lora=D_lora,
                                                        num_kv_cache_tokens=num_kv_cache_tokens,
                                                        rotate_style=rotate_style,
                                                        reuse_freqs_front_part=reuse_freqs_front_part,
                                                        cache_dtype=cache_dtype,
                                                        dtype=dtype,
                                                    )
                                                )
    return configs


ALL_CONFIGS = _build_all_configs()


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs))), configs
    n = len(configs)
    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    return indices, [configs[i] for i in indices]


def _config_label(cfg):
    rs = "NEOX" if cfg["rotate_style"] == RotateStyle.NEOX else "GPTJ"
    cd = "u8" if cfg["cache_dtype"] == torch.uint8 else "bf16"
    return (
        f"T={cfg['T']} QH_per_KH={cfg['QH_per_KH']} KH={cfg['KH']} "
        f"D={cfg['D']} D_q_nope={cfg['D_q_nope']} D_lora={cfg['D_lora']} "
        f"rot={rs} reuse={cfg['reuse_freqs_front_part']} cache={cd}"
    )


def _setup_inputs(cfg):
    """Build inputs for fused_qk_rope_cat_and_cache_mla, matching the test."""
    torch.manual_seed(42)
    T = cfg["T"]
    QH_per_KH = cfg["QH_per_KH"]
    KH = cfg["KH"]
    D = cfg["D"]
    D_q_nope = cfg["D_q_nope"]
    D_lora = cfg["D_lora"]
    num_kv_cache_tokens = cfg["num_kv_cache_tokens"]
    rotate_style = cfg["rotate_style"]
    reuse_freqs_front_part = cfg["reuse_freqs_front_part"]
    cache_dtype = cfg["cache_dtype"]
    dtype = cfg["dtype"]

    _, _, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1, T, KH, QH_per_KH, D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=True,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    q = torch.randn((T, QH_per_KH * KH, D_q_nope + D), dtype=dtype, device="cuda")
    q_nope, q_pe = q.split((D_q_nope, D), dim=-1)
    k_lora = torch.randn((T, KH, D_lora), dtype=dtype, device="cuda") / (
        20 if cache_dtype == torch.uint8 else 1
    )
    k_pe = torch.randn((T, KH, D), dtype=dtype, device="cuda") / (
        20 if cache_dtype == torch.uint8 else 1
    )

    kv_cache = torch.zeros(
        (num_kv_cache_tokens, KH, D_lora + D), dtype=cache_dtype, device="cuda"
    )

    if cache_dtype == torch.uint8:
        if arch_info.get_arch() in ["gfx950"]:
            cache_dtype_actual = torch.float8_e4m3fn
        else:
            cache_dtype_actual = torch.float8_e4m3fnuz
        k_scale = torch.randn([1], dtype=torch.float32, device="cuda")[0]
    else:
        cache_dtype_actual = None
        k_scale = torch.ones([1], dtype=torch.float32, device="cuda")[0]

    slot_mapping = torch.randperm(T, device="cuda")

    return dict(
        q_nope=q_nope.contiguous(),
        q_pe=q_pe.contiguous(),
        k_lora=k_lora,
        k_pe=k_pe,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        cos=cos,
        sin=sin,
        k_scale=k_scale,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        cache_dtype=cache_dtype,
        cache_dtype_actual=cache_dtype_actual,
        dtype=dtype,
        freqs=freqs,
        offsets=offsets,
        T=T,
        QH_per_KH=QH_per_KH,
        KH=KH,
        D=D,
        D_q_nope=D_q_nope,
        D_lora=D_lora,
    )


def _run_kernel(inp):
    """Run the fused kernel and return outputs."""
    kv_cache_clone = inp["kv_cache"].clone()
    if inp["cache_dtype"] == torch.uint8:
        kv_cache_clone = kv_cache_clone.view(inp["cache_dtype_actual"])

    result = fused_qk_rope_cat_and_cache_mla(
        inp["q_nope"],
        inp["q_pe"],
        inp["k_lora"],
        inp["k_pe"],
        kv_cache_clone,
        inp["slot_mapping"],
        inp["positions"],
        inp["cos"],
        inp["sin"],
        inp["k_scale"],
        (inp["rotate_style"] == RotateStyle.NEOX),
        num_decode_toks_for_zeros=inp["T"],
        apply_scale=(inp["k_pe"].dtype != inp["kv_cache"].dtype),
        q_out=None,
        decode_q_pe_out=None,
        k_pe_out=None,
    )
    # Kernel returns (q_out, decode_q_pe_out, k_pe_out, kv_cache[, q_nope_zeros_out])
    if len(result) == 5:
        q_out, decode_q_pe_out, k_pe_out, _kv, q_nope_zeros_out = result
    else:
        q_out, decode_q_pe_out, k_pe_out, _kv = result
        q_nope_zeros_out = torch.zeros(0)
    return q_out, decode_q_pe_out, k_pe_out, q_nope_zeros_out, kv_cache_clone


def _run_reference(inp):
    """Run the reference (torch) implementation."""
    T = inp["T"]
    QH_per_KH = inp["QH_per_KH"]
    KH = inp["KH"]
    D = inp["D"]
    D_q_nope = inp["D_q_nope"]
    D_lora = inp["D_lora"]
    dtype = inp["dtype"]
    cache_dtype = inp["cache_dtype"]
    rotate_style = inp["rotate_style"]
    reuse_freqs_front_part = inp["reuse_freqs_front_part"]

    freqs = inp["freqs"]
    positions = inp["positions"]
    offsets = inp["offsets"]

    ref_freqs = freqs[
        positions if offsets is None else torch.add(positions, offsets)
    ].squeeze(-2)

    torch_q_nope = inp["q_nope"]
    torch_q_pe = inp["q_pe"].clone()
    torch_k_lora = inp["k_lora"].clone()
    torch_k_pe = inp["k_pe"].clone()

    torch_q_pe = ref_rope_sbhd_fwd(
        torch_q_pe.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)
    torch_k_pe_roped = ref_rope_sbhd_fwd(
        torch_k_pe.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)

    kv_cache_clone = inp["kv_cache"].clone()
    kv_cache_og_dtype = kv_cache_clone.dtype
    k_scale = inp["k_scale"]
    slot_mapping = inp["slot_mapping"]

    if cache_dtype == torch.uint8:
        cache_dtype_actual = inp["cache_dtype_actual"]
        kv_cache_clone = kv_cache_clone.view(cache_dtype_actual)
        torch_k_lora_scaled = (torch_k_lora.to(torch.float32) / k_scale).to(cache_dtype_actual)
        torch_k_pe_scaled = (torch_k_pe_roped.to(torch.float32) / k_scale).to(cache_dtype_actual)
    else:
        torch_k_lora_scaled = torch_k_lora
        torch_k_pe_scaled = torch_k_pe_roped

    torch_q = torch.cat((torch_q_nope, torch_q_pe), dim=-1)
    torch_decode_q_pe = torch_q_pe
    torch_zeros = torch.zeros(((T, QH_per_KH * KH, D_lora)), dtype=dtype, device="cuda")
    kv_cache_clone[slot_mapping, :, :] = torch.cat(
        (torch_k_lora_scaled, torch_k_pe_scaled), dim=-1
    )
    kv_cache_clone = kv_cache_clone.view(kv_cache_og_dtype)

    return torch_q, torch_decode_q_pe, torch_k_pe_roped, torch_zeros, kv_cache_clone


def _check_correctness_single(cfg):
    """Run correctness check for a single config. Returns True on pass."""
    inp = _setup_inputs(cfg)
    triton_q, triton_decode_q_pe, triton_k_pe, triton_zeros, triton_kv_cache = _run_kernel(inp)
    torch_q, torch_decode_q_pe, torch_k_pe, torch_zeros, torch_kv_cache = _run_reference(inp)

    kv_cache_og_dtype = inp["kv_cache"].dtype
    cache_dtype = inp["cache_dtype"]
    dtype = inp["dtype"]
    slot_mapping = inp["slot_mapping"]

    triton_kv_cache_view = triton_kv_cache.view(kv_cache_og_dtype)

    torch.testing.assert_close(torch_q, triton_q, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_decode_q_pe, triton_decode_q_pe, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_k_pe, triton_k_pe, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_zeros, triton_zeros, atol=0.1, rtol=0.1)

    if cache_dtype == torch.uint8:
        cache_dtype_actual = inp["cache_dtype_actual"]
        ref_kv = torch_kv_cache.view(cache_dtype_actual).to(dtype)
        tri_kv = triton_kv_cache_view.view(cache_dtype_actual).to(dtype)
    else:
        ref_kv = torch_kv_cache
        tri_kv = triton_kv_cache_view

    torch.testing.assert_close(
        ref_kv[slot_mapping, :, :],
        tri_kv[slot_mapping, :, :],
        atol=1e-1,
        rtol=1e-1,
    )
    torch.testing.assert_close(ref_kv, tri_kv, atol=1e-1, rtol=1e-1)
    return True


def _benchmark_single(cfg):
    """Benchmark a single config. Returns median latency in ms."""
    inp = _setup_inputs(cfg)

    # Build a closure for the kernel call
    def _kernel_fn():
        kv_cache_clone = inp["kv_cache"].clone()
        if inp["cache_dtype"] == torch.uint8:
            kv_cache_clone = kv_cache_clone.view(inp["cache_dtype_actual"])
        fused_qk_rope_cat_and_cache_mla(
            inp["q_nope"],
            inp["q_pe"],
            inp["k_lora"],
            inp["k_pe"],
            kv_cache_clone,
            inp["slot_mapping"],
            inp["positions"],
            inp["cos"],
            inp["sin"],
            inp["k_scale"],
            (inp["rotate_style"] == RotateStyle.NEOX),
            num_decode_toks_for_zeros=inp["T"],
            apply_scale=(inp["k_pe"].dtype != inp["kv_cache"].dtype),
            q_out=None,
            decode_q_pe_out=None,
            k_pe_out=None,
        )

    # Warmup
    for _ in range(WARMUP):
        _kernel_fn()
    torch.cuda.synchronize()

    # Timed iterations using GPU events
    times = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _kernel_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median_ms = times[len(times) // 2]
    return median_ms


def main():
    parser = argparse.ArgumentParser(description="Test harness for fused_kv_cache")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not any([args.correctness, args.benchmark, args.full_benchmark, args.profile]):
        parser.print_help()
        sys.exit(1)

    if args.correctness:
        indices, configs = _pick(ALL_CONFIGS, 25)
        print(f"Running correctness on {len(configs)} configs...")
        for i, (idx, cfg) in enumerate(zip(indices, configs)):
            label = _config_label(cfg)
            try:
                _check_correctness_single(cfg)
                print(f"  [{i+1}/{len(configs)}] PASS  {label}")
            except Exception as e:
                print(f"  [{i+1}/{len(configs)}] FAIL  {label}: {e}")
                print(f"GEAK_SHAPES_USED={indices}")
                sys.exit(1)
        print("All correctness checks passed.")
        print(f"GEAK_SHAPES_USED={indices}")

    if args.profile:
        indices, configs = _pick(ALL_CONFIGS, 5)
        print(f"Running profile on {len(configs)} configs...")
        latencies = []
        for i, (idx, cfg) in enumerate(zip(indices, configs)):
            label = _config_label(cfg)
            ms = _benchmark_single(cfg)
            latencies.append(ms)
            print(f"  {label}  {ms:.4f}ms")
        geo_mean = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
        print(f"GEAK_SHAPES_USED={indices}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")

    if args.benchmark:
        indices, configs = _pick(ALL_CONFIGS, 25)
        print(f"Running benchmark on {len(configs)} configs...")
        latencies = []
        for i, (idx, cfg) in enumerate(zip(indices, configs)):
            label = _config_label(cfg)
            ms = _benchmark_single(cfg)
            latencies.append(ms)
            print(f"  {label}  {ms:.4f}ms")
        geo_mean = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
        print(f"GEAK_SHAPES_USED={indices}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")

    if args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        configs = ALL_CONFIGS
        print(f"Running full benchmark on {len(configs)} configs...")
        latencies = []
        for i, (idx, cfg) in enumerate(zip(indices, configs)):
            label = _config_label(cfg)
            ms = _benchmark_single(cfg)
            latencies.append(ms)
            print(f"  {label}  {ms:.4f}ms")
        geo_mean = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
        print(f"GEAK_SHAPES_USED={indices}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


if __name__ == "__main__":
    main()
