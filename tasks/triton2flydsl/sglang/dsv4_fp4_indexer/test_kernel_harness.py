#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/dsv4_fp4_indexer.

Standalone harness for sglang's DeepSeek-V4 FP4 indexer quantizer Triton kernels
(`quantize_fp4_indexer_tensor` -> `_quantize_fp4_indexer_kernel`, and
`store_fp4_index_k_cache` -> `_store_fp4_index_k_cache_kernel`): per 128-wide token
row, 4 groups of 32 -> UE8M0 (power-of-two) block scales packed into one int32, and
each value scaled by 2^(exp-127) then e2m1-encoded to a 4-bit code, two codes per
int8 byte (64 bytes/token). The store kernel pages the codes + scale bytes into a
KV cache. Pure integer/bitwise quantization -> validated BIT-EXACTLY on gfx942.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton quantize/store vs bit-exact torch reference (EXACT).
  --full-benchmark : graph-first GPU timing, write build/performance_report.json
"""
import sys
import os
import json
import time
import argparse
import importlib.util
from _aka_benchmark import benchmark_cuda_graph_or_events

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/sglang/dsv4_fp4_indexer"
SOURCE_FILE = os.path.join(TASK_DIR, "dsv4_fp4_indexer.py")
HEAD = 128  # the indexer key dim (fixed by the kernel: x.shape[-1] == 128)

# Token counts to quantize (last dim is always 128).
TEST_SHAPES = [
    {"N": 1, "dtype": "fp32"},
    {"N": 7, "dtype": "fp32"},
    {"N": 128, "dtype": "fp32"},
    {"N": 1000, "dtype": "fp32"},
    {"N": 64, "dtype": "bf16"},
    {"N": 256, "dtype": "fp16"},
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_module():
    spec = importlib.util.spec_from_file_location("dsv4_fp4_indexer_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err):
    return "out of memory" in str(err).lower()


def _retry_oom(fn):
    import torch
    delay = 1.0
    for attempt in range(MAX_OOM_RETRIES):
        try:
            return fn()
        except RuntimeError as e:
            if _is_oom(e) and attempt < MAX_OOM_RETRIES - 1:
                torch.cuda.empty_cache()
                time.sleep(delay)
                delay *= 2.0
                continue
            raise


def make_inputs(cfg, device="cuda"):
    import torch
    dt = getattr(torch, _DTYPES[cfg["dtype"]])
    # mix of magnitudes so all e2m1 levels / group scales are exercised
    x = torch.randn(cfg["N"], HEAD, device=device, dtype=dt) * 3.0
    return x


def _ceil_ue8m0_exp(sf):
    import torch
    bits = sf.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    exp = exp + (mant != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _e2m1_code(v):
    import torch
    ax = v.abs().clamp(max=6.0)
    idx = torch.zeros_like(v, dtype=torch.int32)
    for thr in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx = idx + (ax > thr).to(torch.int32)
    sign = ((v < 0) & (idx != 0)).to(torch.int32)
    return (idx | (sign << 3)).to(torch.uint8)


def reference(x):
    import torch
    N = x.shape[0]
    xf = x.float().contiguous()
    g = xf.view(N, 4, 32)
    amax = g.abs().amax(dim=2)  # [N, 4]
    sf = (amax / 6.0).clamp(min=1.0e-4)  # [N, 4]
    exp = _ceil_ue8m0_exp(sf).to(torch.int64)  # [N, 4]
    ps = exp[:, 0] | (exp[:, 1] << 8) | (exp[:, 2] << 16) | (exp[:, 3] << 24)
    ps = torch.where(ps >= 2 ** 31, ps - 2 ** 32, ps).to(torch.int32)  # [N]
    scale_exp_full = exp.repeat_interleave(32, dim=1).to(torch.int32)  # [N, 128]
    scale_full = (scale_exp_full << 23).view(torch.float32)
    v = xf / scale_full  # [N, 128]
    codes = _e2m1_code(v)  # [N, 128] uint8 (0..15)
    c0 = codes[:, 0::2]
    c1 = codes[:, 1::2]
    packed = (c0 & 0x0F) | ((c1 & 0x0F) << 4)  # uint8 [N, 64]
    x_fp4 = packed.view(torch.int8)
    return x_fp4, ps


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "quantize_fp4_indexer_tensor"), \
            "Missing entry quantize_fp4_indexer_tensor"
        assert hasattr(mod, "store_fp4_index_k_cache"), \
            "Missing entry store_fp4_index_k_cache"
        assert hasattr(mod, "_quantize_fp4_indexer_kernel"), \
            "Missing @triton.jit _quantize_fp4_indexer_kernel"
        assert hasattr(mod, "_store_fp4_index_k_cache_kernel"), \
            "Missing @triton.jit _store_fp4_index_k_cache_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def _store_reference(x, x_fp4, x_sf, loc, page_size, num_pages):
    import torch
    N = x_fp4.shape[0]
    cache = torch.zeros(num_pages, page_size * (64 + 4), device=x.device,
                        dtype=torch.uint8)
    k_u8 = x_fp4.view(torch.uint8)
    for t in range(N):
        cl = int(loc[t].item())
        page = cl // page_size
        off = cl % page_size
        cache[page, off * 64:off * 64 + 64] = k_u8[t]
        sf = int(x_sf[t].item()) & 0xFFFFFFFF
        sf_bytes = torch.tensor([(sf >> (j * 8)) & 0xFF for j in range(4)],
                                device=x.device, dtype=torch.uint8)
        base = page_size * 64 + off * 4
        cache[page, base:base + 4] = sf_bytes
    return cache


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        N = cfg["N"]
        try:
            torch.manual_seed(42 + i)
            x = make_inputs(cfg, "cuda")
            x_fp4, x_sf = _retry_oom(lambda: mod.quantize_fp4_indexer_tensor(x))
            torch.cuda.synchronize()
            ref_fp4, ref_sf = reference(x)
            fp4_ok = bool(torch.equal(x_fp4.cpu(), ref_fp4.cpu()))
            sf_ok = bool(torch.equal(x_sf.cpu(), ref_sf.cpu()))

            # also validate the paged cache store (bit-exact)
            page_size = 8
            num_pages = (N + page_size - 1) // page_size + 1
            loc = torch.randperm(num_pages * page_size, device="cuda")[:N].to(torch.int32)
            cache = torch.zeros(num_pages, page_size * (64 + 4), device="cuda",
                                dtype=torch.uint8)
            _retry_oom(lambda: mod.store_fp4_index_k_cache(
                x, cache, loc, page_size=page_size))
            torch.cuda.synchronize()
            ref_cache = _store_reference(x, ref_fp4, ref_sf, loc, page_size, num_pages)
            store_ok = bool(torch.equal(cache.cpu(), ref_cache.cpu()))

            passed = fp4_ok and sf_ok and store_ok
            details.append({"shape_id": i + 1, "N": N, "dtype": cfg["dtype"],
                            "fp4_ok": fp4_ok, "sf_ok": sf_ok, "store_ok": store_ok,
                            "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} N={N} ({cfg['dtype']}): "
                               f"fp4_ok={fp4_ok} sf_ok={sf_ok} "
                               f"store_ok={store_ok}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "N": N, "error": str(e)})
            return False, f"Shape {i+1} N={N}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        params = {"N": cfg["N"], "dtype": cfg["dtype"]}
        try:
            torch.manual_seed(42 + ti)
            x = make_inputs(cfg, "cuda")

            def fn():
                mod.quantize_fp4_indexer_tensor(x)

            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                fn()
            torch.cuda.synchronize()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS
            )
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": elapsed_ms,
                               **bench_meta,
                               "params": params})
        except Exception:
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": -1.0,
                               "benchmark_method": "benchmark_failed",
                               "benchmark_fallback_reason": "performance case failed before timing completed",
                               "params": params})
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(build_dir, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err,
                   "num_shapes": len(TEST_SHAPES), "details": details},
                  open(os.path.join(build_dir, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "passed" in d:
                print(f"  shape {d['shape_id']} N={d['N']} {d['dtype']}: "
                      f"fp4_ok={d['fp4_ok']} sf_ok={d['sf_ok']} "
                      f"store_ok={d['store_ok']} -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} N={d['N']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        json.dump(test_cases, open(os.path.join(build_dir, "performance_report.json"), "w"), indent=2)
        if test_cases:
            total = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} case(s), total {total:.4f} ms")
            for c in test_cases:
                print(f"  {c['test_case_id']} {c['params']}: {c['execution_time_ms']:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
