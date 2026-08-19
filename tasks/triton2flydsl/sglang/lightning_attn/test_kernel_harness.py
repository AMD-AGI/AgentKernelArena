#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/lightning_attn.

Standalone harness for sglang's MiniMax/Bailing lightning-attention decode Triton
kernel (`linear_decode_forward_triton` -> `_linear_attn_decode_kernel`): a single
decode step of linear attention with a per-head exponentially-decayed KV state.
Per (batch, head):
  kv_new = outer(k, v) + exp(-slope) * kv_old
  out    = q . kv_new      (contraction over the qk dim)
  kv_old = kv_new          (in-place state update; fp32 state)

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton decode vs torch fp32 recurrence reference (output AND
                     updated KV state), assert close.
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

TASK_NAME = "triton2flydsl/sglang/lightning_attn"
SOURCE_FILE = os.path.join(TASK_DIR, "lightning_attn.py")

# [B, H, D]; D % BLOCK_SIZE == 0. Real MiniMax linear-attn head_dim is 96/128.
TEST_SHAPES = [
    {"B": 4, "H": 32, "D": 128, "block": 32, "dtype": "bf16"},
    {"B": 1, "H": 32, "D": 128, "block": 32, "dtype": "bf16"},
    {"B": 8, "H": 16, "D": 64, "block": 32, "dtype": "bf16"},
    {"B": 16, "H": 8, "D": 128, "block": 64, "dtype": "bf16"},
    {"B": 2, "H": 40, "D": 128, "block": 32, "dtype": "fp16"},
    {"B": 4, "H": 16, "D": 128, "block": 64, "dtype": "fp32"},
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_module():
    spec = importlib.util.spec_from_file_location("lightning_attn_src", SOURCE_FILE)
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
    B, H, D = cfg["B"], cfg["H"], cfg["D"]
    dt = getattr(torch, _DTYPES[cfg["dtype"]])
    q = torch.randn(B, H, 1, D, device=device, dtype=dt)
    k = torch.randn(B, H, 1, D, device=device, dtype=dt)
    v = torch.randn(B, H, 1, D, device=device, dtype=dt)
    # one cache slot per batch entry; fp32 state, prefilled with a prior history.
    kv_caches = torch.randn(B, H, D, D, device=device, dtype=torch.float32) * 0.1
    # per-head decay slopes (positive), MiniMax-style.
    slope_rate = (torch.arange(1, H + 1, device=device, dtype=torch.float32) / H)
    slot_idx = torch.arange(B, device=device, dtype=torch.int32)
    return q, k, v, kv_caches, slope_rate, slot_idx


def reference(q, k, v, kv_caches, slope_rate, slot_idx, cfg):
    import torch
    B, H, D = cfg["B"], cfg["H"], cfg["D"]
    out = torch.empty(B, H, D, device=q.device, dtype=torch.float32)
    kv_new_all = kv_caches.clone()
    for b in range(B):
        slot = int(slot_idx[b].item())
        if slot == -1:
            continue
        for h in range(H):
            ratio = torch.exp(-slope_rate[h])
            kq = k[b, h, 0].float()  # [D]
            vq = v[b, h, 0].float()  # [D]
            qq = q[b, h, 0].float()  # [D]
            kv_old = kv_caches[slot, h]  # [D, D]
            kv = torch.outer(kq, vq) + ratio * kv_old
            out[b, h] = (qq[:, None] * kv).sum(dim=0)
            kv_new_all[slot, h] = kv
    out_flat = out.reshape(B, H * D).to(q.dtype)
    return out_flat, kv_new_all


def _shape_of(cfg):
    return [cfg["B"], cfg["H"], cfg["D"]]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "linear_decode_forward_triton"), \
            "Missing entry linear_decode_forward_triton"
        assert hasattr(mod, "_linear_attn_decode_kernel"), \
            "Missing @triton.jit _linear_attn_decode_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    # Convention bf16 gate: isclose(atol=1e-2, rtol=1e-2) over >=99.9% of elements
    # OR normalized worst-element max|ref-out|/max|ref| <= 1e-2. The kernel computes
    # the k(x)v outer product in the input dtype (bf16), so a few output elements
    # exceed a raw 1e-2 while the normalized band (~4e-3) holds; fp32 path is tight.
    def _gate(out, ref):
        diff = (out.float() - ref.float()).abs().max().item()
        denom = ref.float().abs().max().item()
        rel = diff / denom if denom > 0 else diff
        frac = torch.isclose(out.float(), ref.float(),
                             atol=1e-2, rtol=1e-2).float().mean().item()
        return diff, rel, frac, (frac >= 0.999 or rel <= 1e-2)

    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        shape = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            q, k, v, kv_caches, slope_rate, slot_idx = make_inputs(cfg, "cuda")
            kv_clone = kv_caches.clone()
            out = _retry_oom(lambda: mod.linear_decode_forward_triton(
                q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE=cfg["block"]))
            torch.cuda.synchronize()
            ref_out, ref_kv = reference(q, k, v, kv_clone, slope_rate, slot_idx, cfg)
            finite = bool(torch.isfinite(out).all().item())
            odiff, orel, ofrac, ook = _gate(out, ref_out)
            kvdiff, kvrel, kvfrac, kvok = _gate(kv_caches, ref_kv)
            passed = finite and ook and kvok
            details.append({"shape_id": i + 1, "shape": shape, "dtype": cfg["dtype"],
                            "out_diff": odiff, "out_rel": orel, "kv_diff": kvdiff,
                            "kv_rel": kvrel, "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {shape} ({cfg['dtype']}): "
                               f"out_diff={odiff:.4e} out_rel={orel:.4e} "
                               f"kv_diff={kvdiff:.4e} finite={finite}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": shape, "error": str(e)})
            return False, f"Shape {i+1} {shape}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        params = {"shape": _shape_of(cfg), "dtype": cfg["dtype"]}
        try:
            torch.manual_seed(42 + ti)
            q, k, v, kv_caches, slope_rate, slot_idx = make_inputs(cfg, "cuda")
            kvc = kv_caches.clone()

            def prepare_fn():
                kvc.copy_(kv_caches)

            def fn():
                mod.linear_decode_forward_triton(
                    q, k, v, kvc, slope_rate, slot_idx, BLOCK_SIZE=cfg["block"])

            prepare_fn()
            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                prepare_fn()
                fn()
            torch.cuda.synchronize()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS,
                prepare_fn=prepare_fn,
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
                print(f"  shape {d['shape_id']} {d['shape']} {d['dtype']}: "
                      f"out_diff={d['out_diff']:.4e} out_rel={d['out_rel']:.4e} "
                      f"kv_diff={d['kv_diff']:.4e} -> "
                      f"{'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
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
