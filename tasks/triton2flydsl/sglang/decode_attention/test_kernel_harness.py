#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/decode_attention.

Standalone harness for sglang's two-stage flash-decoding Triton kernels
(`decode_attention_fwd` -> stage1 normal/grouped + stage2 combine): single-query
(decode) attention where each batch's KV range is split into num_kv_splits chunks
(stage 1 computes per-split partial softmax-V + LSE) and recombined (stage 2),
with grouped-query attention and paged page size = 1.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton decode attn vs torch fp32 softmax SDPA, close.
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

TASK_NAME = "triton2flydsl/sglang/decode_attention"
SOURCE_FILE = os.path.join(TASK_DIR, "decode_attention.py")
MAX_KV_SPLITS = 8

# Per-batch KV seq lens + head config. kv_group=1 hits the MHA normal stage1;
# kv_group>1 hits the grouped stage1 (GQA/MQA).
TEST_SHAPES = [
    {"seqs": [128, 256, 64, 200], "head": 32, "kv_head": 32, "Lk": 128, "Lv": 128},  # MHA
    {"seqs": [512, 300], "head": 32, "kv_head": 8, "Lk": 128, "Lv": 128},  # GQA
    {"seqs": [128, 128, 77], "head": 16, "kv_head": 1, "Lk": 128, "Lv": 128},  # MQA
    {"seqs": [256, 128], "head": 16, "kv_head": 2, "Lk": 64, "Lv": 64},  # GQA d=64
    {"seqs": [1000], "head": 8, "kv_head": 8, "Lk": 128, "Lv": 128},  # MHA long (splits)
    {"seqs": [100, 33], "head": 28, "kv_head": 4, "Lk": 128, "Lv": 128},  # GQA ragged
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("decode_attention_src", SOURCE_FILE)
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
    seqs = cfg["seqs"]
    batch = len(seqs)
    H, KVH, Lk, Lv = cfg["head"], cfg["kv_head"], cfg["Lk"], cfg["Lv"]
    total = sum(seqs)
    dt = torch.bfloat16
    q = torch.randn(batch, H, Lk, device=device, dtype=dt)
    k_buffer = torch.randn(total, KVH, Lk, device=device, dtype=dt)
    v_buffer = torch.randn(total, KVH, Lv, device=device, dtype=dt)
    o = torch.empty(batch, H, Lv, device=device, dtype=dt)
    kv_indptr = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(seqs, device=device, dtype=torch.int32), 0)
    kv_indices = torch.arange(total, device=device, dtype=torch.int64)
    num_kv_splits = torch.full((batch,), MAX_KV_SPLITS, device=device, dtype=torch.int32)
    attn_logits = torch.empty(batch, H, MAX_KV_SPLITS, Lv, device=device,
                              dtype=torch.float32)
    attn_lse = torch.empty(batch, H, MAX_KV_SPLITS, device=device, dtype=torch.float32)
    return (q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
            num_kv_splits)


def reference(q, k_buffer, v_buffer, kv_indices, cfg, sm_scale=None):
    import torch
    seqs = cfg["seqs"]
    H, KVH, Lk, Lv = cfg["head"], cfg["kv_head"], cfg["Lk"], cfg["Lv"]
    group = H // KVH
    if sm_scale is None:
        sm_scale = 1.0 / (Lk ** 0.5)
    out = torch.empty(len(seqs), H, Lv, device=q.device, dtype=torch.float32)
    start = 0
    for b, L in enumerate(seqs):
        q_b = q[b].float()  # [H, Lk]
        idx = kv_indices[start:start + L]
        k = k_buffer[idx].float()  # [L, KVH, Lk]
        v = v_buffer[idx].float()  # [L, KVH, Lv]
        k_e = k.repeat_interleave(group, dim=1)  # [L, H, Lk]
        v_e = v.repeat_interleave(group, dim=1)  # [L, H, Lv]
        scores = torch.einsum("hd,lhd->hl", q_b, k_e) * sm_scale  # [H, L]
        p = torch.softmax(scores, dim=-1)
        o_b = torch.einsum("hl,lhd->hd", p, v_e)  # [H, Lv]
        out[b] = o_b
        start += L
    return out


def _shape_of(cfg):
    return {"seqs": cfg["seqs"], "head": cfg["head"], "kv_head": cfg["kv_head"],
            "Lk": cfg["Lk"], "Lv": cfg["Lv"]}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "decode_attention_fwd"), \
            "Missing entry decode_attention_fwd"
        assert hasattr(mod, "_fwd_kernel_stage1"), \
            "Missing @triton.jit _fwd_kernel_stage1"
        assert hasattr(mod, "_fwd_grouped_kernel_stage1"), \
            "Missing @triton.jit _fwd_grouped_kernel_stage1"
        assert hasattr(mod, "_fwd_kernel_stage2"), \
            "Missing @triton.jit _fwd_kernel_stage2"
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
    # OR normalized worst-element max|ref-out|/max|ref| <= 1e-2. (The MHA normal
    # stage1 reduces q*k in bf16 elementwise rather than via MFMA, so a few
    # output elements exceed a raw 1e-2 while the 99.9% / normalized band holds.)
    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        sh = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            (q, k_buf, v_buf, o, kvp, kvi, al, alse, nks) = make_inputs(cfg, "cuda")
            sm_scale = 1.0 / (cfg["Lk"] ** 0.5)
            _retry_oom(lambda: mod.decode_attention_fwd(
                q, k_buf, v_buf, o, kvp, kvi, al, alse, nks, MAX_KV_SPLITS,
                sm_scale, 1.0, 1.0))
            torch.cuda.synchronize()
            ref = reference(q, k_buf, v_buf, kvi, cfg, sm_scale)
            finite = bool(torch.isfinite(o).all().item())
            diff = (o.float() - ref.float()).abs().max().item()
            denom = ref.float().abs().max().item()
            rel = diff / denom if denom > 0 else diff
            frac = torch.isclose(o.float(), ref.float(),
                                 atol=1e-2, rtol=1e-2).float().mean().item()
            passed = finite and (frac >= 0.999 or rel <= 1e-2)
            details.append({"shape_id": i + 1, "shape": sh, "max_diff": diff,
                            "rel": rel, "frac": frac, "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {sh}: max_diff={diff:.4e} "
                               f"rel={rel:.4e} frac={frac:.5f} "
                               f"finite={finite}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": sh, "error": str(e)})
            return False, f"Shape {i+1} {sh}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        params = _shape_of(cfg)
        try:
            torch.manual_seed(42 + ti)
            (q, k_buf, v_buf, o, kvp, kvi, al, alse, nks) = make_inputs(cfg, "cuda")
            sm_scale = 1.0 / (cfg["Lk"] ** 0.5)

            def fn():
                mod.decode_attention_fwd(
                    q, k_buf, v_buf, o, kvp, kvi, al, alse, nks, MAX_KV_SPLITS,
                    sm_scale, 1.0, 1.0)

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
                print(f"  shape {d['shape_id']} {d['shape']}: "
                      f"max_diff={d['max_diff']:.4e} rel={d['rel']:.4e} "
                      f"frac={d['frac']:.5f} -> {'PASS' if d['passed'] else 'FAIL'}")
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
