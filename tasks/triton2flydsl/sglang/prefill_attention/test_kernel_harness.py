#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/prefill_attention.

Standalone harness for sglang's memory-efficient prefill attention Triton kernel
(`context_attention_fwd` -> `_fwd_kernel`): varlen (ragged-batch) flash attention
over packed q/k/v [total_tokens, head, head_dim] with per-batch offsets, optional
causal masking, and grouped-query attention.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton prefill attn vs torch fp32 masked-softmax SDPA, close.
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

TASK_NAME = "triton2flydsl/sglang/prefill_attention"
SOURCE_FILE = os.path.join(TASK_DIR, "prefill_attention.py")

# Varlen prefill: list of per-batch sequence lengths, q heads, kv heads, head_dim,
# causal flag. GQA group = head // kv_head.
TEST_SHAPES = [
    {"seqs": [128, 128], "head": 32, "kv_head": 8, "d": 128, "causal": True},
    {"seqs": [256], "head": 32, "kv_head": 32, "d": 128, "causal": True},  # MHA
    {"seqs": [64, 200, 37], "head": 28, "kv_head": 4, "d": 128, "causal": True},  # ragged
    {"seqs": [128, 128], "head": 16, "kv_head": 16, "d": 64, "causal": True},
    {"seqs": [192], "head": 32, "kv_head": 8, "d": 128, "causal": False},  # bidirectional
    {"seqs": [100, 100], "head": 16, "kv_head": 2, "d": 128, "causal": True},
    {"seqs": [333], "head": 8, "kv_head": 1, "d": 128, "causal": True},  # MQA, T%block!=0
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("prefill_attention_src", SOURCE_FILE)
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
    total = sum(seqs)
    head, kv_head, d = cfg["head"], cfg["kv_head"], cfg["d"]
    q = torch.randn(total, head, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, kv_head, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, kv_head, d, device=device, dtype=torch.bfloat16)
    o = torch.empty(total, head, d, device=device, dtype=torch.bfloat16)
    b_seq_len = torch.tensor(seqs, device=device, dtype=torch.int32)
    b_start_loc = torch.zeros(len(seqs), device=device, dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
    return q, k, v, o, b_start_loc, b_seq_len, max(seqs)


def reference(q, k, v, cfg, sm_scale=None):
    import torch
    seqs = cfg["seqs"]
    head, kv_head, d = cfg["head"], cfg["kv_head"], cfg["d"]
    group = head // kv_head
    if sm_scale is None:
        sm_scale = 1.0 / (d ** 0.5)
    out = torch.empty_like(q, dtype=torch.float32)
    start = 0
    for L in seqs:
        qb = q[start:start + L].float()  # [L, head, d]
        kb = k[start:start + L].float()  # [L, kv, d]
        vb = v[start:start + L].float()
        kb_e = kb.repeat_interleave(group, dim=1)  # [L, head, d]
        vb_e = vb.repeat_interleave(group, dim=1)
        scores = torch.einsum("lhd,mhd->hlm", qb, kb_e) * sm_scale  # [head, L, L]
        if cfg["causal"]:
            cm = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(cm[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        ob = torch.einsum("hlm,mhd->lhd", p, vb_e)  # [L, head, d]
        out[start:start + L] = ob
        start += L
    return out


def _shape_of(cfg):
    return {"seqs": cfg["seqs"], "head": cfg["head"], "kv_head": cfg["kv_head"],
            "d": cfg["d"], "causal": cfg["causal"]}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "context_attention_fwd"), \
            "Missing entry context_attention_fwd"
        assert hasattr(mod, "_fwd_kernel"), "Missing @triton.jit _fwd_kernel"
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
    # OR normalized worst-element max|ref-out|/max|ref| <= 1e-2. Numerically stable
    # softmax, fp32 (MFMA) accumulation in both kernel and reference.
    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        sh = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            q, k, v, o, bsl, bseq, max_len = make_inputs(cfg, "cuda")
            _retry_oom(lambda: mod.context_attention_fwd(
                q, k, v, o, bsl, bseq, max_len, is_causal=cfg["causal"]))
            torch.cuda.synchronize()
            ref = reference(q, k, v, cfg)
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
            q, k, v, o, bsl, bseq, max_len = make_inputs(cfg, "cuda")

            def fn():
                mod.context_attention_fwd(
                    q, k, v, o, bsl, bseq, max_len, is_causal=cfg["causal"])

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
