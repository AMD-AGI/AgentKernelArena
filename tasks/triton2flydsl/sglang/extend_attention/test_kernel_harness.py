#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/extend_attention.

Standalone harness for sglang's extend (prefill-with-KV-cache) attention Triton
kernel (`extend_attention_fwd` -> `_fwd_kernel`): new query tokens attend to a
cached prefix (gathered from k_buffer/v_buffer via kv_indices, full visibility)
and to the freshly-appended extend KV (causal), online-softmax in fp32, GQA.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton extend attn vs torch fp32 masked-softmax SDPA, close.
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

TASK_NAME = "triton2flydsl/sglang/extend_attention"
SOURCE_FILE = os.path.join(TASK_DIR, "extend_attention.py")

# Per-batch (prefix_len, extend_len) plus head config. Lk == Lq (q/k share head
# dim); Lv may differ (MLA: Lq=Lk=192, Lv=128 exercises the BLOCK_DPE rope-PE path).
TEST_SHAPES = [
    {"prefix": [64, 128], "extend": [32, 32], "head": 32, "kv_head": 8,
     "Lq": 128, "Lv": 128, "causal": True},
    {"prefix": [200], "extend": [50], "head": 16, "kv_head": 16,
     "Lq": 128, "Lv": 128, "causal": True},  # MHA
    {"prefix": [0, 64, 100], "extend": [16, 16, 24], "head": 28, "kv_head": 4,
     "Lq": 128, "Lv": 128, "causal": True},  # ragged, one pure-prefill (prefix=0)
    {"prefix": [128, 128], "extend": [40, 40], "head": 16, "kv_head": 2,
     "Lq": 64, "Lv": 64, "causal": True},  # GQA, d=64
    {"prefix": [128], "extend": [32], "head": 16, "kv_head": 16,
     "Lq": 192, "Lv": 128, "causal": True},  # MLA-style split head (BLOCK_DPE)
    {"prefix": [96], "extend": [48], "head": 32, "kv_head": 8,
     "Lq": 128, "Lv": 128, "causal": False},  # bidirectional extend
    {"prefix": [150], "extend": [70], "head": 8, "kv_head": 1,
     "Lq": 128, "Lv": 128, "causal": True},  # MQA
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("extend_attention_src", SOURCE_FILE)
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
    prefix, extend = cfg["prefix"], cfg["extend"]
    H, KVH, Lq, Lv = cfg["head"], cfg["kv_head"], cfg["Lq"], cfg["Lv"]
    ext_total = sum(extend)
    pre_total = sum(prefix)
    dt = torch.bfloat16
    q_extend = torch.randn(ext_total, H, Lq, device=device, dtype=dt)
    k_extend = torch.randn(ext_total, KVH, Lq, device=device, dtype=dt)
    v_extend = torch.randn(ext_total, KVH, Lv, device=device, dtype=dt)
    o_extend = torch.empty(ext_total, H, Lv, device=device, dtype=dt)
    k_buffer = torch.randn(max(pre_total, 1), KVH, Lq, device=device, dtype=dt)
    v_buffer = torch.randn(max(pre_total, 1), KVH, Lv, device=device, dtype=dt)
    qo_indptr = torch.zeros(len(extend) + 1, device=device, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(
        torch.tensor(extend, device=device, dtype=torch.int32), 0)
    kv_indptr = torch.zeros(len(prefix) + 1, device=device, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(prefix, device=device, dtype=torch.int32), 0)
    kv_indices = torch.arange(pre_total, device=device, dtype=torch.int64)
    max_len_extend = max(extend)
    return (q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices, max_len_extend)


def reference(q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indices, cfg,
              sm_scale=None):
    import torch
    prefix, extend = cfg["prefix"], cfg["extend"]
    H, KVH, Lq, Lv = cfg["head"], cfg["kv_head"], cfg["Lq"], cfg["Lv"]
    group = H // KVH
    if sm_scale is None:
        sm_scale = 1.0 / (Lq ** 0.5)
    out = torch.empty(sum(extend), H, Lv, device=q_extend.device, dtype=torch.float32)
    qo = 0
    kvo = 0
    for i, (Lp, Le) in enumerate(zip(prefix, extend)):
        q_i = q_extend[qo:qo + Le].float()  # [Le, H, Lq]
        ke_i = k_extend[qo:qo + Le].float()  # [Le, KVH, Lq]
        ve_i = v_extend[qo:qo + Le].float()  # [Le, KVH, Lv]
        if Lp > 0:
            idx = kv_indices[kvo:kvo + Lp]
            kp = k_buffer[idx].float()  # [Lp, KVH, Lq]
            vp = v_buffer[idx].float()  # [Lp, KVH, Lv]
            K = torch.cat([kp, ke_i], dim=0)  # [Lp+Le, KVH, Lq]
            V = torch.cat([vp, ve_i], dim=0)
        else:
            K = ke_i
            V = ve_i
        Lt = Lp + Le
        K_e = K.repeat_interleave(group, dim=1)  # [Lt, H, Lq]
        V_e = V.repeat_interleave(group, dim=1)  # [Lt, H, Lv]
        scores = torch.einsum("lhd,mhd->hlm", q_i, K_e) * sm_scale  # [H, Le, Lt]
        col = torch.arange(Lt, device=q_extend.device)
        row = torch.arange(Le, device=q_extend.device)
        if cfg["causal"]:
            # prefix (col < Lp) always visible; extend region causal q >= (col-Lp)
            ext_col = col - Lp
            visible = torch.where(
                col[None, :] < Lp,
                torch.ones(Le, Lt, dtype=torch.bool, device=q_extend.device),
                row[:, None] >= ext_col[None, :],
            )
            scores = scores.masked_fill(~visible[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        ob = torch.einsum("hlm,mhd->lhd", p, V_e)  # [Le, H, Lv]
        out[qo:qo + Le] = ob
        qo += Le
        kvo += Lp
    return out


def _shape_of(cfg):
    return {"prefix": cfg["prefix"], "extend": cfg["extend"], "head": cfg["head"],
            "kv_head": cfg["kv_head"], "Lq": cfg["Lq"], "Lv": cfg["Lv"],
            "causal": cfg["causal"]}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "extend_attention_fwd"), \
            "Missing entry extend_attention_fwd"
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
    # OR normalized worst-element max|ref-out|/max|ref| <= 1e-2. fp32 (MFMA)
    # accumulation in both kernel and reference.
    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        sh = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            (q_e, k_e, v_e, o_e, k_buf, v_buf, qo, kvp, kvi, mle) = make_inputs(cfg, "cuda")
            _retry_oom(lambda: mod.extend_attention_fwd(
                q_e, k_e, v_e, o_e, k_buf, v_buf, qo, kvp, kvi,
                None, cfg["causal"], None, mle, 1.0, 1.0))
            torch.cuda.synchronize()
            ref = reference(q_e, k_e, v_e, k_buf, v_buf, kvi, cfg)
            finite = bool(torch.isfinite(o_e).all().item())
            diff = (o_e.float() - ref.float()).abs().max().item()
            denom = ref.float().abs().max().item()
            rel = diff / denom if denom > 0 else diff
            frac = torch.isclose(o_e.float(), ref.float(),
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
            (q_e, k_e, v_e, o_e, k_buf, v_buf, qo, kvp, kvi, mle) = make_inputs(cfg, "cuda")

            def fn():
                mod.extend_attention_fwd(
                    q_e, k_e, v_e, o_e, k_buf, v_buf, qo, kvp, kvi,
                    None, cfg["causal"], None, mle, 1.0, 1.0)

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
