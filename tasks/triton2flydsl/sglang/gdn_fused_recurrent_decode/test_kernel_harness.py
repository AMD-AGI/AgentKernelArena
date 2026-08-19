#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/gdn_fused_recurrent_decode.

Standalone harness for the GDN packed recurrent DECODE Triton kernel
(Qwen3.5-35B-A3B linear-attention decode hot kernel).

Modes:
  --compile        : ast-parse + import the standalone source, assert symbols.
  --correctness    : run the Triton kernel vs a torch fp32 reference on real
                     Qwen3.5-35B-A3B GDN decode shapes; assert close (output AND
                     updated recurrent state).
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

The Triton kernel `fused_recurrent_gated_delta_rule_packed_decode` is the kernel
under test; `reference_decode` (pure torch, fp32) is the golden. The flydsl
target, when it lands, is dropped in next to the source and compared the same way.

GPU may be shared; kernel launches retry with backoff on transient CUDA/HIP OOM.
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

TASK_NAME = "triton2flydsl/sglang/gdn_fused_recurrent_decode"
SOURCE_FILE = os.path.join(TASK_DIR, "gdn_fused_recurrent_decode.py")

# Test configs: (B, H, HV, K, V, pool_size) -- real Qwen3.5-35B-A3B GDN decode.
#   TP=2 serving => H=8,  HV=16, K=128, V=128 ; batch swept around CONC~16.
#   TP=1         => H=16, HV=32, K=128, V=128.
TEST_SHAPES = [
    (1, 8, 16, 128, 128, 32),
    (2, 8, 16, 128, 128, 32),
    (4, 8, 16, 128, 128, 32),
    (8, 8, 16, 128, 128, 64),
    (16, 8, 16, 128, 128, 64),     # ~ real serving concurrency
    (32, 8, 16, 128, 128, 128),
    (64, 8, 16, 128, 128, 128),
    (128, 8, 16, 128, 128, 256),
    (1, 16, 32, 128, 128, 32),
    (32, 16, 32, 128, 128, 128),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5

DTYPE_NAME = os.environ.get("GDN_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("gdn_decode_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err):
    msg = str(err).lower()
    return "out of memory" in msg


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


def make_test_data(B, H, HV, K, V, pool_size, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    qkv_dim = 2 * H * K + HV * V
    return {
        "B": B, "H": H, "HV": HV, "K": K, "V": V, "pool_size": pool_size,
        "mixed_qkv": torch.randn(B, qkv_dim, device=device, dtype=dtype),
        "a": torch.randn(B, HV, device=device, dtype=dtype),
        "b": torch.randn(B, HV, device=device, dtype=dtype),
        "A_log": torch.randn(HV, device=device, dtype=dtype),
        "dt_bias": torch.randn(HV, device=device, dtype=dtype),
        "ssm_states": torch.randn(pool_size, HV, V, K, device=device, dtype=dtype) * 0.1,
        "cache_indices": torch.arange(B, device=device, dtype=torch.int32),
        "scale": K ** -0.5,
    }


def reference_decode(inp):
    """Pure-torch fp32 golden, mirroring the Triton kernel exactly.

    Returns (out[B,1,HV,V], updated_state[pool,HV,V,K]) in the input dtype.
    """
    import torch
    B, H, HV, K, V = inp["B"], inp["H"], inp["HV"], inp["K"], inp["V"]
    scale = inp["scale"]
    dtype = inp["mixed_qkv"].dtype
    mq = inp["mixed_qkv"].float()
    a = inp["a"].float()
    b = inp["b"].float()
    A_log = inp["A_log"].float()
    dt_bias = inp["dt_bias"].float()
    idxs = inp["cache_indices"]
    state = inp["ssm_states"].clone()
    state_f = state.float()
    out = inp["mixed_qkv"].new_zeros(B, 1, HV, V)

    rep = HV // H
    for n in range(B):
        idx = int(idxs[n].item())
        if idx < 0:
            continue
        for hv in range(HV):
            i_h = hv // rep
            q = mq[n, i_h * K: i_h * K + K]
            k = mq[n, H * K + i_h * K: H * K + i_h * K + K]
            v = mq[n, 2 * H * K + hv * V: 2 * H * K + hv * V + V].clone()
            q = q / torch.sqrt((q * q).sum() + 1e-6)
            k = k / torch.sqrt((k * k).sum() + 1e-6)
            q = q * scale
            x = a[n, hv] + dt_bias[hv]
            softplus_x = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
            g = -torch.exp(A_log[hv]) * softplus_x
            beta = torch.sigmoid(b[n, hv])
            h = state_f[idx, hv].clone()                 # [V, K]
            h = h * torch.exp(g)
            v = v - (h * k[None, :]).sum(dim=1)          # [V]
            v = v * beta
            h = h + v[:, None] * k[None, :]
            o = (h * q[None, :]).sum(dim=1)              # [V]
            out[n, 0, hv] = o.to(dtype)
            state_f[idx, hv] = h
    return out, state_f.to(dtype)


def _run_triton(mod, inp):
    import torch
    B, HV, V = inp["B"], inp["HV"], inp["V"]
    state = inp["ssm_states"].clone()
    out = inp["mixed_qkv"].new_empty(B, 1, HV, V)
    _retry_oom(lambda: mod.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=inp["mixed_qkv"], a=inp["a"], b=inp["b"],
        A_log=inp["A_log"], dt_bias=inp["dt_bias"], scale=inp["scale"],
        initial_state=state, out=out, ssm_state_indices=inp["cache_indices"],
        use_qk_l2norm_in_kernel=True,
    ))
    torch.cuda.synchronize()
    return out, state


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "fused_recurrent_gated_delta_rule_packed_decode"), \
            "Missing entry fused_recurrent_gated_delta_rule_packed_decode"
        assert hasattr(mod, "fused_recurrent_gated_delta_rule_packed_decode_kernel"), \
            "Missing @triton.jit kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    dtype = getattr(torch, DTYPE_NAME)
    atol = 1e-4 if dtype == torch.float32 else 2e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    details = []

    for i, (B, H, HV, K, V, pool) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            inp = make_test_data(B, H, HV, K, V, pool, "cuda", dtype)
            out_t, state_t = _run_triton(mod, inp)
            out_r, state_r = reference_decode(inp)

            idxs = inp["cache_indices"]
            out_diff = (out_t.float() - out_r.float()).abs().max().item()
            st_diff = (state_t[idxs].float() - state_r[idxs].float()).abs().max().item()

            out_ok = torch.allclose(out_t.float(), out_r.float(), atol=atol, rtol=rtol)
            st_ok = torch.allclose(state_t[idxs].float(), state_r[idxs].float(), atol=atol, rtol=rtol)
            passed = bool(out_ok and st_ok)
            details.append({
                "shape_id": i + 1, "shape": [B, H, HV, K, V, pool],
                "out_max_diff": out_diff, "state_max_diff": st_diff,
                "passed": passed,
            })
            if not passed:
                return False, (f"Shape {i+1} {TEST_SHAPES[i]}: out_diff={out_diff:.4e} "
                               f"state_diff={st_diff:.4e}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": [B, H, HV, K, V, pool], "error": str(e)})
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details

    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    dtype = getattr(torch, DTYPE_NAME)
    test_cases = []
    for ti, (B, H, HV, K, V, pool) in enumerate(TEST_SHAPES):
        params = {"B": B, "H": H, "HV": HV, "K": K, "V": V, "pool_size": pool}
        try:
            torch.manual_seed(42 + ti)
            inp = make_test_data(B, H, HV, K, V, pool, "cuda", dtype)
            state = inp["ssm_states"]
            initial_state = state.clone()
            out = inp["mixed_qkv"].new_empty(B, 1, HV, V)

            def prepare_fn():
                state.copy_(initial_state)

            def fn():
                mod.fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=inp["mixed_qkv"], a=inp["a"], b=inp["b"],
                    A_log=inp["A_log"], dt_bias=inp["dt_bias"], scale=inp["scale"],
                    initial_state=state, out=out,
                    ssm_state_indices=inp["cache_indices"],
                    use_qk_l2norm_in_kernel=True,
                )

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
            test_cases.append({
                "test_case_id": f"perf{ti + 1}",
                "execution_time_ms": elapsed_ms,
                **bench_meta,
                "params": params,
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{ti + 1}",
                "execution_time_ms": -1.0,
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": "performance case failed before timing completed",
                "params": params,
            })
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
                print(f"  shape {d['shape_id']} {d['shape']}: out_diff={d['out_max_diff']:.4e} "
                      f"state_diff={d['state_max_diff']:.4e} -> {'PASS' if d['passed'] else 'FAIL'}")
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
