#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fla_fused_recurrent"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_fla_fused_recurrent"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fla_fused_recurrent.py")

SEEDS = [42, 43, 44, 45, 46]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
PERF_SEED_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(q, k, v, g, beta, scale, initial_state=None):
    import torch
    B, T, H, K = q.shape
    V = v.shape[-1]
    q_c, k_c, v_c, g_c, b_c = q.float().cpu(), k.float().cpu(), v.float().cpu(), g.float().cpu(), beta.float().cpu()
    h0 = initial_state.float().cpu() if initial_state is not None else None
    o = torch.zeros(B, T, H, V)
    for b in range(B):
        for h in range(H):
            state = h0[b, h].clone() if h0 is not None else torch.zeros(V, K)
            for t in range(T):
                bq = q_c[b, t, h] * scale
                bk = k_c[b, t, h]
                bv = v_c[b, t, h]
                bg = g_c[b, t, h]
                bb = b_c[b, t, h]
                state = state * torch.exp(torch.tensor(bg))
                bv = bv - state @ bk
                bv = bv * bb
                state = state + bv[:, None] * bk[None, :]
                o[b, t, h] = state @ bq
    return o


def gen_inputs(seed, device):
    import torch
    torch.manual_seed(seed)
    B, T, H, K, V = 1, 32, 2, 16, 16
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32) * 0.1
    g = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid().log()
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, V, K, device=device, dtype=torch.float32) * 0.01
    return (q, k, v, g, beta, 0.25, h0), {}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fused_recurrent_gated_delta_rule_fwd"), "Missing fused_recurrent_gated_delta_rule_fwd"
        assert hasattr(mod, "fused_recurrent_gated_delta_rule_fwd_kernel"), "Missing fused_recurrent_gated_delta_rule_fwd_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    for i, seed in enumerate(SEEDS):
        try:
            args, kwargs = gen_inputs(seed, device)
            args_cpu = tuple(a.float().cpu() if isinstance(a, torch.Tensor) else a for a in args)

            result_tuple = mod.fused_recurrent_gated_delta_rule_fwd(*args, **kwargs)
            result = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
            ref = reference(*args_cpu, **kwargs)
            r_cpu = result.float().cpu()
            ref_f = ref.float()
            if not torch.allclose(r_cpu, ref_f, atol=5e-2, rtol=5e-2):
                max_diff = (r_cpu - ref_f).abs().max().item()
                return False, f"Shape {i+1}: max diff = {max_diff:.6f}"

            torch.cuda.synchronize()
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, seed in enumerate(SEEDS):
        try:
            args, kwargs = gen_inputs(seed, device)

            for _ in range(WARMUP_ITERATIONS):
                mod.fused_recurrent_gated_delta_rule_fwd(*args, **kwargs)
            torch.cuda.synchronize()

            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

            for j in range(n_iter):
                start_events[j].record()
                mod.fused_recurrent_gated_delta_rule_fwd(*args, **kwargs)
                end_events[j].record()

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": {
                    "seed": seed
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "seed": seed
                }
            })

    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args_parsed = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args_parsed.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(SEEDS)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
