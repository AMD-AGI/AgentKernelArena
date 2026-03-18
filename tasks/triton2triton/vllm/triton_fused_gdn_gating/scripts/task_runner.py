#!/usr/bin/env python3
"""Task runner for triton_fused_gdn_gating"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_fused_gdn_gating"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fused_gdn_gating.py")

# (batch, num_heads)
TEST_SHAPES = [
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(batch, num_heads, device="cpu"):
    import torch
    torch.manual_seed(42)
    A_log = torch.randn(num_heads, dtype=torch.float32) * 0.5
    a = torch.randn(batch, num_heads, dtype=torch.float16)
    b = torch.randn(batch, num_heads, dtype=torch.float16)
    dt_bias = torch.randn(num_heads, dtype=torch.float32) * 0.1

    if device != "cpu":
        A_log = A_log.to(device)
        a = a.to(device)
        b = b.to(device)
        dt_bias = dt_bias.to(device)
    return A_log, a, b, dt_bias


def reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    import torch
    import torch.nn.functional as F
    x = a.float() + dt_bias.float()
    softplus_x = F.softplus(x, beta=beta, threshold=threshold)
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float()).to(b.dtype)
    return g.unsqueeze(0), beta_output.unsqueeze(0)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fused_gdn_gating"), "Missing wrapper"
        assert hasattr(mod, "fused_gdn_gating_kernel"), "Missing kernel"
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
    for i, (batch, nh) in enumerate(TEST_SHAPES):
        try:
            inputs = make_inputs(batch, nh, device)
            res_g, res_beta = mod.fused_gdn_gating(*inputs)
            torch.cuda.synchronize()

            cpu_inputs = make_inputs(batch, nh, "cpu")
            ref_g, ref_beta = reference(*cpu_inputs)

            if not torch.allclose(res_g.cpu(), ref_g, atol=1e-3, rtol=1e-3):
                max_diff = (res_g.cpu() - ref_g).abs().max().item()
                return False, f"Shape {i+1}: g mismatch, max_diff={max_diff:.6f}"
            if not torch.allclose(res_beta.cpu().float(), ref_beta.float(), atol=1e-3, rtol=1e-3):
                max_diff = (res_beta.cpu().float() - ref_beta.float()).abs().max().item()
                return False, f"Shape {i+1}: beta mismatch, max_diff={max_diff:.6f}"
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

    for test_idx, (batch, nh) in enumerate(TEST_SHAPES):
        try:
            inputs = make_inputs(batch, nh, device)

            for _ in range(WARMUP_ITERATIONS):
                mod.fused_gdn_gating(*inputs)
            torch.cuda.synchronize()

            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                start_events[j].record()
                mod.fused_gdn_gating(*inputs)
                end_events[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": {
                    "batch": batch,
                    "num_heads": nh
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "num_heads": nh
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
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
