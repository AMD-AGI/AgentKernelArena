#!/usr/bin/env python3
"""Task runner for triton2triton/triton_batched_moe"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_batched_moe"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_batched_moe.py")

# (E, max_tokens, K, N)
TEST_SHAPES = [
    (4, 16, 64, 64),
    (8, 32, 128, 128),
    (8, 64, 256, 256),
    (16, 64, 512, 512),
    (8, 128, 1024, 512),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "batched_moe_gemm"), "Missing batched_moe_gemm"
        assert hasattr(mod, "batched_triton_kernel"), "Missing batched_triton_kernel"
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
    for i, (E, max_tokens, K, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            A = torch.randn(E, max_tokens, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            expert_num_tokens = torch.randint(1, max_tokens + 1, (E,), device=device, dtype=torch.int32)

            result = mod.batched_moe_gemm(A, B, expert_num_tokens)
            torch.cuda.synchronize()

            # Reference: per-expert matmul
            ref = torch.zeros_like(result)
            for e in range(E):
                nt = expert_num_tokens[e].item()
                if nt > 0:
                    ref[e, :nt] = (A[e, :nt].float() @ B[e].float().T).to(torch.float16)

            if not torch.allclose(result.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (result.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1}: max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0
    device = "cuda"
    E, max_tokens, K, N = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    A = torch.randn(E, max_tokens, K, device=device, dtype=torch.float16) * 0.1
    B = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
    expert_num_tokens = torch.full((E,), max_tokens, device=device, dtype=torch.int32)

    for _ in range(5):
        mod.batched_moe_gemm(A, B, expert_num_tokens)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.batched_moe_gemm(A, B, expert_num_tokens)
        end_events[j].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


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
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
