#!/usr/bin/env python3
"""Task runner for triton2triton/triton_count_expert_tokens"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_count_expert_tokens"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_count_expert_tokens.py")

# (num_tokens, topk, num_experts)
TEST_SHAPES = [
    (32, 2, 8),
    (64, 2, 16),
    (128, 4, 8),
    (256, 2, 32),
    (512, 2, 64),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_count(topk_ids, num_experts):
    import torch
    flat = topk_ids.flatten()
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    for e in range(num_experts):
        counts[e] = (flat == e).sum().item()
    return counts


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "count_expert_num_tokens"), "Missing count_expert_num_tokens"
        assert hasattr(mod, "_count_expert_num_tokens"), "Missing _count_expert_num_tokens"
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
    for i, (num_tokens, topk, num_experts) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)

            result = mod.count_expert_num_tokens(topk_ids, num_experts)
            torch.cuda.synchronize()

            ref = reference_count(topk_ids, num_experts)
            if not torch.equal(result, ref):
                diff = (result - ref).abs().max().item()
                return False, f"Shape {i+1}: max diff = {diff}"
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
    num_tokens, topk, num_experts = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)

    for _ in range(5):
        mod.count_expert_num_tokens(topk_ids, num_experts)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.count_expert_num_tokens(topk_ids, num_experts)
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
