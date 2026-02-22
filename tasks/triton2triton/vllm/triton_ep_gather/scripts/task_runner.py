#!/usr/bin/env python3
"""Task runner for triton2triton/triton_ep_gather"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_ep_gather"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ep_gather.py")

# (num_tokens, hidden_size, total_slots, topk)
TEST_SHAPES = [
    (16, 128, 64, 2),
    (32, 256, 128, 2),
    (64, 512, 256, 2),
    (128, 1024, 512, 2),
    (256, 1024, 1024, 2),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_gather(input_tensor, topk_ids, topk_weight, input_index, num_tokens, hidden_size):
    import torch
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    topk = topk_ids.shape[1]
    for t in range(num_tokens):
        for k in range(topk):
            eid = topk_ids[t, k].item()
            if eid >= 0:
                idx = input_index[t, k].item()
                w = topk_weight[t, k].item()
                output[t] += input_tensor[idx].float() * w
    return output


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "ep_gather"), "Missing ep_gather"
        assert hasattr(mod, "_fwd_kernel_ep_gather"), "Missing _fwd_kernel_ep_gather"
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
    for i, (num_tokens, hidden_size, total_slots, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            input_tensor = torch.randn(total_slots, hidden_size, device=device, dtype=torch.float16)
            topk_ids = torch.randint(0, 8, (num_tokens, topk), device=device, dtype=torch.int32)
            topk_weight = torch.randn(num_tokens, topk, device=device, dtype=torch.float32).abs()
            input_index = torch.randint(0, total_slots, (num_tokens, topk), device=device, dtype=torch.int32)
            output_tensor = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float16)

            mod.ep_gather(input_tensor, topk_ids, topk_weight, input_index, output_tensor)
            torch.cuda.synchronize()

            ref = reference_gather(input_tensor.cpu(), topk_ids.cpu(), topk_weight.cpu(), input_index.cpu(),
                                   num_tokens, hidden_size).to(torch.float16).to(device)
            if not torch.allclose(output_tensor.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (output_tensor.float() - ref.float()).abs().max().item()
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
    num_tokens, hidden_size, total_slots, topk = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    input_tensor = torch.randn(total_slots, hidden_size, device=device, dtype=torch.float16)
    topk_ids = torch.randint(0, 8, (num_tokens, topk), device=device, dtype=torch.int32)
    topk_weight = torch.randn(num_tokens, topk, device=device, dtype=torch.float32).abs()
    input_index = torch.randint(0, total_slots, (num_tokens, topk), device=device, dtype=torch.int32)
    output_tensor = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float16)

    for _ in range(5):
        mod.ep_gather(input_tensor, topk_ids, topk_weight, input_index, output_tensor)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.ep_gather(input_tensor, topk_ids, topk_weight, input_index, output_tensor)
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
