#!/usr/bin/env python3
"""Task runner for triton2triton/triton_ep_scatter_2"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_ep_scatter_2"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ep_scatter_2.py")

# (num_tokens, hidden_size, num_experts, topk)
TEST_SHAPES = [
    (16, 64, 4, 2),
    (32, 128, 8, 2),
    (64, 256, 8, 2),
    (128, 512, 16, 2),
    (256, 512, 8, 2),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def round_up_128(x):
    return ((x + 127) // 128) * 128


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "ep_scatter_2"), "Missing ep_scatter_2"
        assert hasattr(mod, "_fwd_kernel_ep_scatter_2"), "Missing _fwd_kernel_ep_scatter_2"
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
    for i, (num_tokens, hidden_size, num_experts, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            recv_x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)
            recv_topk = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)

            # Compute tokens per expert
            counts = torch.zeros(num_experts, dtype=torch.int32)
            for e in range(num_experts):
                counts[e] = (recv_topk.cpu() == e).sum().item()
            aligned = [round_up_128(c.item()) for c in counts]
            total = sum(aligned)

            starts = []
            s = 0
            for a in aligned:
                starts.append(s)
                s += a
            expert_start_loc = torch.tensor(starts, device=device, dtype=torch.int32)

            output_tensor = torch.zeros(total, hidden_size, device=device, dtype=torch.float16)
            output_index = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)

            mod.ep_scatter_2(recv_x, recv_topk, expert_start_loc, output_tensor, output_index)
            torch.cuda.synchronize()

            # Verify: each token's data was scattered to output_tensor at output_index positions
            for t in range(min(num_tokens, 16)):  # Check subset
                for k in range(topk):
                    eid = recv_topk[t, k].item()
                    if eid >= 0:
                        idx = output_index[t, k].item()
                        if idx < 0:
                            return False, f"Shape {i+1}: output_index not set for token {t} topk {k}"
                        if not torch.allclose(output_tensor[idx], recv_x[t], atol=1e-5):
                            return False, f"Shape {i+1}: data mismatch at token {t} topk {k}"
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
    num_tokens, hidden_size, num_experts, topk = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    recv_x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)
    recv_topk = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)

    counts = torch.zeros(num_experts, dtype=torch.int32)
    for e in range(num_experts):
        counts[e] = (recv_topk.cpu() == e).sum().item()
    aligned = [round_up_128(c.item()) for c in counts]
    total = sum(aligned)
    starts = []
    s = 0
    for a in aligned:
        starts.append(s)
        s += a

    for _ in range(5):
        expert_start_loc = torch.tensor(starts, device=device, dtype=torch.int32)
        output_tensor = torch.zeros(total, hidden_size, device=device, dtype=torch.float16)
        output_index = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
        mod.ep_scatter_2(recv_x, recv_topk, expert_start_loc, output_tensor, output_index)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        expert_start_loc_j = torch.tensor(starts, device=device, dtype=torch.int32)
        output_tensor_j = torch.zeros(total, hidden_size, device=device, dtype=torch.float16)
        output_index_j = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
        start_events[j].record()
        mod.ep_scatter_2(recv_x, recv_topk, expert_start_loc_j, output_tensor_j, output_index_j)
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
