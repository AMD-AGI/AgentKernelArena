#!/usr/bin/env python3
"""Task runner for triton2triton/triton_ep_scatter_1"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_ep_scatter_1"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ep_scatter_1.py")

# (num_experts, max_tokens_per_expert)
TEST_SHAPES = [
    (4, 32),
    (8, 64),
    (16, 128),
    (32, 64),
    (64, 128),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def round_up_128(x):
    return ((x + 127) // 128) * 128


def reference_scatter_1(tokens_per_expert):
    import torch
    num_experts = len(tokens_per_expert)
    aligned = [round_up_128(t.item()) for t in tokens_per_expert]
    starts = []
    s = 0
    for a in aligned:
        starts.append(s)
        s += a
    total = s
    expert_start_loc = torch.tensor(starts, dtype=torch.int32)
    m_indices = torch.full((total,), -1, dtype=torch.int32)
    for e in range(num_experts):
        nt = tokens_per_expert[e].item()
        st = starts[e]
        m_indices[st:st + nt] = e
    return expert_start_loc, m_indices


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "ep_scatter_1"), "Missing ep_scatter_1"
        assert hasattr(mod, "_fwd_kernel_ep_scatter_1"), "Missing _fwd_kernel_ep_scatter_1"
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
    for i, (num_experts, max_tpe) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            tokens_per_expert = torch.randint(0, max_tpe + 1, (num_experts,), device=device, dtype=torch.int32)

            # Compute total aligned size
            aligned_counts = [round_up_128(t.item()) for t in tokens_per_expert]
            total = sum(aligned_counts)

            expert_start_loc = torch.empty(num_experts, device=device, dtype=torch.int32)
            m_indices = torch.full((total,), -1, device=device, dtype=torch.int32)

            mod.ep_scatter_1(tokens_per_expert, expert_start_loc, m_indices)
            torch.cuda.synchronize()

            ref_starts, ref_m_indices = reference_scatter_1(tokens_per_expert.cpu())

            if not torch.equal(expert_start_loc.cpu(), ref_starts):
                return False, f"Shape {i+1}: expert_start_loc mismatch"
            if not torch.equal(m_indices.cpu(), ref_m_indices):
                return False, f"Shape {i+1}: m_indices mismatch"
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
    num_experts, max_tpe = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    tokens_per_expert = torch.randint(1, max_tpe + 1, (num_experts,), device=device, dtype=torch.int32)
    aligned_counts = [round_up_128(t.item()) for t in tokens_per_expert]
    total = sum(aligned_counts)
    expert_start_loc = torch.empty(num_experts, device=device, dtype=torch.int32)
    m_indices = torch.full((total,), -1, device=device, dtype=torch.int32)

    for _ in range(5):
        mod.ep_scatter_1(tokens_per_expert, expert_start_loc, m_indices)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        m_indices.fill_(-1)
        start_events[j].record()
        mod.ep_scatter_1(tokens_per_expert, expert_start_loc, m_indices)
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
