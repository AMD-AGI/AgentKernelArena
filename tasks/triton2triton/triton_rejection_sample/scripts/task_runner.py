#!/usr/bin/env python3
"""Task runner for triton_rejection_sample"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_rejection_sample"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_rejection_sample.py")

# (num_reqs, num_speculative_steps)
TEST_SHAPES = [
    (4, 3),
    (8, 5),
    (16, 4),
    (32, 6),
    (64, 5),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(num_reqs, num_spec_steps, device="cpu"):
    import torch
    torch.manual_seed(42)
    # Each request has (num_spec_steps + 1) logits/tokens
    tokens_per_req = num_spec_steps + 1
    total_tokens = num_reqs * tokens_per_req

    # Target sampled tokens
    target_sampled = torch.randint(0, 100, (total_tokens,), dtype=torch.int64)
    # Input IDs (draft tokens); make some match target to test acceptance
    input_ids = target_sampled.clone()
    # Introduce mismatches at random positions
    for r in range(num_reqs):
        base = r * tokens_per_req
        # Reject at a random position (or not at all)
        reject_pos = torch.randint(0, tokens_per_req + 1, (1,)).item()
        if reject_pos < tokens_per_req:
            for j in range(reject_pos, tokens_per_req):
                input_ids[base + j] = (target_sampled[base + j].item() + 1) % 100

    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32)
    for r in range(num_reqs):
        cu_num_logits[r + 1] = cu_num_logits[r] + tokens_per_req

    if device != "cpu":
        target_sampled = target_sampled.to(device)
        input_ids = input_ids.to(device)
        cu_num_logits = cu_num_logits.to(device)
    return target_sampled, input_ids, cu_num_logits


def reference(target_sampled, input_ids, cu_num_logits, num_spec_steps):
    import torch
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = torch.empty(num_reqs, num_spec_steps + 1, dtype=target_sampled.dtype)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32)

    for r in range(num_reqs):
        start = cu_num_logits[r].item()
        end = cu_num_logits[r + 1].item()
        nt = end - start
        ns = 0
        rejected = False
        for i in range(nt - 1):
            if not rejected:
                ts = target_sampled[start + i].item()
                ds = input_ids[start + i + 1].item()
                sampled[r, i] = ts
                ns += 1
                if ts != ds:
                    rejected = True
        if not rejected:
            ts = target_sampled[start + nt - 1].item()
            sampled[r, nt - 1] = ts
            ns += 1
        num_sampled[r] = ns
    return sampled, num_sampled


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "rejection_sample"), "Missing wrapper"
        assert hasattr(mod, "_rejection_sample_kernel"), "Missing kernel"
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
    for i, (nr, nss) in enumerate(TEST_SHAPES):
        try:
            ts, ids, cu = make_inputs(nr, nss, device)
            res_sampled, res_ns = mod.rejection_sample(ts, ids, cu, nss)
            torch.cuda.synchronize()

            ts_c, ids_c, cu_c = make_inputs(nr, nss, "cpu")
            ref_sampled, ref_ns = reference(ts_c, ids_c, cu_c, nss)

            if not torch.equal(res_ns.cpu(), ref_ns):
                return False, f"Shape {i+1}: num_sampled mismatch: {res_ns.cpu()} vs {ref_ns}"
            # Only check sampled values up to num_sampled for each request
            for r in range(nr):
                ns = ref_ns[r].item()
                if not torch.equal(res_sampled[r, :ns].cpu(), ref_sampled[r, :ns]):
                    return False, f"Shape {i+1}: sampled mismatch at req {r}"
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
    nr, nss = TEST_SHAPES[PERF_SHAPE_IDX]
    ts, ids, cu = make_inputs(nr, nss, device)

    for _ in range(5):
        mod.rejection_sample(ts, ids, cu, nss)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.rejection_sample(ts, ids, cu, nss)
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
