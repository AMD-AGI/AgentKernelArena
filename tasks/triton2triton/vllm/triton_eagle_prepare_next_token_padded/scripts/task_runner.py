#!/usr/bin/env python3
"""Task runner for triton_eagle_prepare_next_token_padded"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_eagle_prepare_next_token_padded"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_eagle_prepare_next_token_padded.py")

# (num_reqs, num_sampled_tokens_per_req, vocab_size)
TEST_SHAPES = [
    (4, 4, 32000),
    (8, 6, 32000),
    (16, 8, 50000),
    (32, 5, 32000),
    (64, 7, 128000),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(num_reqs, num_sampled, vocab_size, device="cpu"):
    import torch
    torch.manual_seed(42)
    # Sampled token IDs: mix of valid tokens and -1 (rejected)
    sampled = torch.randint(0, vocab_size, (num_reqs, num_sampled), dtype=torch.int32)
    # Randomly mark some tokens as rejected (-1)
    reject_mask = torch.rand(num_reqs, num_sampled) < 0.3
    sampled[reject_mask] = -1
    # Some requests fully rejected
    if num_reqs > 2:
        sampled[0] = -1  # all rejected

    discard_mask = torch.zeros(num_reqs, dtype=torch.bool)
    if num_reqs > 3:
        discard_mask[1] = True

    backup = torch.randint(0, vocab_size, (num_reqs,), dtype=torch.int32)

    if device != "cpu":
        sampled = sampled.to(device)
        discard_mask = discard_mask.to(device)
        backup = backup.to(device)
    return sampled, discard_mask, backup


def reference(sampled, discard_mask, backup, vocab_size):
    import torch
    num_reqs, num_sampled = sampled.shape
    next_tokens = torch.empty(num_reqs, dtype=torch.int32)
    valid_counts = torch.empty(num_reqs, dtype=torch.int32)

    for i in range(num_reqs):
        if discard_mask[i]:
            next_tokens[i] = backup[i]
            valid_counts[i] = 0
        else:
            valid_indices = []
            for j in range(num_sampled):
                tid = sampled[i, j].item()
                if tid != -1 and tid < vocab_size:
                    valid_indices.append(j)
            vc = len(valid_indices)
            valid_counts[i] = vc
            if vc > 0:
                last_idx = max(valid_indices)
                next_tokens[i] = sampled[i, last_idx].item()
            else:
                next_tokens[i] = backup[i]
    return next_tokens, valid_counts


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "eagle_prepare_next_token_padded"), "Missing wrapper"
        assert hasattr(mod, "eagle_prepare_next_token_padded_kernel"), "Missing kernel"
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
    for i, (nr, ns, vs) in enumerate(TEST_SHAPES):
        try:
            sampled, dm, backup = make_inputs(nr, ns, vs, device)
            res_next, res_vc = mod.eagle_prepare_next_token_padded(sampled, dm, backup, vs)
            torch.cuda.synchronize()

            ref_next, ref_vc = reference(sampled.cpu(), dm.cpu(), backup.cpu(), vs)
            if not torch.equal(res_next.cpu(), ref_next):
                return False, f"Shape {i+1}: next_token_ids mismatch"
            if not torch.equal(res_vc.cpu(), ref_vc):
                return False, f"Shape {i+1}: valid_count mismatch"
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
    nr, ns, vs = TEST_SHAPES[PERF_SHAPE_IDX]
    sampled, dm, backup = make_inputs(nr, ns, vs, device)

    for _ in range(5):
        mod.eagle_prepare_next_token_padded(sampled, dm, backup, vs)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.eagle_prepare_next_token_padded(sampled, dm, backup, vs)
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
