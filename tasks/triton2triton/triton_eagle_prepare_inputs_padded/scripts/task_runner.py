#!/usr/bin/env python3
"""Task runner for triton_eagle_prepare_inputs_padded"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_eagle_prepare_inputs_padded"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_eagle_prepare_inputs_padded.py")

# (num_reqs, max_draft_per_req, max_query_len_per_req)
TEST_SHAPES = [
    (4, 3, 10),
    (8, 5, 16),
    (16, 7, 32),
    (32, 4, 20),
    (64, 6, 24),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(num_reqs, max_draft, max_qlen, device="cpu"):
    import torch
    torch.manual_seed(42)
    # Per-request draft tokens (non-negative)
    draft_per_req = torch.randint(0, max_draft + 1, (num_reqs,), dtype=torch.int32)
    cu_num_draft_tokens = torch.cumsum(draft_per_req, dim=0).to(torch.int32)

    # valid_sampled count in [1, draft+1] (at least 1 valid if drafts > 0)
    valid_sampled = torch.zeros(num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        d = draft_per_req[i].item()
        if d > 0:
            valid_sampled[i] = torch.randint(1, d + 2, (1,)).item()
        else:
            valid_sampled[i] = 1  # no draft => 1 valid (the target token)

    # query_start_loc: cumulative query lengths, with padding for rejected tokens
    query_lens = torch.randint(1, max_qlen + 1, (num_reqs,), dtype=torch.int32)
    # Make sure query_len >= draft + 1 for meaningful test
    for i in range(num_reqs):
        d = draft_per_req[i].item()
        query_lens[i] = max(query_lens[i].item(), d + 2)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    for i in range(num_reqs):
        query_start_loc[i + 1] = query_start_loc[i] + query_lens[i]

    if device != "cpu":
        cu_num_draft_tokens = cu_num_draft_tokens.to(device)
        valid_sampled = valid_sampled.to(device)
        query_start_loc = query_start_loc.to(device)

    return cu_num_draft_tokens, valid_sampled, query_start_loc, draft_per_req


def reference(cu_num_draft_tokens, valid_sampled, query_start_loc):
    import torch
    num_reqs = cu_num_draft_tokens.shape[0]
    token_indices = torch.empty(num_reqs, dtype=torch.int32)
    num_rejected = torch.empty(num_reqs, dtype=torch.int32)

    for i in range(num_reqs):
        cu_curr = cu_num_draft_tokens[i].item()
        if i == 0:
            ndraft = cu_curr
        else:
            ndraft = cu_curr - cu_num_draft_tokens[i - 1].item()
        vc = valid_sampled[i].item()
        nrej = ndraft + 1 - vc
        if ndraft <= 0:
            nrej = 0
        q_last = query_start_loc[i + 1].item() - 1
        token_indices[i] = q_last - nrej
        num_rejected[i] = nrej

    return token_indices, num_rejected


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "eagle_prepare_inputs_padded"), "Missing eagle_prepare_inputs_padded"
        assert hasattr(mod, "eagle_prepare_inputs_padded_kernel"), "Missing kernel"
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
    for i, (nr, md, mq) in enumerate(TEST_SHAPES):
        try:
            cu, vs, qsl, _ = make_inputs(nr, md, mq, device)
            res_idx, res_rej = mod.eagle_prepare_inputs_padded(cu, vs, qsl)
            torch.cuda.synchronize()

            ref_idx, ref_rej = reference(cu.cpu(), vs.cpu(), qsl.cpu())
            if not torch.equal(res_idx.cpu(), ref_idx):
                return False, f"Shape {i+1}: token_indices mismatch"
            if not torch.equal(res_rej.cpu(), ref_rej):
                return False, f"Shape {i+1}: num_rejected mismatch"
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
    nr, md, mq = TEST_SHAPES[PERF_SHAPE_IDX]
    cu, vs, qsl, _ = make_inputs(nr, md, mq, device)

    for _ in range(5):
        mod.eagle_prepare_inputs_padded(cu, vs, qsl)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.eagle_prepare_inputs_padded(cu, vs, qsl)
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
