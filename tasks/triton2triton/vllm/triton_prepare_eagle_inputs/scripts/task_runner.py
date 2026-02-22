#!/usr/bin/env python3
"""Task runner for triton_prepare_eagle_inputs"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_prepare_eagle_inputs"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_eagle_inputs.py")

# (num_reqs, tokens_per_req, num_rejected_max)
TEST_SHAPES = [
    (4, 16, 2),
    (8, 32, 3),
    (16, 24, 2),
    (32, 20, 4),
    (64, 32, 3),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(num_reqs, tokens_per_req, num_rej_max, device="cpu"):
    import torch
    torch.manual_seed(42)
    total_tokens = num_reqs * tokens_per_req
    target_input_ids = torch.randint(0, 32000, (total_tokens,), dtype=torch.int32)
    target_positions = torch.zeros(total_tokens, dtype=torch.int32)
    for r in range(num_reqs):
        start = r * tokens_per_req
        for t in range(tokens_per_req):
            target_positions[start + t] = t

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    max_num_reqs = num_reqs + 8
    last_sampled = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int64)
    next_prefill_tokens = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int32)
    num_sampled = torch.ones(num_reqs, dtype=torch.int32)  # all have sampled tokens
    num_rejected = torch.randint(0, min(num_rej_max + 1, tokens_per_req - 1), (num_reqs,), dtype=torch.int32)

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + tokens_per_req

    if device != "cpu":
        target_input_ids = target_input_ids.to(device)
        target_positions = target_positions.to(device)
        idx_mapping = idx_mapping.to(device)
        last_sampled = last_sampled.to(device)
        next_prefill_tokens = next_prefill_tokens.to(device)
        num_sampled = num_sampled.to(device)
        num_rejected = num_rejected.to(device)
        query_start_loc = query_start_loc.to(device)
    return (target_input_ids, target_positions, idx_mapping, last_sampled,
            next_prefill_tokens, num_sampled, num_rejected, query_start_loc)


def reference(target_input_ids, target_positions, idx_mapping, last_sampled,
              next_prefill_tokens, num_sampled, num_rejected, query_start_loc):
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = target_input_ids.shape[0]
    eagle_input_ids = torch.zeros_like(target_input_ids)
    eagle_positions = torch.zeros_like(target_positions)
    last_token_indices = torch.empty(num_reqs, dtype=torch.int64)

    for b in range(num_reqs):
        req_idx = idx_mapping[b].item()
        qs = query_start_loc[b].item()
        qe = query_start_loc[b + 1].item()
        ql = qe - qs
        nrej = num_rejected[b].item()
        ql -= nrej

        ns = num_sampled[b].item()
        if ns > 0:
            nt = int(last_sampled[req_idx].item())
        else:
            nt = next_prefill_tokens[req_idx].item()

        # Shift input ids
        for j in range(1, ql):
            eagle_input_ids[qs + j - 1] = target_input_ids[qs + j]
        lti = qs + ql - 1
        last_token_indices[b] = lti
        eagle_input_ids[lti] = nt

        # Copy positions
        for j in range(ql):
            eagle_positions[qs + j] = target_positions[qs + j]

    return last_token_indices, eagle_input_ids, eagle_positions


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_eagle_inputs"), "Missing wrapper"
        assert hasattr(mod, "_prepare_eagle_inputs_kernel"), "Missing kernel"
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
    for i, (nr, tpr, nrm) in enumerate(TEST_SHAPES):
        try:
            inputs = make_inputs(nr, tpr, nrm, device)
            res_lti, res_eids, res_epos = mod.prepare_eagle_inputs(*inputs)
            torch.cuda.synchronize()

            cpu_inputs = make_inputs(nr, tpr, nrm, "cpu")
            ref_lti, ref_eids, ref_epos = reference(*cpu_inputs)

            if not torch.equal(res_lti.cpu(), ref_lti):
                return False, f"Shape {i+1}: last_token_indices mismatch"
            if not torch.equal(res_eids.cpu(), ref_eids):
                return False, f"Shape {i+1}: eagle_input_ids mismatch"
            if not torch.equal(res_epos.cpu(), ref_epos):
                return False, f"Shape {i+1}: eagle_positions mismatch"
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
    nr, tpr, nrm = TEST_SHAPES[PERF_SHAPE_IDX]
    inputs = make_inputs(nr, tpr, nrm, device)

    for _ in range(5):
        mod.prepare_eagle_inputs(*inputs)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.prepare_eagle_inputs(*inputs)
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
