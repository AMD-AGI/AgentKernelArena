#!/usr/bin/env python3
"""Task runner for triton_copy_and_expand_eagle_inputs"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_copy_and_expand_eagle_inputs"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_copy_and_expand_eagle_inputs.py")

# (num_reqs, tokens_per_req, num_padding_slots)
TEST_SHAPES = [
    (2, 8, 1),
    (4, 10, 2),
    (8, 16, 1),
    (16, 12, 3),
    (32, 20, 2),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(num_reqs, tokens_per_req, num_pad_slots, device="cpu"):
    import torch
    torch.manual_seed(42)
    # Each request has tokens_per_req tokens, last few may be "rejected"
    total_tokens = num_reqs * tokens_per_req
    target_token_ids = torch.randint(0, 32000, (total_tokens,), dtype=torch.int32)
    target_positions = torch.zeros(total_tokens, dtype=torch.int32)
    for r in range(num_reqs):
        start = r * tokens_per_req
        for t in range(tokens_per_req):
            target_positions[start + t] = r * 100 + t

    next_token_ids = torch.randint(0, 32000, (num_reqs,), dtype=torch.int32)

    # query_start_loc and query_end_loc
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_end_loc = torch.zeros(num_reqs, dtype=torch.int32)
    for r in range(num_reqs):
        query_start_loc[r] = r * tokens_per_req
        # query_end_loc is a few tokens before the end (simulating some accepted, some rejected)
        num_rejected = min(2, tokens_per_req - 1)
        query_end_loc[r] = r * tokens_per_req + tokens_per_req - 1 - num_rejected
    query_start_loc[num_reqs] = total_tokens

    if device != "cpu":
        target_token_ids = target_token_ids.to(device)
        target_positions = target_positions.to(device)
        next_token_ids = next_token_ids.to(device)
        query_start_loc = query_start_loc.to(device)
        query_end_loc = query_end_loc.to(device)

    return target_token_ids, target_positions, next_token_ids, query_start_loc, query_end_loc


def reference_copy_and_expand(
    target_token_ids,
    target_positions,
    next_token_ids,
    query_start_loc,
    query_end_loc,
    padding_token_id,
    parallel_drafting_token_id,
    num_padding_slots_per_request,
    shift_input_ids,
):
    import torch

    num_reqs = next_token_ids.shape[0]
    total_input = target_token_ids.shape[0]
    total_out = total_input + num_reqs * (num_padding_slots_per_request + 10)
    device = target_token_ids.device

    out_input_ids = torch.zeros(total_out, dtype=torch.int32, device=device)
    out_positions = torch.zeros(total_out, dtype=torch.int32, device=device)
    out_is_rejected = torch.zeros(total_out, dtype=torch.bool, device=device)
    out_is_masked = torch.zeros(total_out, dtype=torch.bool, device=device)
    out_new_token_indices = torch.zeros(
        num_padding_slots_per_request * num_reqs, dtype=torch.int32, device=device
    )
    out_hidden_state_mapping = torch.zeros(total_input, dtype=torch.int32, device=device)

    for req in range(num_reqs):
        q_start = query_start_loc[req].item()
        q_next = query_start_loc[req + 1].item()
        q_end = query_end_loc[req].item()

        if shift_input_ids:
            num_valid = q_end - q_start
            input_offset = 1
            out_start = q_start + req * (num_padding_slots_per_request - 1)
        else:
            num_valid = q_end - q_start + 1
            input_offset = 0
            out_start = q_start + req * num_padding_slots_per_request

        num_rejected = q_next - q_end - 1
        total_output = num_valid + num_padding_slots_per_request + num_rejected
        start_pos = target_positions[q_start].item()

        for j in range(total_output):
            out_idx = out_start + j
            if j < num_valid:
                in_idx = q_start + input_offset + j
                token = target_token_ids[in_idx]
                pos = start_pos + j
                is_rej = False
                is_msk = False
            elif j == num_valid:
                token = next_token_ids[req]
                pos = start_pos + j
                is_rej = False
                is_msk = False
            elif j < num_valid + num_padding_slots_per_request:
                token = torch.tensor(parallel_drafting_token_id, dtype=torch.int32, device=device)
                pos = start_pos + j
                is_rej = False
                is_msk = True
            else:
                token = torch.tensor(padding_token_id, dtype=torch.int32, device=device)
                pos = 0
                is_rej = True
                is_msk = False

            out_input_ids[out_idx] = token
            out_positions[out_idx] = pos
            out_is_rejected[out_idx] = is_rej
            out_is_masked[out_idx] = is_msk

            if num_valid <= j < (num_valid + num_padding_slots_per_request):
                local_new_idx = j - num_valid
                out_new_token_indices[req * num_padding_slots_per_request + local_new_idx] = out_idx

        if shift_input_ids:
            num_input_this_req = q_next - q_start
            for j in range(num_input_this_req):
                src_idx = q_start + j
                out_hidden_state_mapping[src_idx] = out_start + j

    return (
        out_input_ids,
        out_positions,
        out_is_rejected,
        out_is_masked,
        out_new_token_indices,
        out_hidden_state_mapping,
    )


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "copy_and_expand_eagle_inputs"), "Missing wrapper"
        assert hasattr(mod, "copy_and_expand_eagle_inputs_kernel"), "Missing kernel"
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
    for i, (nr, tpr, nps) in enumerate(TEST_SHAPES):
        try:
            tt, tp, nt, qsl, qel = make_inputs(nr, tpr, nps, device)
            for shift in (False, True):
                result = mod.copy_and_expand_eagle_inputs(
                    tt, tp, nt, qsl, qel,
                    padding_token_id=-1,
                    parallel_drafting_token_id=-2,
                    num_padding_slots_per_request=nps,
                    shift_input_ids=shift,
                    max_output_tokens_per_req=tpr + nps + 5,
                )
                torch.cuda.synchronize()
                ref = reference_copy_and_expand(
                    tt, tp, nt, qsl, qel,
                    padding_token_id=-1,
                    parallel_drafting_token_id=-2,
                    num_padding_slots_per_request=nps,
                    shift_input_ids=shift,
                )
                for j, (got, exp) in enumerate(zip(result, ref)):
                    if not torch.equal(got, exp):
                        return False, f"Shape {i+1}, shift={shift}: output[{j}] mismatch"
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
    nr, tpr, nps = TEST_SHAPES[PERF_SHAPE_IDX]
    tt, tp, nt, qsl, qel = make_inputs(nr, tpr, nps, device)

    for _ in range(5):
        mod.copy_and_expand_eagle_inputs(
            tt, tp, nt, qsl, qel, -1, -2, nps, False, tpr + nps + 5,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.copy_and_expand_eagle_inputs(
            tt, tp, nt, qsl, qel, -1, -2, nps, False, tpr + nps + 5,
        )
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
