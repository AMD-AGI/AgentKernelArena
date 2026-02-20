#!/usr/bin/env python3
"""Task runner for triton_prepare_eagle_docode"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_prepare_eagle_docode"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_eagle_docode.py")

# (num_reqs, total_tokens, hidden_size, max_model_len, max_num_reqs)
TEST_SHAPES = [
    (4, 64, 256, 2048, 8),
    (8, 128, 512, 4096, 16),
    (16, 256, 768, 4096, 32),
    (32, 512, 1024, 8192, 64),
    (64, 1024, 2048, 8192, 128),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(num_reqs, total_tokens, hidden_size, max_model_len, max_num_reqs, device="cpu"):
    import torch
    torch.manual_seed(42)
    draft_tokens = torch.randint(0, 32000, (num_reqs,), dtype=torch.int64)
    output_hidden_states = torch.randn(total_tokens, hidden_size, dtype=torch.float16)
    last_token_indices = torch.randint(0, total_tokens, (num_reqs,), dtype=torch.int64)
    target_seq_lens = torch.randint(10, max_model_len // 2, (num_reqs,), dtype=torch.int32)
    num_rejected = torch.randint(0, 3, (num_reqs,), dtype=torch.int32)
    positions = torch.randint(0, max_model_len - 2, (max(total_tokens, max_num_reqs),), dtype=torch.int32)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    input_ids = torch.zeros(max(total_tokens, max_num_reqs), dtype=torch.int64)
    input_hidden_states = torch.zeros(max(total_tokens, max_num_reqs), hidden_size, dtype=torch.float16)

    if device != "cpu":
        draft_tokens = draft_tokens.to(device)
        output_hidden_states = output_hidden_states.to(device)
        last_token_indices = last_token_indices.to(device)
        target_seq_lens = target_seq_lens.to(device)
        num_rejected = num_rejected.to(device)
        positions = positions.to(device)
        seq_lens = seq_lens.to(device)
        query_start_loc = query_start_loc.to(device)
        input_ids = input_ids.to(device)
        input_hidden_states = input_hidden_states.to(device)

    return (draft_tokens, output_hidden_states, last_token_indices,
            target_seq_lens, num_rejected, positions, seq_lens,
            query_start_loc, input_ids, input_hidden_states)


def reference(draft_tokens, output_hs, last_ti, target_sl, num_rej,
              positions, seq_lens, qsl, input_ids, input_hs, max_ml, max_nr):
    import torch
    num_reqs = draft_tokens.shape[0]
    hidden_size = output_hs.shape[-1]

    # Work on clones
    positions = positions.clone()
    seq_lens = seq_lens.clone()
    qsl = qsl.clone()
    input_ids = input_ids.clone()
    input_hs = input_hs.clone()

    for r in range(num_reqs):
        input_ids[r] = draft_tokens[r]
        src = last_ti[r].item()
        input_hs[r] = output_hs[src]
        pos = min(positions[r].item() + 1, max_ml - 1)
        positions[r] = pos
        sl = target_sl[r].item() - num_rej[r].item()
        sl = min(sl + 1, max_ml)
        seq_lens[r] = sl

    # query_start_loc
    for i in range(max_nr + 1):
        if i < num_reqs:
            qsl[i] = i
        else:
            qsl[i] = num_reqs

    # pad seq_lens
    for i in range(num_reqs, max_nr):
        seq_lens[i] = 0

    return positions, seq_lens, qsl, input_ids, input_hs


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_eagle_decode"), "Missing wrapper"
        assert hasattr(mod, "_prepare_eagle_docode_kernel"), "Missing kernel"
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
    for i, (nr, tt, hs, mml, mnr) in enumerate(TEST_SHAPES):
        try:
            inputs_gpu = make_inputs(nr, tt, hs, mml, mnr, device)
            inputs_cpu = make_inputs(nr, tt, hs, mml, mnr, "cpu")

            mod.prepare_eagle_decode(
                inputs_gpu[0], inputs_gpu[1], inputs_gpu[2], inputs_gpu[3],
                inputs_gpu[4], inputs_gpu[5], inputs_gpu[6], inputs_gpu[7],
                inputs_gpu[8], inputs_gpu[9], mml, mnr,
            )
            torch.cuda.synchronize()

            ref = reference(
                inputs_cpu[0], inputs_cpu[1], inputs_cpu[2], inputs_cpu[3],
                inputs_cpu[4], inputs_cpu[5], inputs_cpu[6], inputs_cpu[7],
                inputs_cpu[8], inputs_cpu[9], mml, mnr,
            )

            # Check input_ids
            if not torch.equal(inputs_gpu[8][:nr].cpu(), ref[3][:nr]):
                return False, f"Shape {i+1}: input_ids mismatch"
            # Check positions
            if not torch.equal(inputs_gpu[5][:nr].cpu(), ref[0][:nr]):
                return False, f"Shape {i+1}: positions mismatch"
            # Check seq_lens
            if not torch.equal(inputs_gpu[6].cpu(), ref[1]):
                return False, f"Shape {i+1}: seq_lens mismatch"
            # Check query_start_loc
            if not torch.equal(inputs_gpu[7].cpu(), ref[2]):
                return False, f"Shape {i+1}: query_start_loc mismatch"
            # Check hidden states
            if not torch.allclose(inputs_gpu[9][:nr].cpu().float(), ref[4][:nr].float(), atol=1e-3, rtol=1e-3):
                return False, f"Shape {i+1}: hidden states mismatch"

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
    nr, tt, hs, mml, mnr = TEST_SHAPES[PERF_SHAPE_IDX]
    inputs = make_inputs(nr, tt, hs, mml, mnr, device)

    for _ in range(5):
        mod.prepare_eagle_decode(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], mml, mnr,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.prepare_eagle_decode(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], mml, mnr,
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
