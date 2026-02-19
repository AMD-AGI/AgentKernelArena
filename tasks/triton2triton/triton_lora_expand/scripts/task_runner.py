#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lora_expand"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_lora_expand"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lora_expand.py")

# (M, hidden_size, lora_rank, num_loras, num_slices)
TEST_SHAPES = [
    (16, 64, 8, 2, 1),
    (32, 128, 16, 4, 1),
    (64, 256, 16, 4, 2),
    (128, 512, 32, 8, 1),
    (256, 1024, 32, 8, 2),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_lora_expand(inputs, lora_b_weights, token_indices, num_tokens_per_lora,
                          lora_token_start_loc, lora_ids, offset_start, add_inputs,
                          output_tensor):
    """CPU reference for LoRA expand (B) operation."""
    import torch
    num_slices = inputs.shape[0]
    M = inputs.shape[1]
    result = output_tensor.clone().float()

    for lora_idx in range(lora_ids.shape[0]):
        lora_id = lora_ids[lora_idx].item()
        if lora_id == -1:
            continue
        n_tokens = num_tokens_per_lora[lora_idx].item()
        start = lora_token_start_loc[lora_idx].item()
        for t in range(n_tokens):
            token_id = token_indices[start + t].item()
            for s in range(num_slices):
                w = lora_b_weights[s]
                if w.ndim == 4:
                    w = w.squeeze(1)
                # w shape: [num_loras, hidden_size, lora_rank]
                # inputs shape: [num_slices, M, lora_rank]
                inp = inputs[s, token_id].float()
                weight = w[lora_id].float()  # [hidden_size, lora_rank]
                out_row = inp @ weight.T  # [hidden_size]
                hidden_size = w.shape[1]
                col_start = offset_start + s * hidden_size
                col_end = col_start + hidden_size
                if add_inputs:
                    result[token_id, col_start:col_end] += out_row
                else:
                    result[token_id, col_start:col_end] = out_row

    return result.to(inputs.dtype)


def make_test_data(M, hidden_size, lora_rank, num_loras, num_slices, device, seed):
    import torch
    torch.manual_seed(seed)

    inputs = torch.randn(num_slices, M, lora_rank, device=device, dtype=torch.float16) * 0.1

    lora_b_weights = []
    for _ in range(num_slices):
        w = torch.randn(num_loras, hidden_size, lora_rank, device=device, dtype=torch.float16) * 0.1
        lora_b_weights.append(w)

    total_out_dim = hidden_size * num_slices
    output_tensor = torch.zeros(M, total_out_dim, device=device, dtype=torch.float16)

    # Assign each token to a random lora (all tokens get a lora for simplicity)
    token_lora_mapping = torch.randint(0, num_loras, (M,), device=device, dtype=torch.int64)

    # Build sorted indices by lora
    lora_ids_list = list(range(num_loras))
    lora_ids = torch.tensor(lora_ids_list, device=device, dtype=torch.int64)

    # Sort tokens by lora id
    sorted_indices = []
    num_tokens_list = []
    for lid in lora_ids_list:
        mask = (token_lora_mapping == lid)
        indices = mask.nonzero(as_tuple=True)[0]
        sorted_indices.append(indices)
        num_tokens_list.append(len(indices))

    token_indices_sorted = torch.cat(sorted_indices).to(device)
    num_tokens_per_lora = torch.tensor(num_tokens_list, device=device, dtype=torch.int64)
    cumsum = [0]
    for n in num_tokens_list:
        cumsum.append(cumsum[-1] + n)
    lora_token_start_loc = torch.tensor(cumsum, device=device, dtype=torch.int64)

    num_active_loras = num_loras

    return (inputs, lora_b_weights, output_tensor, token_lora_mapping,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
            lora_ids, num_active_loras)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lora_expand"), "Missing lora_expand"
        assert hasattr(mod, "_lora_expand_kernel"), "Missing _lora_expand_kernel"
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
    for i, (M, hidden_size, lora_rank, num_loras, num_slices) in enumerate(TEST_SHAPES):
        try:
            (inputs, lora_b_weights, output_tensor, token_lora_mapping,
             token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
             lora_ids, num_active_loras) = make_test_data(
                M, hidden_size, lora_rank, num_loras, num_slices, device, 42 + i)

            mod.lora_expand(
                inputs, lora_b_weights, output_tensor, token_lora_mapping,
                token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
                lora_ids, num_active_loras, offset_start=0, add_inputs=False,
            )
            torch.cuda.synchronize()

            ref_output = torch.zeros_like(output_tensor)
            ref = reference_lora_expand(
                inputs, lora_b_weights, token_indices_sorted, num_tokens_per_lora,
                lora_token_start_loc, lora_ids, 0, False, ref_output).to(device)

            if not torch.allclose(output_tensor.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (output_tensor.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1} (M={M}): max diff = {max_diff:.6f}"
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
    M, hidden_size, lora_rank, num_loras, num_slices = TEST_SHAPES[PERF_SHAPE_IDX]
    (inputs, lora_b_weights, output_tensor, token_lora_mapping,
     token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
     lora_ids, num_active_loras) = make_test_data(
        M, hidden_size, lora_rank, num_loras, num_slices, device, 0)

    for _ in range(5):
        mod.lora_expand(
            inputs, lora_b_weights, output_tensor, token_lora_mapping,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
            lora_ids, num_active_loras, offset_start=0, add_inputs=False,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        output_tensor.zero_()
        start_events[j].record()
        mod.lora_expand(
            inputs, lora_b_weights, output_tensor, token_lora_mapping,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
            lora_ids, num_active_loras, offset_start=0, add_inputs=False,
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
