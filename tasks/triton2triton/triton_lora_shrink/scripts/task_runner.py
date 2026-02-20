#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lora_shrink"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_lora_shrink"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lora_shrink.py")

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


def reference_lora_shrink(inputs, lora_a_weights, token_indices, num_tokens_per_lora,
                          lora_token_start_loc, lora_ids, scaling):
    """CPU reference for LoRA shrink (A) operation."""
    import torch
    num_slices = len(lora_a_weights)
    M = inputs.shape[0]
    lora_rank = lora_a_weights[0].shape[-2]
    output = torch.zeros(num_slices, M, lora_rank, device=inputs.device, dtype=torch.float32)

    for lora_idx in range(lora_ids.shape[0]):
        lora_id = lora_ids[lora_idx].item()
        if lora_id == -1:
            continue
        n_tokens = num_tokens_per_lora[lora_idx].item()
        start = lora_token_start_loc[lora_idx].item()
        for t in range(n_tokens):
            token_id = token_indices[start + t].item()
            for s in range(num_slices):
                w = lora_a_weights[s]
                if w.ndim == 4:
                    w = w.squeeze(1)
                # w shape: [num_loras, lora_rank, hidden_size]
                inp = inputs[token_id].float()  # [hidden_size]
                weight = w[lora_id].float()  # [lora_rank, hidden_size]
                out_row = inp @ weight.T  # [lora_rank]
                output[s, token_id] = out_row * scaling

    return output.to(inputs.dtype)


def make_test_data(M, hidden_size, lora_rank, num_loras, num_slices, device, seed):
    import torch
    torch.manual_seed(seed)

    inputs = torch.randn(M, hidden_size, device=device, dtype=torch.float16) * 0.1

    lora_a_weights = []
    for _ in range(num_slices):
        w = torch.randn(num_loras, lora_rank, hidden_size, device=device, dtype=torch.float16) * 0.1
        lora_a_weights.append(w)

    output_tensor = torch.zeros(num_slices, M, lora_rank, device=device, dtype=torch.float32)

    token_lora_mapping = torch.randint(0, num_loras, (M,), device=device, dtype=torch.int64)

    lora_ids_list = list(range(num_loras))
    lora_ids = torch.tensor(lora_ids_list, device=device, dtype=torch.int64)

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
    scaling = 0.5

    return (inputs, lora_a_weights, output_tensor, token_lora_mapping,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
            lora_ids, num_active_loras, scaling)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lora_shrink"), "Missing lora_shrink"
        assert hasattr(mod, "_lora_shrink_kernel"), "Missing _lora_shrink_kernel"
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
            (inputs, lora_a_weights, output_tensor, token_lora_mapping,
             token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
             lora_ids, num_active_loras, scaling) = make_test_data(
                M, hidden_size, lora_rank, num_loras, num_slices, device, 42 + i)

            mod.lora_shrink(
                inputs, lora_a_weights, output_tensor, token_lora_mapping,
                token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
                lora_ids, num_active_loras, scaling,
            )
            torch.cuda.synchronize()

            ref = reference_lora_shrink(
                inputs, lora_a_weights, token_indices_sorted, num_tokens_per_lora,
                lora_token_start_loc, lora_ids, scaling).to(device)

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
    (inputs, lora_a_weights, output_tensor, token_lora_mapping,
     token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
     lora_ids, num_active_loras, scaling) = make_test_data(
        M, hidden_size, lora_rank, num_loras, num_slices, device, 0)

    for _ in range(5):
        mod.lora_shrink(
            inputs, lora_a_weights, output_tensor, token_lora_mapping,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
            lora_ids, num_active_loras, scaling,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        output_tensor.zero_()
        start_events[j].record()
        mod.lora_shrink(
            inputs, lora_a_weights, output_tensor, token_lora_mapping,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
            lora_ids, num_active_loras, scaling,
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
