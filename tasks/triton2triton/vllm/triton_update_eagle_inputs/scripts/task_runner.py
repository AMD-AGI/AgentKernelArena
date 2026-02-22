#!/usr/bin/env python3
"""Task runner for triton_update_eagle_inputs"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_update_eagle_inputs"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_update_eagle_inputs.py")

# (num_reqs, hidden_size, max_model_len)
TEST_SHAPES = [
    (4, 256, 2048),
    (8, 512, 4096),
    (16, 768, 4096),
    (32, 1024, 8192),
    (64, 2048, 8192),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(num_reqs, hidden_size, max_model_len, device="cpu"):
    import torch
    torch.manual_seed(42)
    draft_tokens = torch.randint(0, 32000, (num_reqs,), dtype=torch.int64)
    output_hs = torch.randn(num_reqs, hidden_size, dtype=torch.float16)
    input_ids = torch.zeros(num_reqs, dtype=torch.int64)
    positions = torch.randint(0, max_model_len - 2, (num_reqs,), dtype=torch.int32)
    input_hs = torch.zeros(num_reqs, hidden_size, dtype=torch.float16)
    seq_lens = torch.randint(1, max_model_len - 1, (num_reqs,), dtype=torch.int32)

    if device != "cpu":
        draft_tokens = draft_tokens.to(device)
        output_hs = output_hs.to(device)
        input_ids = input_ids.to(device)
        positions = positions.to(device)
        input_hs = input_hs.to(device)
        seq_lens = seq_lens.to(device)
    return draft_tokens, output_hs, input_ids, positions, input_hs, seq_lens


def reference(draft_tokens, output_hs, input_ids, positions, input_hs, seq_lens, max_ml):
    import torch
    num_reqs = draft_tokens.shape[0]
    input_ids = input_ids.clone()
    positions = positions.clone()
    input_hs = input_hs.clone()
    seq_lens = seq_lens.clone()

    for r in range(num_reqs):
        input_ids[r] = draft_tokens[r]
        input_hs[r] = output_hs[r]
        positions[r] = min(positions[r].item() + 1, max_ml - 1)
        seq_lens[r] = min(seq_lens[r].item() + 1, max_ml)

    return input_ids, positions, input_hs, seq_lens


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "update_eagle_inputs"), "Missing wrapper"
        assert hasattr(mod, "_update_eagle_inputs_kernel"), "Missing kernel"
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
    for i, (nr, hs, mml) in enumerate(TEST_SHAPES):
        try:
            gpu_inputs = make_inputs(nr, hs, mml, device)
            cpu_inputs = make_inputs(nr, hs, mml, "cpu")

            mod.update_eagle_inputs(
                gpu_inputs[0], gpu_inputs[1], gpu_inputs[2],
                gpu_inputs[3], gpu_inputs[4], gpu_inputs[5], mml,
            )
            torch.cuda.synchronize()

            ref = reference(cpu_inputs[0], cpu_inputs[1], cpu_inputs[2],
                          cpu_inputs[3], cpu_inputs[4], cpu_inputs[5], mml)

            if not torch.equal(gpu_inputs[2].cpu(), ref[0]):
                return False, f"Shape {i+1}: input_ids mismatch"
            if not torch.equal(gpu_inputs[3].cpu(), ref[1]):
                return False, f"Shape {i+1}: positions mismatch"
            if not torch.allclose(gpu_inputs[4].cpu().float(), ref[2].float(), atol=1e-3, rtol=1e-3):
                return False, f"Shape {i+1}: hidden states mismatch"
            if not torch.equal(gpu_inputs[5].cpu(), ref[3]):
                return False, f"Shape {i+1}: seq_lens mismatch"
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
    nr, hs, mml = TEST_SHAPES[PERF_SHAPE_IDX]
    inputs = make_inputs(nr, hs, mml, device)

    for _ in range(5):
        mod.update_eagle_inputs(inputs[0], inputs[1], inputs[2],
                               inputs[3], inputs[4], inputs[5], mml)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.update_eagle_inputs(inputs[0], inputs[1], inputs[2],
                               inputs[3], inputs[4], inputs[5], mml)
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
