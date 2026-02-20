#!/usr/bin/env python3
"""Task runner for triton2triton/triton_silu_mul_fp8_quant_dg"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_silu_mul_fp8_quant_dg"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_silu_mul_fp8_quant_dg.py")

# (E, T, H, group_size) -- input is [E, T, 2*H]
TEST_SHAPES = [
    (4, 16, 128, 128),
    (4, 32, 256, 128),
    (8, 32, 512, 128),
    (8, 64, 1024, 128),
    (16, 64, 1024, 128),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_silu_mul_fp8(y, tokens_per_expert, group_size):
    """CPU reference: silu(gate) * up with per-group scale."""
    import torch
    E, T, H2 = y.shape
    H = H2 // 2
    G = H // group_size

    try:
        fp8_dtype = torch.float8_e4m3fnuz
        _ = torch.tensor([1.0]).to(fp8_dtype)
    except (RuntimeError, AttributeError):
        fp8_dtype = torch.float8_e4m3fn

    fp8_max = torch.finfo(fp8_dtype).max
    results_float = torch.zeros(E, T, H, dtype=torch.float32)

    for e in range(E):
        nt = tokens_per_expert[e].item()
        for t in range(nt):
            gate = y[e, t, :H].float()
            up = y[e, t, H:].float()
            silu_gate = gate * torch.sigmoid(gate)
            results_float[e, t] = silu_gate * up

    return results_float


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "silu_mul_fp8_quant"), "Missing silu_mul_fp8_quant"
        assert hasattr(mod, "_silu_mul_fp8_quant_deep_gemm"), "Missing _silu_mul_fp8_quant_deep_gemm"
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
    for i, (E, T, H, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            y = torch.randn(E, T, 2 * H, device=device, dtype=torch.float16) * 0.5
            tokens_per_expert = torch.randint(1, T + 1, (E,), device=device, dtype=torch.int32)

            y_q, y_s = mod.silu_mul_fp8_quant(y, tokens_per_expert, group_size)
            torch.cuda.synchronize()

            ref_float = reference_silu_mul_fp8(y.cpu(), tokens_per_expert.cpu(), group_size)

            # Check that dequantized output is close to reference
            y_q_float = y_q.float()
            for e in range(E):
                nt = tokens_per_expert[e].item()
                if nt == 0:
                    continue
                for t in range(min(nt, 4)):  # spot-check
                    for g in range(H // group_size):
                        s = y_s[e, t, g].item()
                        start = g * group_size
                        end = start + group_size
                        deq = y_q_float[e, t, start:end].cpu() * s
                        ref_slice = ref_float[e, t, start:end]
                        if not torch.allclose(deq, ref_slice, atol=0.5, rtol=0.2):
                            max_diff = (deq - ref_slice).abs().max().item()
                            return False, f"Shape {i+1}: e={e},t={t},g={g} max_diff={max_diff:.4f}"
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
    E, T, H, group_size = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    y = torch.randn(E, T, 2 * H, device=device, dtype=torch.float16) * 0.5
    tokens_per_expert = torch.full((E,), T, device=device, dtype=torch.int32)

    for _ in range(5):
        mod.silu_mul_fp8_quant(y, tokens_per_expert, group_size)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.silu_mul_fp8_quant(y, tokens_per_expert, group_size)
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
