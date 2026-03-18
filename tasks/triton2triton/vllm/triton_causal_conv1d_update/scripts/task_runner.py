#!/usr/bin/env python3
"""Task runner for triton_causal_conv1d_update"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_causal_conv1d_update.py")

# (batch, dim, seqlen, width, has_bias, activation)
TEST_SHAPES = [
    (4, 64, 1, 4, True, "silu"),
    (8, 128, 1, 4, False, "silu"),
    (2, 256, 1, 3, True, None),
    (16, 64, 1, 4, True, "silu"),
    (4, 128, 1, 4, True, "silu"),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_conv1d_update(x, conv_state, weight, bias, activation):
    """CPU reference: x is (batch, dim, seqlen), conv_state is (batch, dim, state_len)"""
    import torch, math
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    out = torch.zeros_like(x, dtype=torch.float32)
    for b in range(batch):
        for d in range(dim):
            state = conv_state[b, d, :state_len].float().tolist()
            for t in range(seqlen):
                full = state + [x[b, d, t].float().item()]
                val = 0.0
                if bias is not None:
                    val = bias[d].float().item()
                for w in range(width):
                    val += full[w] * weight[d, w].float().item()
                if activation in ["silu", "swish"]:
                    val = val / (1.0 + math.exp(-val))
                out[b, d, t] = val
                state = state[1:] + [x[b, d, t].float().item()]
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_causal_conv1d_update_kernel")
        assert hasattr(mod, "causal_conv1d_update")
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Load failed: {e}"
    device = "cuda"
    for i, (batch, dim, seqlen, width, has_bias, activation) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            state_len = width - 1
            x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
            weight = torch.randn(dim, width, device=device, dtype=torch.float32)
            bias_t = torch.randn(dim, device=device, dtype=torch.float32) if has_bias else None
            # conv_state needs stride(-2)==1 (contiguous along dim)
            conv_state = torch.randn(batch, dim, state_len, device=device, dtype=torch.float32)
            conv_state = conv_state.transpose(1, 2).contiguous().transpose(1, 2)
            conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)

            ref = reference_conv1d_update(x, conv_state, weight, bias_t, activation).to(device)
            result = mod.causal_conv1d_update(x.clone(), conv_state.clone(), weight, bias=bias_t,
                                              activation=activation, conv_state_indices=conv_state_indices)
            if not torch.allclose(result.float(), ref.float(), atol=1e-1, rtol=1e-1):
                diff = (result.float() - ref.float()).abs().max().item()
                return False, f"Shape {i}: max diff={diff}"
        except Exception as e:
            return False, f"Shape {i}: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch, dim, seqlen, width, has_bias, activation) in enumerate(TEST_SHAPES):
        try:
            state_len = width - 1
            torch.manual_seed(42 + test_idx)
            x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
            weight = torch.randn(dim, width, device=device, dtype=torch.float32)
            bias_t = torch.randn(dim, device=device, dtype=torch.float32) if has_bias else None
            conv_state = torch.randn(batch, dim, state_len, device=device, dtype=torch.float32)
            conv_state = conv_state.transpose(1, 2).contiguous().transpose(1, 2)
            conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
            for _ in range(WARMUP_ITERATIONS):
                mod.causal_conv1d_update(x.clone(), conv_state.clone(), weight, bias=bias_t,
                                         activation=activation, conv_state_indices=conv_state_indices)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                starts[j].record()
                mod.causal_conv1d_update(x.clone(), conv_state.clone(), weight, bias=bias_t,
                                         activation=activation, conv_state_indices=conv_state_indices)
                ends[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": {
                    "batch": batch,
                    "dim": dim,
                    "seqlen": seqlen,
                    "width": width,
                    "has_bias": has_bias,
                    "activation": activation
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "dim": dim,
                    "seqlen": seqlen,
                    "width": width,
                    "has_bias": has_bias,
                    "activation": activation
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser()
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
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
