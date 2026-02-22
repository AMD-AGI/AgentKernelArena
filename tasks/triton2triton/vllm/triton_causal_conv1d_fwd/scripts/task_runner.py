#!/usr/bin/env python3
"""Task runner for triton_causal_conv1d_fwd"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_causal_conv1d_fwd.py")

# (dim, seqlens_list, width, has_bias, activation)
TEST_SHAPES = [
    (64, [8, 4], 4, True, "silu"),
    (128, [16, 8, 4], 4, False, "silu"),
    (256, [32], 4, True, None),
    (64, [16, 16], 3, True, "silu"),
    (128, [8, 8, 8, 8], 4, True, "silu"),
]
PERF_IDX = 1


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_conv1d(x_2d, weight, bias, conv_states, query_start_loc, has_initial_state, activation):
    """CPU reference for causal conv1d: x is (dim, cu_seqlen), weight is (dim, width)"""
    import torch
    dim, cu_seqlen = x_2d.shape
    _, width = weight.shape
    batch = len(query_start_loc) - 1
    out = torch.zeros_like(x_2d, dtype=torch.float32)
    for b in range(batch):
        s = query_start_loc[b].item()
        e = query_start_loc[b + 1].item()
        slen = e - s
        for d in range(dim):
            # Build input with state prepended
            if has_initial_state[b]:
                state = conv_states[b, d, :width-1].float().tolist()
            else:
                state = [0.0] * (width - 1)
            seq = x_2d[d, s:e].float().tolist()
            full = state + seq
            for t in range(slen):
                val = 0.0
                if bias is not None:
                    val = bias[d].float().item()
                for w in range(width):
                    idx = t + w
                    val += full[idx] * weight[d, w].float().item()
                if activation in ["silu", "swish"]:
                    import math
                    val = val / (1.0 + math.exp(-val))
                out[d, s + t] = val
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_causal_conv1d_fwd_kernel")
        assert hasattr(mod, "causal_conv1d_fwd")
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
    for i, (dim, seqlens_list, width, has_bias, activation) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            batch = len(seqlens_list)
            cu_seqlen = sum(seqlens_list)
            x = torch.randn(dim, cu_seqlen, device=device, dtype=torch.float32)
            weight = torch.randn(dim, width, device=device, dtype=torch.float32)
            weight_cl = weight.clone()
            # Ensure weight stride(1)==1 (contiguous along width)
            if weight_cl.stride(1) != 1:
                weight_cl = weight_cl.contiguous()
            bias_t = torch.randn(dim, device=device, dtype=torch.float32) if has_bias else None
            conv_states = torch.randn(batch, dim, width - 1, device=device, dtype=torch.float32)
            # Make conv_states contiguous along dim (stride(-2)==1)
            conv_states = conv_states.transpose(1, 2).contiguous().transpose(1, 2)
            query_start_loc = torch.zeros(batch + 1, device=device, dtype=torch.int32)
            for b in range(batch):
                query_start_loc[b + 1] = query_start_loc[b] + seqlens_list[b]
            cache_indices = torch.arange(batch, device=device, dtype=torch.int32)
            has_init = torch.ones(batch, device=device, dtype=torch.int32)

            conv_states_ref = conv_states.clone()
            result = mod.causal_conv1d_fwd(x, weight_cl, bias_t, conv_states, query_start_loc,
                                           cache_indices, has_init, activation=activation)
            ref = reference_conv1d(x, weight_cl, bias_t, conv_states_ref, query_start_loc.cpu(),
                                   has_init.cpu(), activation).to(device)
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
        return -1.0
    device = "cuda"
    dim, seqlens_list, width, has_bias, activation = TEST_SHAPES[PERF_IDX]
    batch = len(seqlens_list)
    cu_seqlen = sum(seqlens_list)
    torch.manual_seed(0)
    x = torch.randn(dim, cu_seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias_t = torch.randn(dim, device=device, dtype=torch.float32) if has_bias else None
    conv_states = torch.randn(batch, dim, width - 1, device=device, dtype=torch.float32)
    conv_states = conv_states.transpose(1, 2).contiguous().transpose(1, 2)
    query_start_loc = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    for b in range(batch):
        query_start_loc[b + 1] = query_start_loc[b] + seqlens_list[b]
    cache_indices = torch.arange(batch, device=device, dtype=torch.int32)
    has_init = torch.ones(batch, device=device, dtype=torch.int32)
    for _ in range(5):
        mod.causal_conv1d_fwd(x, weight, bias_t, conv_states, query_start_loc,
                              cache_indices, has_init, activation=activation)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.causal_conv1d_fwd(x, weight, bias_t, conv_states, query_start_loc,
                              cache_indices, has_init, activation=activation)
        ends[j].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times) / len(times)


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
        ms = run_performance()
        report = {"execution_time_ms": ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
