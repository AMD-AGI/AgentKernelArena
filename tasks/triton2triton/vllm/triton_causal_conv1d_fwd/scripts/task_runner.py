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
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<

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
        return []
    device = "cuda"
    test_cases = []

    for test_idx, (dim, seqlens_list, width, has_bias, activation) in enumerate(TEST_SHAPES):
        try:
            batch = len(seqlens_list)
            cu_seqlen = sum(seqlens_list)
            torch.manual_seed(42 + test_idx)
            x = torch.randn(dim, cu_seqlen, device=device, dtype=torch.float32)
            weight = torch.randn(dim, width, device=device, dtype=torch.float32)
            bias_t = torch.randn(dim, device=device, dtype=torch.float32) if has_bias else None
            conv_states = torch.randn(batch, dim, width - 1, device=device, dtype=torch.float32)
            conv_states = conv_states.transpose(1, 2).contiguous().transpose(1, 2)
            initial_conv_states = conv_states.clone()
            query_start_loc = torch.zeros(batch + 1, device=device, dtype=torch.int32)
            for b in range(batch):
                query_start_loc[b + 1] = query_start_loc[b] + seqlens_list[b]
            cache_indices = torch.arange(batch, device=device, dtype=torch.int32)
            has_init = torch.ones(batch, device=device, dtype=torch.int32)

            # The public wrapper derives this launch map through a CPU round-trip
            # from query_start_loc.  Test shapes already provide the same values
            # on the host, so materialize the map once outside the timed region.
            block_m = 8
            block_n = 256
            batch_programs = []
            token_chunk_offsets = []
            for batch_idx, seq_len in enumerate(seqlens_list):
                num_chunks = (seq_len + block_m - 1) // block_m
                batch_programs.extend([batch_idx] * num_chunks)
                token_chunk_offsets.extend(range(num_chunks))
            batch_ptr = torch.tensor(
                batch_programs, device=device, dtype=torch.int32
            )
            token_chunk_offset_ptr = torch.tensor(
                token_chunk_offsets, device=device, dtype=torch.int32
            )
            out = torch.empty_like(x)
            state_len = width - 1
            np2_statelen = 1 << (state_len - 1).bit_length()
            grid = (
                len(batch_programs),
                (dim + block_n - 1) // block_n,
            )

            def _bench_fn():
                # Launch the target kernel directly so host-side shape discovery
                # and temporary allocations in causal_conv1d_fwd are excluded.
                mod._causal_conv1d_fwd_kernel[grid](
                    x,
                    weight,
                    bias_t,
                    conv_states,
                    cache_indices,
                    has_init,
                    query_start_loc,
                    batch_ptr,
                    token_chunk_offset_ptr,
                    None,
                    None,
                    None,
                    None,
                    out,
                    dim,
                    cu_seqlen,
                    batch,
                    x.stride(0),
                    x.stride(1),
                    weight.stride(0),
                    weight.stride(1),
                    conv_states.stride(0),
                    conv_states.stride(1),
                    conv_states.stride(2),
                    cache_indices.stride(0),
                    out.stride(0),
                    out.stride(1),
                    1,
                    mod.PAD_SLOT_ID,
                    HAS_BIAS=bias_t is not None,
                    KERNEL_WIDTH=width,
                    SILU_ACTIVATION=activation in ["silu", "swish"],
                    IS_APC_ENABLED=False,
                    USE_PAD_SLOT=True,
                    NP2_STATELEN=np2_statelen,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_stages=2,
                )
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                prepare_fn=lambda: conv_states.copy_(initial_conv_states),
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "dim": dim,
                    "seqlens_list": seqlens_list,
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
                    "dim": dim,
                    "seqlens_list": seqlens_list,
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
