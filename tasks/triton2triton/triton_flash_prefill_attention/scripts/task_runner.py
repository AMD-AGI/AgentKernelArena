#!/usr/bin/env python3
"""Task runner for triton2triton/triton_flash_prefill_attention"""
import sys
import os
import json
import argparse
import time
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_flash_prefill_attention"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_flash_prefill_attention.py")

# Test configurations: (batch_size, seq_len, num_heads, num_kv_heads, head_dim)
TEST_SHAPES = [
    (2, 128, 8, 8, 64),     # small, MHA
    (4, 256, 16, 4, 64),    # medium, GQA
    (2, 512, 32, 8, 128),   # large, GQA
    (1, 1024, 16, 16, 64),  # long seq, MHA
    (8, 64, 8, 1, 64),      # batched, MQA
]
PERF_SHAPE_IDX = 2


def load_module():
    """Dynamically load the source module."""
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_attention(q, k, v, b_start_loc, b_seq_len, is_causal=True):
    """
    CPU/PyTorch reference for variable-length packed flash attention.

    q, k, v: [total_tokens, num_heads, head_dim]
    b_start_loc: [batch]
    b_seq_len: [batch]
    Returns: output [total_tokens, num_heads, head_dim]
    """
    import torch
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group_num = num_heads // num_kv_heads

    out = torch.zeros_like(q)
    sm_scale = 1.0 / (head_dim ** 0.5)

    for b in range(len(b_seq_len)):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()

        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_b = q[start:start + seq_len, h, :]  # [S, D]
            k_b = k[start:start + seq_len, kv_h, :]  # [S, D]
            v_b = v[start:start + seq_len, kv_h, :]  # [S, D]

            # [S, S]
            scores = (q_b @ k_b.T) * sm_scale

            if is_causal:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(mask, float("-inf"))

            attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            out[start:start + seq_len, h, :] = attn @ v_b

    return out


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "context_attention_fwd"), "Missing context_attention_fwd"
        assert hasattr(mod, "_fwd_kernel"), "Missing _fwd_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    """Run correctness checks against PyTorch reference."""
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    dtype = torch.float16

    for i, (bs, seq_len, nh, nkv, hd) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            total_tokens = bs * seq_len

            q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
            k = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            v = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            o = torch.zeros_like(q)

            b_seq_len = torch.full((bs,), seq_len, device=device, dtype=torch.int32)
            b_start_loc = torch.zeros(bs, device=device, dtype=torch.int32)
            for j in range(bs):
                b_start_loc[j] = j * seq_len

            # Run Triton kernel
            mod.context_attention_fwd(
                q, k, v, o, b_start_loc, b_seq_len,
                max_input_len=seq_len, is_causal=True,
            )
            torch.cuda.synchronize()

            # Run reference
            ref = reference_attention(q, k, v, b_start_loc, b_seq_len, is_causal=True)

            # Compare
            if not torch.allclose(o, ref, atol=1e-2, rtol=1e-2):
                max_diff = (o - ref).abs().max().item()
                return False, (
                    f"Shape {i + 1} (bs={bs}, seq={seq_len}, nh={nh}, nkv={nkv}, hd={hd}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i + 1} (bs={bs}, seq={seq_len}, nh={nh}, nkv={nkv}, hd={hd}): "
                f"exception: {e}"
            )

    return True, None


def run_performance():
    """Measure kernel execution time."""
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    bs, seq_len, nh, nkv, hd = TEST_SHAPES[PERF_SHAPE_IDX]
    total_tokens = bs * seq_len

    torch.manual_seed(0)
    q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
    k = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
    v = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
    o = torch.zeros_like(q)

    b_seq_len = torch.full((bs,), seq_len, device=device, dtype=torch.int32)
    b_start_loc = torch.zeros(bs, device=device, dtype=torch.int32)
    for j in range(bs):
        b_start_loc[j] = j * seq_len

    # Warmup
    for _ in range(5):
        mod.context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len,
            max_input_len=seq_len, is_causal=True,
        )
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len,
            max_input_len=seq_len, is_causal=True,
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
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        elapsed_ms = run_performance()
        bs, seq_len, nh, nkv, hd = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {
                "batch_size": bs,
                "seq_len": seq_len,
                "num_heads": nh,
                "num_kv_heads": nkv,
                "head_dim": hd,
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
