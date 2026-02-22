#!/usr/bin/env python3
"""Task runner for triton2triton/triton_decode_attn_stage2"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_decode_attn_stage2"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "_fwd_kernel_stage2.py")

# Test configurations: (bs, num_heads, num_kv_heads, head_dim, max_seq, num_kv_splits, page_size)
TEST_SHAPES = [
    (1, 8, 8, 64, 128, 4, 16),
    (4, 16, 4, 64, 256, 8, 16),
    (2, 32, 8, 128, 512, 4, 32),
    (1, 8, 1, 64, 64, 2, 16),    # MQA
    (8, 8, 8, 64, 128, 4, 16),
]
PERF_SHAPE_IDX = 2


def load_module():
    """Dynamically load the source module."""
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_stage2(mid_o, b_seqlen, num_kv_splits, Lv):
    """
    CPU/PyTorch reference for decode attention stage2.

    Combines partial results from stage1 via logsumexp:
      max_lse = max(lse across splits)
      o = sum(exp(lse_i - max_lse) * mid_o_i) / sum(exp(lse_i - max_lse))
      final_lse = max_lse + log(sum(exp(lse_i - max_lse)))
    """
    import torch
    batch, num_heads = mid_o.shape[0], mid_o.shape[1]
    head_dim = Lv

    o = torch.zeros(batch, num_heads, head_dim, device=mid_o.device, dtype=torch.float32)
    lse = torch.zeros(batch, num_heads, device=mid_o.device, dtype=torch.float32)

    for b in range(batch):
        seq_len = b_seqlen[b].item()
        kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits

        for h in range(num_heads):
            e_max = -float("inf")
            e_sum = 0.0
            acc = torch.zeros(head_dim, device=mid_o.device, dtype=torch.float32)

            for s in range(num_kv_splits):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, seq_len)
                if end <= start:
                    continue

                tv = mid_o[b, h, s, :Lv].float()
                tlogic = mid_o[b, h, s, Lv].float().item()

                n_e_max = max(tlogic, e_max)
                old_scale = torch.exp(torch.tensor(e_max - n_e_max))
                acc = acc * old_scale
                exp_logic = torch.exp(torch.tensor(tlogic - n_e_max))
                acc = acc + exp_logic * tv

                e_sum = e_sum * old_scale.item() + exp_logic.item()
                e_max = n_e_max

            if e_sum > 0:
                o[b, h, :] = acc / e_sum
                lse[b, h] = e_max + torch.log(torch.tensor(e_sum)).item()

    return o, lse


def make_stage1_outputs(bs, num_heads, num_kv_heads, head_dim, max_seq,
                        num_kv_splits, page_size, device="cuda", dtype=None):
    """
    Create synthetic stage1 outputs for testing stage2.

    We simulate what stage1 would produce: for each split, create a random
    partial attention output and a plausible logsumexp value.
    """
    import torch
    if dtype is None:
        dtype = torch.float16

    torch.manual_seed(42)
    Lv = head_dim

    # mid_o: [batch, num_heads, num_kv_splits, head_dim + 1]
    mid_o = torch.zeros(bs, num_heads, num_kv_splits, Lv + 1,
                        device=device, dtype=torch.float32)

    b_seqlen = torch.full((bs,), max_seq, device=device, dtype=torch.int32)
    kv_len_per_split = (max_seq + num_kv_splits - 1) // num_kv_splits

    for b in range(bs):
        for h in range(num_heads):
            for s in range(num_kv_splits):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, max_seq)
                if end <= start:
                    continue
                # Random partial output (normalized, like after softmax @ V)
                mid_o[b, h, s, :Lv] = torch.randn(Lv, device=device)
                # Plausible logsumexp value
                mid_o[b, h, s, Lv] = torch.randn(1, device=device).item() * 2.0

    # Output tensors
    o = torch.zeros(bs, num_heads, head_dim, device=device, dtype=dtype)
    lse = torch.zeros(bs, num_heads, device=device, dtype=torch.float32)

    # v_buffer shape info (only Lv used)
    v_buffer = torch.empty(1, num_kv_heads, head_dim, device=device, dtype=dtype)

    # q shape info
    q = torch.empty(bs, num_heads, head_dim, device=device, dtype=dtype)

    return mid_o, q, o, lse, v_buffer, b_seqlen


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "decode_softmax_reducev_fwd"), "Missing decode_softmax_reducev_fwd"
        assert hasattr(mod, "_fwd_kernel_stage2"), "Missing _fwd_kernel_stage2"
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

    for i, (bs, nh, nkv, hd, max_seq, num_splits, ps) in enumerate(TEST_SHAPES):
        try:
            mid_o, q, o, lse, v_buffer, b_seqlen = \
                make_stage1_outputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype)

            mod.decode_softmax_reducev_fwd(
                mid_o, q, o, lse, v_buffer, b_seqlen, num_splits,
            )
            torch.cuda.synchronize()

            ref_o, ref_lse = reference_stage2(mid_o, b_seqlen, num_splits, hd)

            o_f32 = o.float()
            if not torch.allclose(o_f32, ref_o, atol=1e-2, rtol=1e-2):
                max_diff = (o_f32 - ref_o).abs().max().item()
                return False, (
                    f"Shape {i+1} output (bs={bs}, nh={nh}, hd={hd}, "
                    f"splits={num_splits}): max diff = {max_diff:.6f}"
                )

            if not torch.allclose(lse, ref_lse, atol=1e-2, rtol=1e-2):
                max_diff = (lse - ref_lse).abs().max().item()
                return False, (
                    f"Shape {i+1} lse (bs={bs}, nh={nh}, hd={hd}, "
                    f"splits={num_splits}): max diff = {max_diff:.6f}"
                )

        except Exception as e:
            return False, (
                f"Shape {i+1} (bs={bs}, nh={nh}, nkv={nkv}, hd={hd}, "
                f"seq={max_seq}, splits={num_splits}, ps={ps}): "
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
    bs, nh, nkv, hd, max_seq, num_splits, ps = TEST_SHAPES[PERF_SHAPE_IDX]

    mid_o, q, o, lse, v_buffer, b_seqlen = \
        make_stage1_outputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype)

    # Warmup
    for _ in range(5):
        o.zero_()
        lse.zero_()
        mod.decode_softmax_reducev_fwd(
            mid_o, q, o, lse, v_buffer, b_seqlen, num_splits,
        )
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        o.zero_()
        lse.zero_()
        start_events[j].record()
        mod.decode_softmax_reducev_fwd(
            mid_o, q, o, lse, v_buffer, b_seqlen, num_splits,
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
        bs, nh, nkv, hd, max_seq, num_splits, ps = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {
                "batch_size": bs, "num_heads": nh, "num_kv_heads": nkv,
                "head_dim": hd, "max_seq": max_seq,
                "num_kv_splits": num_splits, "page_size": ps,
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
