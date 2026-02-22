#!/usr/bin/env python3
"""Task runner for triton2triton/triton_decode_attn_grouped_stage1"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_decode_attn_grouped_stage1"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "_fwd_grouped_kernel_stage1.py")

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


def reference_grouped_stage1(q, k_buffer, v_buffer, req_to_tokens, b_seqlen,
                              num_kv_splits, sm_scale, page_size):
    """
    CPU/PyTorch reference for decode attention grouped stage1.

    For each (batch, head, kv_split), compute partial attention over
    the assigned KV range and return partial output + logsumexp.
    Same as stage1 but handles grouped-query attention explicitly.
    """
    import torch
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    Lv = v_buffer.shape[-1]

    att_out = torch.zeros(batch, num_heads, num_kv_splits, Lv + 1,
                          device=q.device, dtype=torch.float32)

    for b in range(batch):
        seq_len = b_seqlen[b].item()
        kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits

        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_vec = q[b, h, :].float()  # [head_dim]

            for s in range(num_kv_splits):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, seq_len)
                if end <= start:
                    continue

                # Gather K and V via paged token table
                positions = torch.arange(start, end, device=q.device)
                page_nums = req_to_tokens[b, positions // page_size]
                kv_locs = page_nums * page_size + positions % page_size

                k_vals = k_buffer[kv_locs, kv_h, :].float()  # [length, head_dim]
                v_vals = v_buffer[kv_locs, kv_h, :].float()  # [length, Lv]

                # Q @ K^T * sm_scale
                scores = (k_vals @ q_vec) * sm_scale  # [length]

                # Numerically stable softmax
                max_score = scores.max()
                exp_scores = torch.exp(scores - max_score)
                sum_exp = exp_scores.sum()

                # Partial output
                partial_out = (exp_scores.unsqueeze(-1) * v_vals).sum(0) / sum_exp

                att_out[b, h, s, :Lv] = partial_out
                att_out[b, h, s, Lv] = max_score + torch.log(sum_exp)

    return att_out


def make_inputs(bs, num_heads, num_kv_heads, head_dim, max_seq, num_kv_splits,
                page_size, device="cuda", dtype=None):
    """Create test inputs for the grouped stage1 kernel."""
    import torch
    if dtype is None:
        dtype = torch.float16

    torch.manual_seed(42)

    q = torch.randn(bs, num_heads, head_dim, device=device, dtype=dtype)

    # Total tokens in KV buffer
    max_pages_per_seq = (max_seq + page_size - 1) // page_size
    total_pages = bs * max_pages_per_seq
    total_tokens = total_pages * page_size

    k_buffer = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_buffer = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Build req_to_tokens: identity mapping
    max_seq_padded = max_pages_per_seq * page_size
    req_to_tokens = torch.zeros(bs, max_seq_padded, device=device, dtype=torch.int32)
    for b in range(bs):
        for pos in range(max_seq):
            page_idx = b * max_pages_per_seq + pos // page_size
            req_to_tokens[b, pos] = page_idx

    b_seqlen = torch.full((bs,), max_seq, device=device, dtype=torch.int32)

    # att_out: [batch, num_heads, num_kv_splits, head_dim + 1]
    att_out = torch.zeros(bs, num_heads, num_kv_splits, head_dim + 1,
                          device=device, dtype=torch.float32)

    sm_scale = 1.0 / (head_dim ** 0.5)

    return q, k_buffer, v_buffer, att_out, req_to_tokens, b_seqlen, sm_scale


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "decode_grouped_att_m_fwd"), "Missing decode_grouped_att_m_fwd"
        assert hasattr(mod, "_fwd_grouped_kernel_stage1"), "Missing _fwd_grouped_kernel_stage1"
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
            q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen, sm_scale = \
                make_inputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype)

            mod.decode_grouped_att_m_fwd(
                q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen,
                num_splits, sm_scale, ps, logit_cap=0.0,
            )
            torch.cuda.synchronize()

            ref = reference_grouped_stage1(q, k_buf, v_buf, req_to_tokens, b_seqlen,
                                            num_splits, sm_scale, ps)

            if not torch.allclose(att_out, ref, atol=1e-2, rtol=1e-2):
                max_diff = (att_out - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (bs={bs}, nh={nh}, nkv={nkv}, hd={hd}, "
                    f"seq={max_seq}, splits={num_splits}, ps={ps}): "
                    f"max diff = {max_diff:.6f}"
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

    q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen, sm_scale = \
        make_inputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype)

    # Warmup
    for _ in range(5):
        att_out.zero_()
        mod.decode_grouped_att_m_fwd(
            q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen,
            num_splits, sm_scale, ps, logit_cap=0.0,
        )
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        att_out.zero_()
        start_events[j].record()
        mod.decode_grouped_att_m_fwd(
            q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen,
            num_splits, sm_scale, ps, logit_cap=0.0,
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
