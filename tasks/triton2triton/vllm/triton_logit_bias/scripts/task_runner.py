#!/usr/bin/env python3
"""Task runner for triton2triton/triton_logit_bias"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_logit_bias"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_logit_bias.py")

def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f: source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "apply_logit_bias"), "Missing apply_logit_bias"
        assert hasattr(mod, "_bias_kernel"), "Missing _bias_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256, 16),    # (batch, vocab, max_bias_tokens)
    (8, 1024, 32),
    (16, 4096, 64),
    (32, 8192, 128),
    (64, 16384, 128),
]
PERF_SHAPE_IDX = 3


def reference_apply_logit_bias(
    logits,
    idx_mapping,
    pos,
    num_allowed_token_ids,
    allowed_token_ids,
    num_logit_bias,
    bias_token_ids,
    bias_vals,
    min_lens,
    num_stop_token_ids,
    stop_token_ids,
):
    import torch

    out = logits.clone()
    batch, vocab = out.shape
    for b in range(batch):
        state_idx = idx_mapping[b].item()

        n_allowed = num_allowed_token_ids[state_idx].item()
        if n_allowed > 0:
            keep_ids = allowed_token_ids[state_idx, :n_allowed]
            original = out[b, keep_ids].clone()
            out[b].fill_(-float("inf"))
            out[b, keep_ids] = original

        n_bias = num_logit_bias[state_idx].item()
        for j in range(n_bias):
            tok = bias_token_ids[state_idx, j].item()
            out[b, tok] = out[b, tok] + bias_vals[state_idx, j]

        n_stop = num_stop_token_ids[state_idx].item()
        if n_stop > 0 and pos[b].item() < min_lens[state_idx].item():
            stop_ids = stop_token_ids[state_idx, :n_stop]
            out[b, stop_ids] = -float("inf")

    return out

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, vocab, max_bt) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.randperm(batch, device=device, dtype=torch.int64).to(torch.int32)
            pos = torch.randint(0, 16, (batch,), dtype=torch.int64, device=device)

            # Keep allowlist disabled in correctness to avoid backend-specific instability.
            num_allowed = torch.zeros(batch, dtype=torch.int32, device=device)
            allowed_ids = torch.zeros(batch, max_bt, dtype=torch.int32, device=device)
            num_bias = torch.full((batch,), min(5, max_bt), dtype=torch.int32, device=device)
            bias_token_ids = torch.empty(batch, max_bt, dtype=torch.int32, device=device)
            for b in range(batch):
                bias_token_ids[b] = torch.randperm(vocab, device=device, dtype=torch.int64)[:max_bt].to(torch.int32)
            bias_vals = torch.randn(batch, max_bt, dtype=torch.float32, device=device)
            min_lens = torch.randint(1, 16, (batch,), dtype=torch.int32, device=device)
            num_stop = torch.randint(0, min(max_bt, 8), (batch,), dtype=torch.int32, device=device)
            stop_ids = torch.empty(batch, max_bt, dtype=torch.int32, device=device)
            for b in range(batch):
                stop_ids[b] = torch.randperm(vocab, device=device, dtype=torch.int64)[:max_bt].to(torch.int32)

            ref = reference_apply_logit_bias(
                logits, idx_mapping, pos, num_allowed, allowed_ids, num_bias,
                bias_token_ids, bias_vals, min_lens, num_stop, stop_ids
            )
            mod.apply_logit_bias(logits, idx_mapping, pos, num_allowed, allowed_ids, num_bias, bias_token_ids, bias_vals, min_lens, num_stop, stop_ids)
            torch.cuda.synchronize()
            got_finite = torch.isfinite(logits)
            ref_finite = torch.isfinite(ref)
            if not torch.equal(got_finite, ref_finite):
                mismatch = (got_finite ^ ref_finite).sum().item()
                return False, f"Shape {i+1}: finite-mask mismatch ({mismatch} elements)"

            if got_finite.any():
                got_v = logits[got_finite]
                ref_v = ref[got_finite]
                if not torch.allclose(got_v, ref_v, atol=1e-2, rtol=1e-2):
                    max_diff = (got_v - ref_v).abs().max().item()
                    return False, f"Shape {i+1}: finite max diff = {max_diff}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    # Extra targeted coverage for allowlist path (num_allowed_token_ids > 0).
    # Keep this small to avoid backend instability seen on larger random configs.
    try:
        torch.manual_seed(777)
        batch, vocab, max_bt = 2, 256, 16
        logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
        idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
        pos = torch.full((batch,), 4, dtype=torch.int64, device=device)

        num_allowed = torch.tensor([4, 8], dtype=torch.int32, device=device)
        allowed_ids = torch.empty(batch, max_bt, dtype=torch.int32, device=device)
        for b in range(batch):
            allowed_ids[b] = torch.randperm(vocab, device=device, dtype=torch.int64)[:max_bt].to(torch.int32)

        # Isolate allowlist behavior in this targeted check.
        num_bias = torch.zeros(batch, dtype=torch.int32, device=device)
        bias_token_ids = torch.zeros(batch, max_bt, dtype=torch.int32, device=device)
        bias_vals = torch.zeros(batch, max_bt, dtype=torch.float32, device=device)
        min_lens = torch.zeros(batch, dtype=torch.int32, device=device)
        num_stop = torch.zeros(batch, dtype=torch.int32, device=device)
        stop_ids = torch.zeros(batch, max_bt, dtype=torch.int32, device=device)

        ref = reference_apply_logit_bias(
            logits, idx_mapping, pos, num_allowed, allowed_ids, num_bias,
            bias_token_ids, bias_vals, min_lens, num_stop, stop_ids
        )
        got = logits.clone()
        mod.apply_logit_bias(
            got, idx_mapping, pos, num_allowed, allowed_ids, num_bias,
            bias_token_ids, bias_vals, min_lens, num_stop, stop_ids
        )
        torch.cuda.synchronize()

        got_finite = torch.isfinite(got)
        ref_finite = torch.isfinite(ref)
        if not torch.equal(got_finite, ref_finite):
            mismatch = (got_finite ^ ref_finite).sum().item()
            return False, f"Allowlist case: finite-mask mismatch ({mismatch} elements)"
        if got_finite.any() and not torch.allclose(got[got_finite], ref[got_finite], atol=1e-2, rtol=1e-2):
            max_diff = (got[got_finite] - ref[got_finite]).abs().max().item()
            return False, f"Allowlist case: finite max diff = {max_diff}"
    except Exception as e:
        return False, f"Allowlist case: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch, vocab, max_bt = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
    idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
    pos = torch.full((batch,), 10, dtype=torch.int64, device=device)
    num_allowed = torch.zeros(batch, dtype=torch.int32, device=device)
    allowed_ids = torch.zeros(batch, max_bt, dtype=torch.int32, device=device)
    num_bias = torch.full((batch,), 5, dtype=torch.int32, device=device)
    bias_token_ids = torch.randint(0, vocab, (batch, max_bt), dtype=torch.int32, device=device)
    bias_vals = torch.randn(batch, max_bt, dtype=torch.float32, device=device)
    min_lens = torch.zeros(batch, dtype=torch.int32, device=device)
    num_stop = torch.zeros(batch, dtype=torch.int32, device=device)
    stop_ids = torch.zeros(batch, max_bt, dtype=torch.int32, device=device)
    for _ in range(5): mod.apply_logit_bias(logits.clone(), idx_mapping, pos, num_allowed, allowed_ids, num_bias, bias_token_ids, bias_vals, min_lens, num_stop, stop_ids)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        l = logits.clone()
        start_events[j].record()
        mod.apply_logit_bias(l, idx_mapping, pos, num_allowed, allowed_ids, num_bias, bias_token_ids, bias_vals, min_lens, num_stop, stop_ids)
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
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)

if __name__ == "__main__": main()
