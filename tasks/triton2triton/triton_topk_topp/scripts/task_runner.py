#!/usr/bin/env python3
"""Task runner for triton2triton/triton_topk_topp"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_topk_topp"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_topk_topp.py")

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
        assert hasattr(mod, "apply_top_k_top_p_triton"), "Missing apply_top_k_top_p_triton"
        assert hasattr(mod, "_topk_topp_kernel"), "Missing _topk_topp_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256),    # (batch, vocab)
    (8, 1024),
    (16, 4096),
    (32, 8192),
    (64, 16384),
]
PERF_SHAPE_IDX = 3


def reference_apply_top_k_top_p(logits, k, p):
    import torch
    out = logits.clone()
    batch, _ = out.shape
    for b in range(batch):
        row = out[b]
        if k is not None:
            kv = int(k[b].item())
            if kv < row.numel():
                topk_vals, _ = torch.topk(row, kv)
                kth = topk_vals[-1]
                row = torch.where(row >= kth, row, torch.tensor(float("-inf"), device=row.device, dtype=row.dtype))
        if p is not None:
            pv = float(p[b].item())
            if pv < 1.0:
                sorted_vals, sorted_idx = torch.sort(row, descending=True)
                probs = torch.softmax(sorted_vals, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum > pv
                remove[0] = False
                row[sorted_idx[remove]] = float("-inf")
        out[b] = row
    return out


def compare_masked_logits(got, ref, vocab_size, max_mask_mismatch):
    import torch

    got_mask = torch.isfinite(got)
    ref_mask = torch.isfinite(ref)

    for b in range(got.shape[0]):
        mismatch = (got_mask[b] ^ ref_mask[b]).sum().item()
        if mismatch > max_mask_mismatch:
            return False, f"row {b}: mask mismatch {mismatch} > {max_mask_mismatch}"

    common = got_mask & ref_mask
    if common.any():
        if not torch.allclose(got[common], ref[common], atol=1e-4, rtol=1e-4):
            max_diff = (got[common] - ref[common]).abs().max().item()
            return False, f"common finite values max diff={max_diff}"
    return True, None

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
            k = torch.full((batch_size,), min(50, vocab_size), dtype=torch.int32, device=device)
            p = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)

            logits_topk = logits.clone()
            ref_topk = reference_apply_top_k_top_p(logits.clone(), k, None)
            mod.apply_top_k_top_p_triton(logits_topk, k, None)
            torch.cuda.synchronize()
            ok, msg = compare_masked_logits(logits_topk, ref_topk, vocab_size, max_mask_mismatch=1)
            if not ok:
                return False, f"Shape {i+1}: top-k mismatch ({msg})"

            logits_topkp = logits.clone()
            ref_topkp = reference_apply_top_k_top_p(logits.clone(), k, p)
            mod.apply_top_k_top_p_triton(logits_topkp, k, p)
            torch.cuda.synchronize()
            # Pivot-based GPU implementation may differ slightly at boundary tokens.
            max_mismatch = max(4, vocab_size // 500)
            ok, msg = compare_masked_logits(logits_topkp, ref_topkp, vocab_size, max_mask_mismatch=max_mismatch)
            if not ok:
                return False, f"Shape {i+1}: top-k + top-p mismatch ({msg})"

            # Invariant: top-k + top-p should keep no more tokens than top-k only.
            kept_topk = torch.isfinite(logits_topk).sum(dim=-1)
            kept_topkp = torch.isfinite(logits_topkp).sum(dim=-1)
            if torch.any(kept_topkp > kept_topk):
                return False, f"Shape {i+1}: top-k+topp kept more tokens than top-k"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch_size, vocab_size = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    for _ in range(5):
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
        k = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
        mod.apply_top_k_top_p_triton(logits, k, None)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
        k = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
        start_events[j].record()
        mod.apply_top_k_top_p_triton(logits, k, None)
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
