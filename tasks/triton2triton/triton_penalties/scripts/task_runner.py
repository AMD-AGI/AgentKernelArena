#!/usr/bin/env python3
"""Task runner for triton2triton/triton_penalties"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_penalties"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_penalties.py")

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
        assert hasattr(mod, "apply_penalties"), "Missing apply_penalties"
        assert hasattr(mod, "_penalties_kernel"), "Missing _penalties_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256),
    (8, 1024),
    (16, 8192),
    (32, 16384),
    (64, 32768),
]
PERF_SHAPE_IDX = 3


def unpack_prompt_mask(packed_mask_row, vocab_size):
    import torch
    out = torch.zeros(vocab_size, dtype=torch.bool, device=packed_mask_row.device)
    for tok in range(vocab_size):
        out[tok] = ((packed_mask_row[tok // 32] >> (tok % 32)) & 1) != 0
    return out


def reference_apply_penalties(
    logits,
    idx_mapping,
    token_ids,
    local_pos,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    prompt_bin_mask,
    output_bin_counts,
    num_speculative_tokens,
):
    import torch

    out = logits.clone().float()
    num_tokens, vocab_size = out.shape
    for token_idx in range(num_tokens):
        state_idx = idx_mapping[token_idx].item()
        rep = repetition_penalty[state_idx].item()
        freq = frequency_penalty[state_idx].item()
        pres = presence_penalty[state_idx].item()
        if rep == 1.0 and freq == 0.0 and pres == 0.0:
            continue

        counts = output_bin_counts[state_idx].to(torch.int32).clone()
        if num_speculative_tokens > 0:
            pos = local_pos[token_idx].item()
            start_idx = token_idx - pos
            for prev_pos in range(pos):
                prev_token = token_ids[start_idx + prev_pos + 1].item()
                counts[prev_token] += 1

        output_mask = counts > 0
        prompt_mask = unpack_prompt_mask(prompt_bin_mask[state_idx], vocab_size)
        if rep != 1.0:
            scale_mask = prompt_mask | output_mask
            scale = torch.where(scale_mask, torch.tensor(rep, device=out.device), torch.tensor(1.0, device=out.device))
            out[token_idx] = torch.where(out[token_idx] > 0, out[token_idx] / scale, out[token_idx] * scale)
        out[token_idx] -= freq * counts.float()
        out[token_idx] -= pres * output_mask.float()

    return out.to(logits.dtype)

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, vocab) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            token_ids = torch.randint(0, vocab, (batch,), dtype=torch.int32, device=device)
            local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
            rep_pen = torch.full((batch,), 1.2, device=device)
            freq_pen = torch.full((batch,), 0.5, device=device)
            pres_pen = torch.full((batch,), 0.3, device=device)
            prompt_mask = torch.zeros(batch, (vocab + 31) // 32, dtype=torch.int32, device=device)
            output_counts = torch.randint(0, 4, (batch, vocab), dtype=torch.int32, device=device)
            # Set prompt mask bits to exercise repetition penalty path.
            for b in range(batch):
                prompt_mask[b, 0] = 0b10101
            ref = reference_apply_penalties(
                logits, idx_mapping, token_ids, local_pos, rep_pen, freq_pen,
                pres_pen, prompt_mask, output_counts, 0
            )
            mod.apply_penalties(logits, idx_mapping, token_ids, local_pos, rep_pen, freq_pen, pres_pen, prompt_mask, output_counts, 0)
            torch.cuda.synchronize()
            if not torch.allclose(logits, ref, atol=1e-2, rtol=1e-2):
                return False, f"Shape {i+1}: max diff = {(logits - ref).abs().max().item()}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch, vocab = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
    idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
    token_ids = torch.zeros(batch, dtype=torch.int32, device=device)
    local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
    rep_pen = torch.full((batch,), 1.2, device=device)
    freq_pen = torch.full((batch,), 0.5, device=device)
    pres_pen = torch.full((batch,), 0.3, device=device)
    prompt_mask = torch.zeros(batch, (vocab + 31) // 32, dtype=torch.int32, device=device)
    output_counts = torch.randint(0, 5, (batch, vocab), dtype=torch.int32, device=device)
    for _ in range(5): mod.apply_penalties(logits.clone(), idx_mapping, token_ids, local_pos, rep_pen, freq_pen, pres_pen, prompt_mask, output_counts, 0)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        l = logits.clone()
        start_events[j].record()
        mod.apply_penalties(l, idx_mapping, token_ids, local_pos, rep_pen, freq_pen, pres_pen, prompt_mask, output_counts, 0)
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
