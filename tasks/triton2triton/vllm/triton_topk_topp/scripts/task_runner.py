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


def prepare_direct_launch(mod, logits, k, p, mask_value=float("-inf")):
    """Prepare a stable direct kernel launch with no timed host setup."""
    import torch

    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    assert logits.is_cuda

    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None
    assert batch_size > 0 and (topk_enabled or topp_enabled)

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size and k.is_cuda
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size and p.is_cuda
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits

    num_sm = torch.cuda.get_device_properties(logits.device).multi_processor_count
    num_programs = min(num_sm, batch_size)
    buffer_rows = min(mod._next_power_of_2(num_programs), num_sm)
    buffer = logits.new_empty((buffer_rows, vocab_size))
    if buffer_rows > num_programs:
        buffer = buffer[:num_programs]

    normal_cdf_to_sigma_table = torch.tensor(
        mod._NORMAL_CDF_TO_SIGMA_TABLE,
        dtype=torch.float32,
        device=logits.device,
    )
    percentile_to_std_table = torch.tensor(
        mod._PERCENTILE_TO_STD_TABLE,
        dtype=torch.float32,
        device=logits.device,
    )

    grid = (num_programs,)
    kernel_args = (
        logits,
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
    )
    kernel_meta = {
        "BATCH_SIZE": batch_size,
        "MASK_VALUE": mask_value,
        "VOCAB_SIZE": vocab_size,
        "BLOCK_SIZE": 8192,
        "BLOCK_SIZE_TRUNC": 4096,
        "TOPK_ENABLED": topk_enabled,
        "TOPP_ENABLED": topp_enabled,
    }
    kernel_launcher = mod._topk_topp_kernel[grid]

    def launch():
        kernel_launcher(*kernel_args, **kernel_meta)

    return {
        "launch": launch,
        "buffer": buffer,
        "normal_cdf_to_sigma_table": normal_cdf_to_sigma_table,
        "percentile_to_std_table": percentile_to_std_table,
        "k_ptr": k_ptr,
        "p_ptr": p_ptr,
    }

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

            direct_logits_topk = logits.clone()
            direct_topk = prepare_direct_launch(
                mod, direct_logits_topk, k, None,
            )
            direct_topk["launch"]()
            torch.cuda.synchronize()
            ok, msg = compare_masked_logits(
                direct_logits_topk, ref_topk, vocab_size, max_mask_mismatch=1,
            )
            if not ok:
                return False, f"Shape {i+1}: direct top-k mismatch ({msg})"

            logits_topkp = logits.clone()
            ref_topkp = reference_apply_top_k_top_p(logits.clone(), k, p)
            mod.apply_top_k_top_p_triton(logits_topkp, k, p)
            torch.cuda.synchronize()
            # Pivot-based GPU implementation may differ slightly at boundary tokens.
            max_mismatch = max(4, vocab_size // 500)
            ok, msg = compare_masked_logits(logits_topkp, ref_topkp, vocab_size, max_mask_mismatch=max_mismatch)
            if not ok:
                return False, f"Shape {i+1}: top-k + top-p mismatch ({msg})"

            direct_logits_topkp = logits.clone()
            direct_topkp = prepare_direct_launch(
                mod, direct_logits_topkp, k, p,
            )
            direct_topkp["launch"]()
            torch.cuda.synchronize()
            ok, msg = compare_masked_logits(
                direct_logits_topkp,
                ref_topkp,
                vocab_size,
                max_mask_mismatch=max_mismatch,
            )
            if not ok:
                return False, f"Shape {i+1}: direct top-k + top-p mismatch ({msg})"

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
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (batch_size, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            base_logits = torch.randn(
                batch_size, vocab_size, device=device, dtype=torch.float32,
            )
            logits = base_logits.clone()
            k = torch.full(
                (batch_size,), 50, dtype=torch.int32, device=device,
            )
            direct = prepare_direct_launch(mod, logits, k, None)

            def _prepare_logits():
                # The helper invokes preparation on the active benchmark stream
                # before its start event/replay, so this copy is not timed.
                logits.copy_(base_logits)

            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                direct["launch"],
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                prepare_fn=_prepare_logits,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "batch_size": batch_size,
                    "vocab_size": vocab_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch_size": batch_size,
                    "vocab_size": vocab_size
                }
            })

    return test_cases


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
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

if __name__ == "__main__": main()
