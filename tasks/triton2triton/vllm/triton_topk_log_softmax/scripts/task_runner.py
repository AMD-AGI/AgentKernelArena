#!/usr/bin/env python3
"""Task runner for triton2triton/triton_topk_log_softmax"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_topk_log_softmax"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_topk_log_softmax.py")

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
        assert hasattr(mod, "compute_token_logprobs"), "Missing compute_token_logprobs"
        assert hasattr(mod, "_topk_log_softmax_kernel"), "Missing _topk_log_softmax_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256, 3),   # (batch, vocab, num_tokens)
    (8, 1024, 5),
    (16, 4096, 10),
    (32, 8192, 20),
    (64, 32768, 10),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Non-scoring public-contract coverage. Keep TEST_SHAPES reserved for performance.
CORRECTNESS_CASES = [
    {
        "name": "padded_stride_fp16_nonpower_duplicates",
        "batch": 3,
        "vocab": 1003,
        "num_tokens": 7,
        "logits_dtype": "float16",
        "token_ids_dtype": "int64",
        "row_padding": 29,
    },
    {
        "name": "large_vocab_batch1_bfloat16_int32_duplicates",
        "batch": 1,
        "vocab": 131071,
        "num_tokens": 8,
        "logits_dtype": "bfloat16",
        "token_ids_dtype": "int32",
        "row_padding": 17,
    },
]


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


def _make_topk_correctness_inputs(torch, case, device):
    batch = case["batch"]
    vocab = case["vocab"]
    num_tokens = case["num_tokens"]
    row_padding = case["row_padding"]
    logits_dtype = getattr(torch, case["logits_dtype"])
    token_ids_dtype = getattr(torch, case["token_ids_dtype"])

    logits_storage = torch.randn(
        batch, vocab + row_padding, device=device, dtype=logits_dtype
    )
    logits = logits_storage[:, :vocab]
    if batch > 1:
        assert not logits.is_contiguous()
    assert logits.stride(0) == vocab + row_padding

    duplicate_unsorted_ids = [
        vocab - 1,
        3,
        vocab // 2,
        3,
        0,
        vocab - 7,
        11,
        0,
    ]
    token_ids = torch.tensor(
        duplicate_unsorted_ids[:num_tokens],
        device=device,
        dtype=token_ids_dtype,
    ).repeat(batch, 1)
    return logits, token_ids, logits_storage

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, vocab, ntok) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            token_ids = torch.randint(0, vocab, (batch, ntok), dtype=torch.int64, device=device)
            result = mod.compute_token_logprobs(logits, token_ids)
            torch.cuda.synchronize()
            # CPU ref: log_softmax then gather
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            ref = log_probs.gather(1, token_ids)
            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                return False, f"Shape {i+1}: max diff = {(result - ref).abs().max().item()}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for i, case in enumerate(CORRECTNESS_CASES):
        name = case["name"]
        try:
            torch.manual_seed(142 + i)
            logits, token_ids, logits_storage = _make_topk_correctness_inputs(
                torch, case, device
            )
            original_logits = logits.clone()
            original_token_ids = token_ids.clone()
            original_logits_storage = logits_storage.clone()

            result = mod.compute_token_logprobs(logits, token_ids)
            torch.cuda.synchronize()
            ref = torch.log_softmax(logits.float(), dim=-1).gather(
                1, token_ids.to(torch.int64)
            )

            expected_shape = (case["batch"], case["num_tokens"])
            if result.shape != expected_shape or result.dtype != torch.float32:
                return False, (
                    f"Contract case {name}: expected shape/dtype "
                    f"{expected_shape}/{torch.float32}, got {result.shape}/{result.dtype}"
                )
            result_storage = result.untyped_storage().data_ptr()
            if result_storage in {
                logits.untyped_storage().data_ptr(),
                token_ids.untyped_storage().data_ptr(),
            }:
                return False, f"Contract case {name}: output aliases an input"
            if not torch.equal(logits, original_logits) or not torch.equal(
                token_ids, original_token_ids
            ):
                return False, f"Contract case {name}: input mutation"
            if not torch.equal(logits_storage, original_logits_storage):
                return False, (
                    f"Contract case {name}: write outside logical logits view"
                )
            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                max_diff = (result - ref).abs().max().item()
                return False, f"Contract case {name}: max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"Contract case {name}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch, vocab, ntok) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            token_ids = torch.randint(0, vocab, (batch, ntok), dtype=torch.int64, device=device)
            def _bench_fn():
                mod.compute_token_logprobs(logits, token_ids)
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "batch": batch,
                    "vocab": vocab,
                    "num_tokens": ntok
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "vocab": vocab,
                    "num_tokens": ntok
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
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES) + len(CORRECTNESS_CASES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f: json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

if __name__ == "__main__": main()
