#!/usr/bin/env python3
"""Task runner for triton2triton/triton_ranks"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_ranks"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ranks.py")

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
        assert hasattr(mod, "compute_ranks"), "Missing compute_ranks"
        assert hasattr(mod, "_ranks_kernel"), "Missing _ranks_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256),
    (8, 1024),
    (16, 4096),
    (32, 8192),
    (64, 32768),
]
ADDITIONAL_CORRECTNESS_CASES = [
    {
        "name": "masked_tail",
        "batch": 3,
        "vocab": 8193,
        "dtype": "float32",
        "pattern": "all_ties",
    },
    {
        "name": "above_32768",
        "batch": 2,
        "vocab": 40003,
        "dtype": "float32",
        "pattern": "high_tail",
    },
    {
        "name": "large_batch_small_vocab_fp16_ties",
        "batch": 65,
        "vocab": 257,
        "dtype": "float16",
        "pattern": "repeated_values",
    },
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

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, vocab) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            token_ids = torch.randint(0, vocab, (batch,), dtype=torch.int64, device=device)
            result = mod.compute_ranks(logits, token_ids)
            torch.cuda.synchronize()
            # CPU ref
            ref = torch.zeros(batch, dtype=torch.int64, device=device)
            for b in range(batch):
                x = logits[b, token_ids[b].item()]
                ref[b] = (logits[b] >= x).sum().item()
            if not torch.equal(result, ref):
                return False, f"Shape {i+1}: mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for i, case in enumerate(ADDITIONAL_CORRECTNESS_CASES):
        name = case["name"]
        batch = case["batch"]
        vocab = case["vocab"]
        dtype = getattr(torch, case["dtype"])
        try:
            if case["pattern"] == "all_ties":
                logits = torch.zeros(batch, vocab, device=device, dtype=dtype)
                token_ids = torch.full(
                    (batch,), vocab - 1, dtype=torch.int64, device=device
                )
            elif case["pattern"] == "high_tail":
                logits = torch.zeros(batch, vocab, device=device, dtype=dtype)
                logits[:, 32768:] = 1
                token_ids = torch.tensor(
                    [0, vocab - 1], dtype=torch.int64, device=device
                )
            else:
                values = (torch.arange(vocab, device=device) % 7 - 3).to(dtype)
                logits = values.unsqueeze(0).expand(batch, -1).contiguous()
                token_ids = (
                    torch.arange(batch, dtype=torch.int64, device=device) * 37
                    + vocab
                    - 1
                ) % vocab

            result = mod.compute_ranks(logits, token_ids)
            torch.cuda.synchronize()
            ref = torch.zeros(batch, dtype=torch.int64, device=device)
            for b in range(batch):
                x = logits[b, token_ids[b].item()]
                ref[b] = (logits[b] >= x).sum().item()
            if not torch.equal(result, ref):
                return False, f"Additional case {name}: mismatch"
        except Exception as e:
            return False, f"Additional case {name}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch, vocab) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            token_ids = torch.randint(0, vocab, (batch,), dtype=torch.int64, device=device)
            def _bench_fn():
                mod.compute_ranks(logits, token_ids)
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
                    "vocab": vocab
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "vocab": vocab
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
            "num_shapes": len(TEST_SHAPES) + len(ADDITIONAL_CORRECTNESS_CASES),
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
