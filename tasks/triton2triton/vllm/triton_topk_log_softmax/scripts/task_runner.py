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
CORRECTNESS_CASES = [
    {
        "name": f"random_{i + 1}",
        "shape": shape,
        "dtype": "float32",
        "values": "random",
    }
    for i, shape in enumerate(TEST_SHAPES)
] + [
    {
        "name": "singleton_vocab",
        "shape": (1, 1, 1),
        "dtype": "float32",
        "values": "singleton",
    },
    {
        "name": "non_power_of_two_below_block",
        "shape": (3, 1000, 7),
        "dtype": "float32",
        "values": "random",
    },
    {
        "name": "non_power_of_two_above_block_fp16_edges",
        "shape": (2, 1025, 5),
        "dtype": "float16",
        "values": "numerical_edges",
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

def _make_correctness_inputs(torch, case, seed, device):
    batch, vocab, ntok = case["shape"]
    dtype = getattr(torch, case["dtype"])
    torch.manual_seed(seed)

    if case["values"] == "random":
        logits = torch.randn(batch, vocab, device=device, dtype=dtype)
        token_ids = torch.randint(
            0, vocab, (batch, ntok), dtype=torch.int64, device=device
        )
    elif case["values"] == "singleton":
        logits = torch.tensor([[12345.0]], device=device, dtype=dtype)
        token_ids = torch.tensor([[0]], device=device, dtype=torch.int64)
    elif case["values"] == "numerical_edges":
        logits = torch.empty((batch, vocab), device=device, dtype=dtype)
        logits[0].fill_(-80.0)
        logits[0, 0] = 80.0
        logits[0, vocab // 2] = 79.5
        logits[0, -1] = 80.0
        logits[1].fill_(10000.0)
        token_ids = torch.tensor(
            [
                [0, 1, vocab // 2, vocab - 2, vocab - 1],
                [0, 1, vocab // 2, vocab - 2, vocab - 1],
            ],
            device=device,
            dtype=torch.int64,
        )
    else:
        raise ValueError(f"Unknown correctness value pattern: {case['values']}")

    return logits, token_ids

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, case in enumerate(CORRECTNESS_CASES):
        try:
            logits, token_ids = _make_correctness_inputs(
                torch, case, seed=42 + i, device=device
            )
            result = mod.compute_token_logprobs(logits, token_ids)
            torch.cuda.synchronize()
            # CPU ref: log_softmax then gather
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            ref = log_probs.gather(1, token_ids)
            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                return False, (
                    f"Case {i+1} ({case['name']}): "
                    f"max diff = {(result - ref).abs().max().item()}"
                )
        except Exception as e:
            return False, f"Case {i+1} ({case['name']}): exception: {e}"
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(CORRECTNESS_CASES)}
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
