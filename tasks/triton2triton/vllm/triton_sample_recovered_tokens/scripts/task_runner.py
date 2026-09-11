#!/usr/bin/env python3
"""Task runner for triton2triton/triton_sample_recovered_tokens"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_sample_recovered_tokens"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_sample_recovered_tokens.py")

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
        assert hasattr(mod, "sample_recovered_tokens"), "Missing sample_recovered_tokens"
        assert hasattr(mod, "sample_recovered_tokens_kernel"), "Missing sample_recovered_tokens_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 3, 64),   # (batch, max_draft, vocab)
    (8, 5, 128),
    (16, 4, 256),
    (32, 6, 512),
    (64, 8, 1024),
]
CORRECTNESS_EDGE_CASES = [
    (1, 1, 3, [1]),
    (5, 3, 511, [0, 1, 0, 3, 0]),
]
NUM_CORRECTNESS_CASES = len(TEST_SHAPES) + len(CORRECTNESS_EDGE_CASES) + 1
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

def reference_sample_recovered_tokens(
    cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs, q, vocab_size
):
    import torch
    batch_size = cu_num_draft_tokens.shape[0]
    out = torch.empty_like(draft_token_ids)
    start = 0
    for req in range(batch_size):
        end = cu_num_draft_tokens[req].item()
        for idx in range(start, end):
            if draft_probs is None:
                prob = target_probs[idx].clone()
                prob[draft_token_ids[idx]] = 0
            else:
                prob = torch.maximum(target_probs[idx] - draft_probs[idx], torch.zeros_like(target_probs[idx]))
            out[idx] = torch.argmax(prob / q[req]).to(out.dtype)
        start = end
    return out

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, max_draft, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            num_per_req = [(max_draft - (j % 2)) for j in range(batch_size)]
            cu = torch.cumsum(torch.tensor(num_per_req, dtype=torch.int32), dim=0).to(device)
            total = sum(num_per_req)
            draft_ids = torch.randint(0, vocab_size, (total,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()
            result = mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
            ref = reference_sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, vocab_size)
            if not torch.equal(result, ref):
                return False, f"Shape {i+1}: mismatch with draft_probs path"

            # Also test NO_DRAFT_PROBS path.
            result_no_draft = mod.sample_recovered_tokens(cu, draft_ids, None, target_probs, q, max_draft, vocab_size)
            ref_no_draft = reference_sample_recovered_tokens(cu, draft_ids, None, target_probs, q, vocab_size)
            if not torch.equal(result_no_draft, ref_no_draft):
                return False, f"Shape {i+1}: mismatch with NO_DRAFT_PROBS path"
            torch.cuda.synchronize()
            assert result.shape == (total,), f"Wrong shape: {result.shape}"
            assert result.min() >= 0 and result.max() < vocab_size
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for i, (batch_size, max_draft, vocab_size, num_per_req) in enumerate(
        CORRECTNESS_EDGE_CASES
    ):
        case_name = f"Edge shape {i+1}"
        try:
            torch.manual_seed(100 + i)
            cu = torch.cumsum(
                torch.tensor(num_per_req, dtype=torch.int32, device=device), dim=0
            )
            total = sum(num_per_req)
            draft_ids = torch.randint(
                0, vocab_size, (total,), dtype=torch.int32, device=device
            )
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()

            result = mod.sample_recovered_tokens(
                cu, draft_ids, draft_probs, target_probs, q,
                max_draft, vocab_size,
            )
            ref = reference_sample_recovered_tokens(
                cu, draft_ids, draft_probs, target_probs, q, vocab_size
            )
            if not torch.equal(result, ref):
                return False, f"{case_name}: mismatch with draft_probs path"

            result_no_draft = mod.sample_recovered_tokens(
                cu, draft_ids, None, target_probs, q, max_draft, vocab_size
            )
            ref_no_draft = reference_sample_recovered_tokens(
                cu, draft_ids, None, target_probs, q, vocab_size
            )
            if not torch.equal(result_no_draft, ref_no_draft):
                return False, f"{case_name}: mismatch with NO_DRAFT_PROBS path"
            torch.cuda.synchronize()
            assert result.shape == (total,), f"Wrong shape: {result.shape}"
            assert result.min() >= 0 and result.max() < vocab_size
        except Exception as e:
            return False, f"{case_name}: exception: {e}"

    # Use sparse, normalized distributions to make the expected argmax exact.
    # Row 0 ties at tokens 1 and 3, row 1 has no positive adjusted probability,
    # and row 2 selects the final valid token before power-of-two padding.
    try:
        batch_size, max_draft, vocab_size = 3, 2, 513
        cu = torch.tensor([1, 1, 3], dtype=torch.int32, device=device)
        draft_ids = torch.tensor([0, 5, 0], dtype=torch.int32, device=device)
        draft_probs = torch.zeros(3, vocab_size, device=device)
        target_probs = torch.zeros_like(draft_probs)
        draft_probs[0, 0] = 1.0
        target_probs[0, 1] = 0.5
        target_probs[0, 3] = 0.5
        draft_probs[1, 5] = 1.0
        target_probs[1, 5] = 1.0
        draft_probs[2, 0] = 1.0
        target_probs[2, vocab_size - 1] = 1.0
        q = torch.ones(batch_size, vocab_size, device=device)
        expected = torch.tensor([1, 0, vocab_size - 1], dtype=torch.int32, device=device)

        ref = reference_sample_recovered_tokens(
            cu, draft_ids, draft_probs, target_probs, q, vocab_size
        )
        ref_no_draft = reference_sample_recovered_tokens(
            cu, draft_ids, None, target_probs, q, vocab_size
        )
        if not torch.equal(ref, expected) or not torch.equal(ref_no_draft, expected):
            return False, "Tie/padding case: reference did not produce expected tokens"

        result = mod.sample_recovered_tokens(
            cu, draft_ids, draft_probs, target_probs, q,
            max_draft, vocab_size,
        )
        if not torch.equal(result, expected):
            return False, "Tie/padding case: mismatch with draft_probs path"
        result_no_draft = mod.sample_recovered_tokens(
            cu, draft_ids, None, target_probs, q, max_draft, vocab_size
        )
        if not torch.equal(result_no_draft, expected):
            return False, "Tie/padding case: mismatch with NO_DRAFT_PROBS path"
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"Tie/padding case: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch_size, max_draft, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            num_per_req = [max_draft] * batch_size
            cu = torch.cumsum(torch.tensor(num_per_req, dtype=torch.int32), dim=0).to(device)
            total = sum(num_per_req)
            draft_ids = torch.randint(0, vocab_size, (total,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()
            def _bench_fn():
                mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
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
                    "batch": batch_size,
                    "max_draft": max_draft,
                    "vocab": vocab_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch_size,
                    "max_draft": max_draft,
                    "vocab": vocab_size
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": NUM_CORRECTNESS_CASES}
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
