#!/usr/bin/env python3
"""Task runner for triton2triton/triton_batched_moe"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_batched_moe"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_batched_moe.py")

# (E, max_tokens, K, N)
TEST_SHAPES = [
    (4, 16, 64, 64),
    (8, 32, 128, 128),
    (8, 64, 256, 256),
    (16, 64, 512, 512),
    (8, 128, 1024, 512),
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

sys.path.insert(0, TASK_DIR)
from _contract_checks import checked_call, checked_benchmark, compare_output, perturb_activation


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "batched_moe_gemm"), "Missing batched_moe_gemm"
        assert hasattr(mod, "batched_triton_kernel"), "Missing batched_triton_kernel"
        return True, None
    except Exception as e:
        return False, str(e)



CONTROL_CASES = ('zero_experts', 'ragged_tail')


def reference(inputs):
    import torch
    A, B, counts = inputs['A'], inputs['B'], inputs['counts']
    expected = torch.zeros((A.shape[0], A.shape[1], B.shape[1]), dtype=A.dtype, device=A.device)
    for e in range(A.shape[0]):
        n = int(counts[e].item())
        if n:
            expected[e, :n] = (A[e, :n].float() @ B[e].float().T).to(A.dtype)
    return expected


def control_inputs(name, device):
    import torch
    E, M, K, N = 4, 7, 35, 70
    A = ((torch.arange(E*M*K, device=device).reshape(E,M,K) % 7)-3).to(torch.float16)
    B = ((torch.arange(E*N*K, device=device).reshape(E,N,K) % 5)-2).to(torch.float16)
    counts = torch.tensor([0,0,0,0] if name=='zero_experts' else [0,1,6,7], device=device, dtype=torch.int32)
    return {'A':A, 'B':B, 'counts':counts}


def invoke(mod, inputs):
    return mod.batched_moe_gemm(inputs['A'], inputs['B'], inputs['counts'])


def run_correctness(*, case_index=None, control=None):
    import torch
    try:
        mod = load_module()
        device = 'cuda'
        if control is not None:
            assert control in CONTROL_CASES, 'Unknown control'
            inputs = control_inputs(control, device)
            checked_call(lambda: invoke(mod, inputs), inputs=inputs, reference=reference, check=compare_output)
            return True, None
        for i, (E, max_tokens, K, N) in enumerate(TEST_SHAPES):
            if case_index is not None and i != case_index:
                continue
            torch.manual_seed(42 + i)
            A = torch.randn(E, max_tokens, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            counts = torch.randint(1, max_tokens + 1, (E,), device=device, dtype=torch.int32)
            inputs = {'A': A, 'B': B, 'counts': counts}
            checked_call(lambda: invoke(mod, inputs), inputs=inputs, reference=reference, check=compare_output)
        return True, None
    except Exception as exc:
        return False, exc


def run_performance():
    import torch
    mod = load_module()
    device = 'cuda'
    test_cases = []
    for test_idx, (E, max_tokens, K, N) in enumerate(TEST_SHAPES):
        row = {'test_case_id': f'perf{test_idx+1}', 'params': {'E': E, 'max_tokens': max_tokens, 'K': K, 'N': N}}
        try:
            torch.manual_seed(42 + test_idx)
            A = torch.randn(E, max_tokens, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            counts = torch.full((E,), max_tokens, device=device, dtype=torch.int32)
            inputs = {'A': A, 'B': B, 'counts': counts}
            elapsed_ms, metadata = checked_benchmark(
                _benchmark_cuda_graph_or_events, lambda: invoke(mod, inputs),
                inputs=inputs, reference=reference, check=compare_output,
                perturb=perturb_activation, warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS)
            row.update(execution_time_ms=elapsed_ms, **metadata)
        except Exception as exc:
            row.update(execution_time_ms=-1.0, error=f'{type(exc).__name__}: {exc}',
                       failure_kind=getattr(exc, 'failure_kind', 'measurement_failure'))
        test_cases.append(row)
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": str(err) if err else None}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": str(err) if err else None, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
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


if __name__ == "__main__":
    main()
