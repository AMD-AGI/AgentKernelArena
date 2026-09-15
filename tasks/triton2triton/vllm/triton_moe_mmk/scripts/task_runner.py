#!/usr/bin/env python3
"""Task runner for triton2triton/triton_moe_mmk"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_moe_mmk"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_moe_mmk.py")

# (M, K, N)
TEST_SHAPES = [
    (32, 64, 32),
    (64, 128, 64),
    (128, 256, 128),
    (256, 512, 256),
    (512, 1024, 512),
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
        assert hasattr(mod, "moe_matmul"), "Missing moe_matmul"
        assert hasattr(mod, "moe_mmk"), "Missing moe_mmk"
        return True, None
    except Exception as e:
        return False, str(e)



CONTROL_CASES = ('rectangular_tail',)


def reference(inputs):
    return (inputs['A'].float() @ inputs['B'].float()).to(inputs['A'].dtype)


def control_inputs(name, device):
    import torch
    A = ((torch.arange(17*35, device=device).reshape(17,35) % 7)-3).to(torch.float16)
    B = ((torch.arange(35*70, device=device).reshape(35,70) % 5)-2).to(torch.float16)
    return {'A':A, 'B':B}


def invoke(mod, inputs):
    return mod.moe_matmul(inputs['A'], inputs['B'])


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
        for i, (M, K, N) in enumerate(TEST_SHAPES):
            if case_index is not None and i != case_index:
                continue
            torch.manual_seed(42 + i)
            A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
            inputs = {'A': A, 'B': B}
            checked_call(lambda: invoke(mod, inputs), inputs=inputs, reference=reference, check=compare_output)
        return True, None
    except Exception as exc:
        return False, exc


def run_performance():
    import torch
    mod = load_module()
    device = 'cuda'
    test_cases = []
    for test_idx, (M, K, N) in enumerate(TEST_SHAPES):
        row = {'test_case_id': f'perf{test_idx+1}', 'params': {'M': M, 'K': K, 'N': N}}
        try:
            torch.manual_seed(42 + test_idx)
            A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
            inputs = {'A': A, 'B': B}
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
