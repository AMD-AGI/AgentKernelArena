#!/usr/bin/env python3
"""Auto-generated task runner for triton__gemm_a16_w16_kernel (Triton).

Inputs are generated each run from the shape/dtype signatures in
test_cases.json. The launcher symbol expected in source/triton__gemm_a16_w16_kernel.py is
``_gemm_a16_w16_kernel`` (for raw @triton.jit kernels) or ``None`` (a
wrapper that handles grid + meta resolution if present).
"""
import sys, os, json, argparse, glob, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/campaign20/gemm_a16_w16_kernel"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton__gemm_a16_w16_kernel.py")
KERNEL_NAME = "_gemm_a16_w16_kernel"
LAUNCHER_NAME = "None"
REF_SOURCE = "triton"
TEST_CASES = os.path.join(TASK_DIR, "test_cases.json")


def _load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _test_cases():
    if not os.path.isfile(TEST_CASES):
        return []
    with open(TEST_CASES) as f:
        return json.load(f)


def _resolve_callable(mod):
    # Prefer a wrapping launcher if it was found alongside the @triton.jit kernel.
    for n in (LAUNCHER_NAME, KERNEL_NAME):
        if n and hasattr(mod, n) and n != "None":
            return getattr(mod, n)
    raise AttributeError(f"neither {LAUNCHER_NAME} nor {KERNEL_NAME} found in source")


def run_compile():
    try:
        import ast
        ast.parse(open(SOURCE_FILE).read())
        mod = _load_module()
        _resolve_callable(mod)
        import harness_run as hr
        hr.compile_smoke()
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    """Golden-vs-editable correctness via the real launcher harness."""
    import harness_run as hr
    hr.write_test_cases()
    return hr.run_correctness()


def run_performance():
    """Canonical CUDA-graph timing with event fallback."""
    import harness_run as hr
    hr.write_test_cases()
    return hr.run_performance(_benchmark_cuda_graph_or_events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = ap.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err}, open(os.path.join(build_dir, "compile_report.json"), "w"))
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print("Error:", err)
        sys.exit(0 if ok else 1)
    if args.mode == "correctness":
        ok, err = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err}, open(os.path.join(build_dir, "correctness_report.json"), "w"))
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print("Error:", err)
        sys.exit(0 if ok else 1)
    cases = run_performance()
    json.dump({"test_cases": cases}, open(os.path.join(build_dir, "performance_report.json"), "w"), indent=2)
    for c in cases:
        print(f"Performance: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
    valid = bool(cases) and all(c.get("execution_time_ms", -1) > 0 for c in cases)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
