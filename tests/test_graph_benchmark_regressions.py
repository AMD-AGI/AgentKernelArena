from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hip2hip_gpumode_restores_state_and_uses_reference_method_policy():
    harnesses = sorted(
        (ROOT / "tasks/hip2hip/gpumode").glob("*/eval_tools/cal_kernel_perf.py")
    )
    assert harnesses
    for harness in harnesses:
        source = harness.read_text()
        assert "ref_check_pristine" in source, harness
        assert "opt_check_pristine" in source, harness
        assert "reference_mutates_inputs" in source, harness
        assert "inputs_mutated" in source, harness
        assert "prepare_fn=ref_prepare" in source, harness
        assert "prepare_fn=opt_prepare" in source, harness
        assert "graph_enabled = ref_graph_enabled" in source, harness
        assert "ref_graph_enabled and opt_graph_enabled" not in source, harness


def test_geak_method_mismatch_is_never_aggregated_as_one_x():
    harnesses = [
        ROOT / "tasks/triton2triton/geak_eval/L1/llama_ff_triton/test_kernel_harness.py",
        ROOT / "tasks/triton2triton/geak_eval/L1/moe_routing_sigmoid_top1/test_kernel_harness.py",
        ROOT / "tasks/triton2triton/geak_eval/L2/fast_rms_layernorm/test_kernel_harness.py",
        ROOT / "tasks/triton2triton/geak_eval/L2/topk/test_kernel_harness.py",
        ROOT / "tasks/triton2triton/geak_eval/L3/fused_qkv_rope/test_kernel_harness.py",
        ROOT / "tasks/triton2triton/geak_eval/L3/fused_rms_fp8/test_kernel_harness.py",
    ]
    for harness in harnesses:
        source = harness.read_text()
        assert "else 1.0" not in source, harness
        assert "GEAK_BENCHMARK_METHOD_CONSISTENT" in source, harness
        assert "if speedup is not None" in source, harness


def test_native_drivers_use_nonzero_inputs_and_validate_an_actual_replay():
    matrix = (
        ROOT
        / "tasks/hip2hip/others/matrix_multiplication/scripts/native/benchmark_driver.hip"
    ).read_text()
    mla = (
        ROOT / "tasks/hip2hip/others/mla_decode/scripts/native/benchmark_driver.hip"
    ).read_text()
    helper = (ROOT / "src/tools/perf/native_hip_graph_benchmark.hpp").read_text()

    assert "hipMemcpyAsync(d_a, h_a.data()" in matrix
    assert "matrix replay validation" in matrix
    assert "hipMemcpyAsync(d_q, h_q.data()" in mla
    assert "MLA replay validation" in mla
    assert "graph replay output validation failed" in helper
    assert "validate(graph_exec, graph_stream)" in helper


def test_state_reset_and_scratch_are_outside_timed_workload():
    lora = (
        ROOT / "tasks/triton2triton/vllm/triton_lora_shrink/scripts/task_runner.py"
    ).read_text()
    bench_body = lora.split("def _bench_fn():", 1)[1].split(
        "elapsed_ms, benchmark_metadata", 1
    )[0]
    assert "output_tensor.zero_()" not in bench_body
    assert "prepare_fn=output_tensor.zero_" in lora

    roiaware_cpp = (
        ROOT / "tasks/hip2hip/others/roiaware_pool3d/src/roiaware_pool3d.cpp"
    ).read_text()
    roipoint_cpp = (
        ROOT / "tasks/hip2hip/others/roipoint_pool3d/src/roipoint_pool3d.cpp"
    ).read_text()
    assert "at::empty" not in roiaware_cpp
    assert "at::empty" not in roipoint_cpp
    assert "at::Tensor pts_mask" in roiaware_cpp
    assert "at::Tensor pts_assign" in roipoint_cpp


def test_flydsl_topk_fallback_matches_allocating_output_contract():
    harnesses = sorted(
        ROOT.glob("tasks/torch2flydsl/moe_topk_*_kernel/test_kernel_harness.py")
    )
    harnesses.append(
        ROOT
        / "tasks/torch2flydsl/moe_biased_grouped_topk_kernel/test_kernel_harness.py"
    )
    assert harnesses
    for harness in harnesses:
        source = harness.read_text()
        run_fused = source.split("def run_fused():", 2)[-1]
        assert "fused_w = torch.empty(" in run_fused, harness
        assert "fused_idx = torch.empty(" in run_fused, harness
