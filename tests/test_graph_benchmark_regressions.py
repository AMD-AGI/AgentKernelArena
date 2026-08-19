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


def test_hipblaslt_starter_baselines_are_predetermined_event_only():
    tasks = [
        "batched_gemm_a8w8_kernel",
        "batched_gemm_bf16_kernel",
        "gemm_a16w8_blockscale_kernel",
        "gemm_a16wfp4_kernel",
        "gemm_a4w4_kernel",
        "gemm_a8w8_blockscale_kernel",
        "gemm_a8w8_kernel",
        "gemm_a8w8_per_token_scale_kernel",
        "gemm_a8wfp4_kernel",
        "gemm_afp4wfp4_kernel",
        "gemm_afp8wfp8_kernel",
    ]
    for task in tasks:
        harness = ROOT / "tasks/torch2flydsl" / task / "test_kernel_harness.py"
        source = harness.read_text()
        assert "use_graph = has_kernel" in source, harness
        assert "capture_unsafe_aiter_hipblaslt" in source, harness


def test_implemented_gemm_and_hipblaslt_reference_share_event_policy():
    tasks = ["gemm_a8w8_bpreshuffle_kernel", "hgemm_kernel"]
    for task in tasks:
        harness = ROOT / "tasks/torch2flydsl" / task / "test_kernel_harness.py"
        source = harness.read_text()
        assert source.count("use_cuda_graph=False") == 2, harness
        assert "capture_unsafe_hipblaslt_reference" in source, harness


def test_torch2flydsl_gfx950_configs_use_platform_support():
    tasks = [
        "gemm_a16wfp4_kernel",
        "gemm_a4w4_kernel",
        "gemm_a8wfp4_kernel",
        "gemm_afp4wfp4_kernel",
        "gemm_afp8wfp8_kernel",
        "quant_mxfp4_kernel",
    ]
    for task in tasks:
        config = (ROOT / "tasks/torch2flydsl" / task / "config.yaml").read_text()
        assert "supported_archs:" not in config, task
        assert "platform_support:" in config, task
        assert "required_arch: gfx950" in config, task
        assert "status: active" in config, task

    for task in ["fav3_sage_mxfp4", "gemm_afp8wfp8"]:
        config = (
            ROOT / "tasks/triton2flydsl/aiter" / task / "config.yaml"
        ).read_text()
        assert "supported_archs:" not in config, task
        assert "platform_support:" in config, task
        assert "required_arch: gfx950" in config, task
        assert "status: active" in config, task


def test_aiter_task_venvs_inherit_the_image_parent_packages():
    tasks = [
        "mla_decode_rope",
        "moe_routing_sigmoid_top1_fused",
        "pa_decode",
        "pa_prefill",
        "unified_attention",
    ]
    for task in tasks:
        runner = (
            ROOT / "tasks/repository/aiter" / task / "scripts/task_runner.py"
        ).read_text()
        assert "def _inherit_parent_site_packages(" in runner, task
        assert "aka_parent_site_packages.pth" in runner, task
        assert "site.addsitedir" in runner, task
        assert 'os.environ.setdefault("USER", "agentkernelarena")' in runner, task
        assert 'os.environ.setdefault("LOGNAME", "agentkernelarena")' in runner, task
        assert runner.index("_inherit_parent_site_packages(venv_python)") < (
            runner.index("if not ready_marker.exists()")
        ), task


def test_sglang_mxfp8_runners_handle_anonymous_docker_uids():
    tasks = [
        "mi355x_sglang_triton_mxfp8_grouped_gemm",
        "mi355x_sglang_triton_mxfp8_linear",
    ]
    for task in tasks:
        scripts = ROOT / "tasks/image_kernel" / task / "scripts"
        for name in ("task_runner.py", "standalone_driver.py"):
            source = (scripts / name).read_text()
            assert 'os.environ.setdefault("USER", "agentkernelarena")' in source
            assert 'os.environ.setdefault("LOGNAME", "agentkernelarena")' in source


def test_normal_attention_dot_predetermines_event_timing_for_rocm_teardown():
    runner = (
        ROOT
        / "tasks/hip2hip/gpumode/NormalAttention_dot/eval_tools/cal_kernel_perf.py"
    ).read_text()
    assert 'graph_fallback_reason = "capture_unsafe_rocm_graph_teardown"' in runner
    marker = runner.index('graph_fallback_reason = "capture_unsafe_rocm_graph_teardown"')
    assert "graph_enabled = False" in runner[marker - 200 : marker]


def test_refk_identity_emits_structured_per_case_performance_results():
    harness = (
        ROOT / "tasks/triton2triton/geak_eval/L1/refk_identity/test_kernel_harness.py"
    ).read_text()
    assert '"test_case_id": _label(cfg)' in harness
    assert 'Path("build/performance_report.json")' in harness
    assert "json.dumps(report_cases" in harness
