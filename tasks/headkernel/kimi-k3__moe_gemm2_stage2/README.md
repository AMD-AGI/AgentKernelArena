This task exposes Kimi-K3's actual stage-2 GPU implementations in the pinned SGLang 0.5.20 runtime. Prefill uses FlyDSL-v2 GEMM2 followed by the native BF16 reduction. Decode uses the native Opus A8W4 kernel with explicit kernel ID 2005. The full served workload completed 64 requests at TP8/C64, ISL8192/OSL1024.

The editable bodies are `gemm2_body_v2` in [source/flydsl/mxmoe_gemm_v2.py](source/flydsl/mxmoe_gemm_v2.py) and `opus_moe_stage2_a8w4_decode_kernel_gfx950` in [source/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh](source/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh). The host wrappers, import graph, signatures, launch controls, build recipe, native reduction and all other code are protected. The HIP source is freshly compiled into a task-owned extension. Each FlyDSL leg uses a distinct private dispatcher and emitter package with a source-bound GPU symbol; neither leg replaces the installed dispatcher module.

The three exact cases are:

| Stage | Tokens | Calls per rank | Calls across eight ranks |
| --- | ---: | ---: | ---: |
| Prefill | 8192 | 184 | 1472 |
| Prefill | 16384 | 2852 | 22816 |
| Decode | 64 | 94208 | 753664 |

All cases preserve packed FP4 model weights, FP8 activations, native tiled E8M0 scales, topk 16, intermediate dimension 384, model dimension 3584, captured physical storage and aliases. Decode accumulates into caller-zeroed BF16 output; prefill overwrites BF16 output. Complete empirical work distributions, including dynamic valid sorted-row counts, remain in [provenance/WORK-DISTRIBUTIONS.json](provenance/WORK-DISTRIBUTIONS.json).

The trusted host materializes the exact assets in [fixtures/EXTERNAL-MANIFEST.json](fixtures/EXTERNAL-MANIFEST.json). No tensor blobs belong in the Git task. Every replay creates fresh legal distinct-expert routes at an actually observed valid-row work amount, remaps captured quantized activation payloads and their exact scale layout, and generates normalized routing weights. Model-weight bytes remain readonly.

Before candidate execution, the harness owns CPU input truth. After execution, it freezes candidate outputs and complete post-input storage on CPU before invoking the separately bound frozen reference. Reference GPU output storage is cleared after its CPU snapshot. Correctness retains the 0.05 RMS-relative tolerance, exact metadata and alias checks, and no-op/wrong-output negative controls. Performance uses 10 warmups and 100 checked single-call CUDA graph replays.

In the pinned gfx950 container, after trusted fixture materialization:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

Source and metadata CPU checks pass. GPU correctness, timing and a framework-finalized task-validator `PASS` are still required before publication.
