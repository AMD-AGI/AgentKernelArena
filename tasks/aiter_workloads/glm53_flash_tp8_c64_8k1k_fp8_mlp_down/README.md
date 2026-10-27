# GLM5.3-Flash TP8: FP8 MLP down GEMM

Optimize the complete upstream AITER `gemm_a8w8_blockscale_bpreshuffle`
call for the first dense MLP's local down projection. The benchmark uses the
recorded TP8, concurrency-64, 8,192-input/1,024-output settings with an explicit
fixed batching scenario. This is a configuration-derived task. Its inputs are
newly generated; its frequencies are not measured live scheduler counts.

The pinned model configuration in `model_config.json` is from
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json).
It gives hidden width 4096 and dense intermediate width 12288. TP8 partitions
the input of a row-parallel down projection to `K=12288/8=1536`; `N=4096`.
The first dense MLP down projection is not excluded by the checkpoint's FP8
quantization configuration. This public configuration revision was selected
for the new benchmark; the old serving checkpoint revision remains unknown.

| Case | A | Canonical B | Weight |
| --- | --- | --- | ---: |
| Prefill | `[8192,1536]` | `[4096,1536]` | 64 |
| Decode | `[64,1536]` | `[4096,1536]` | 1024 |

The scenario processes one complete prefill chunk per request, followed by
1024 full decode batches. Partial batches, scheduler changes and speculative
decoding require different cases and weights. `frequency_provenance.json`
records this derivation. The objective is the ratio of weighted total baseline
time to weighted total candidate time, not the mean of the two speedups.

## Public operation and freedom

`source/kernel.py::mlp_down_gemm(x, weight, x_scale, weight_scale)` takes
already quantized E4M3FN activation and weight, FP32 per-1x128 activation
scales with column-major stride, and FP32 128x128 weight scales. The weight
has AITER's `(16,16)` preshuffle. It returns a BF16 `[M,4096]` tensor.

The baseline and starting candidate call AITER at the pinned revision in
`config.yaml`. Normal upstream tuned dispatch is retained. Activation
quantization, checkpoint weight preparation and the TP all-reduce are outside
this public GEMM boundary. Any candidate-specific preparation, conversions,
launches or intermediate buffers belong inside the measured call. Input
tensors are read-only. Output caching or computing input-dependent results
during import is forbidden.

The entire implementation under `source/` is editable using available GPU
backends. Preserve the public signature and numerical semantics. The installed
upstream package, references, cases, policies and evaluation scripts are
protected. No agent or model is required by the task.

## Values, correctness and timing

Fresh operands follow the upstream test's roles with an explicit generated
distribution: uniform `[-2,2]` converted to FP8 and positive power-of-two FP32
scales. The precise exponent ranges and two seeds are in `workloads.json`.
These values do not represent trained checkpoint weights or captured
activation statistics.

The independent reference dequantizes both operands and computes an FP32
matrix product, then rounds to BF16. Every element must pass `rtol=atol=0.01`;
relative Frobenius error must also be at most `0.01` to reject zero answers at
small output scales. Both baseline and candidate face the same checks, with
fresh inputs, a zero-output rejection control and input-mutation checks.

Timing uses Arena's canonical helper: eager GPU events for prefill and a CUDA
graph for decode, with 10 warmups and 100 samples. Every timed sample's output
is checked outside its timed interval. After timing, dynamic values are changed
in the same public buffers and the actual retained callable/graph is replayed
against a recomputed reference. This tests current-value dependence.

The standard seven actions are provided by `scripts/evaluate.py`. Use the
documented [SGLang/AITER/FlyDSL runtime](../../../docs/install/install.md#pinned-aiter-and-flydsl-on-sglang-0520)
and the ordinary Arena `task_validator` workflow. GPU qualification must cover
these exact sources and cases; CPU checks do not establish it. The historical
serving runtime contained overlays, so current serving dispatch and end-to-end
benefit require separate integration and measurement.

## Attribution

AITER is materialized from [ROCm/aiter](https://github.com/ROCm/aiter/tree/2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0)
under its MIT license. The numerical reference follows `run_torch` in
`op_tests/test_gemm_a8w8_blockscale.py`; the task uses none of the old saved
kernel implementations, input corpora or timing results.
