# DeepSeek V4 Pro: FP8 block-scaled dense linear (AITER)

This task optimizes the complete linear call at
`model.layers.0.self_attn.wqkv_a`, rank 0. It uses authentic public-call inputs
from DeepSeek V4 Pro served with TP8, EP1, DP1, concurrency 64, 64 token-ID
requests, 8,192 input tokens and 1,024 output tokens. The baseline uses upstream
AITER commit `2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0`.

The serving runtime is SGLang 0.5.20, commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, on gfx950. Prefill is eager with
chunk size 8,192; decode uses CUDA graphs. The serving settings and source
references are recorded in `workloads.json`. Different parallelism, batching,
input lengths, serving flags or call sites require their own task binding.

## Computation and cases

The public inputs are an FP8 e4m3fn activation `x[M,7168]`, FP32 activation
scales `x_scale[M,56]`, a preshuffled FP8 weight `weight[2048,7168]`, and FP32
weight block scales `weight_scale[16,56]`. The output is BF16 `[M,2048]`.
The activation scales keep the caller's column-major stride `[1,M]`; weights
keep the loader's AITER `(16,16)` preshuffle. There is no bias.

Both cases start from already quantized activations. The producer's RMSNorm
and activation quantization are outside this call and outside its timing.

| Case | M | Timing | Observed calls |
| --- | ---: | --- | ---: |
| `prefill_m8192_n2048_k7168_fp8in` | 8192 | GPU events | 64 |
| `decode_m64_n2048_k7168_fp8in` | 64 | CUDA graph | 1024 |

These counts come from the complete scored accounting window, with warmup
excluded. `frequency_provenance.json` binds the counts to `capture_binding.json`
and the exact case identities. The objective is:

```text
(64 * baseline_prefill_ms + 1024 * baseline_decode_ms)
--------------------------------------------------
(64 * candidate_prefill_ms + 1024 * candidate_decode_ms)
```

The task covers one selected site. Its weights do not pool the other model
linears or extrapolate this site's benefit to other layers or ranks. Each phase
also has a second authentic state for correctness; it adds no timing weight.

## Implementation freedom

`source/kernel.py` defines
`fp8_blockscale_linear(x, weight, weight_scale, x_scale=None)`. It initially
uses the pinned AITER implementation. The optimizing agent may replace that
implementation and add helpers under `source/`, using GPU backends available in
the bound runtime. Internal launches, layouts and intermediate buffers may
change while preserving the public signature and returned tensor semantics.

Inputs are read-only. Candidate-specific preparation or weight conversions
belong inside the measured call. There is no separate load-time transformation
step. Import-time compilation is allowed; computing outputs or reusable
input-dependent intermediates during import is not. The installed AITER
package, protected upstream source, case data, reference, numerical policy and
evaluation scripts must remain unchanged.

The independent protected baseline is `baseline_linear` in
`scripts/evaluate.py`. The reference computes an FP32 matmul of dequantized
operands using the canonical unshuffled weight. It imports neither AITER nor
candidate code. Every output element must satisfy
`torch.testing.assert_close` with `rtol=atol=1e-2`. These are the upstream
`checkAllclose` default thresholds; this task does not allow its additional
5% mismatch allowance.

## Captured inputs and replay

The input distribution is the fixed empirical set of four authentic captured
states: prefill first/second and decode first/second. Primary states are reused
for timing; the second states test current-value replay. No random distribution,
regenerated activation or synthetic weight is part of this task.

Each case binds an input carrier and its SHA256. Tensor shapes, strides,
dtypes, offsets, backing-storage sizes and sharing are declared in
`params.tensors`. The canonical weight is recovered by inverting the preshuffle
and checked by applying the pinned forward shuffle. This checks conversion
consistency; the captured caller, generation and prepared-weight records
establish where the bytes came from.

Large carriers are supplied separately from Git through
`runtime_binding.input_assets`. Each task-relative logical path maps to a
container-visible absolute mount path, SHA256 and byte size. These are read-only
mount points, not paths to a developer's checkout. The loader rejects local
shadows, symlinks, writable mounts, size or digest mismatches, and missing
assets. It does not download inputs or infer a substitute case.

`inputs.current_value_replay` binds the second state from the same site and
prepared weights. After timing and input-mutation checks, the evaluator copies
that state's dynamic backing buffers into the original public storage. Tensor
objects, addresses, offsets, aliases and weights remain unchanged. It computes
a fresh oracle and replays the same callable; graph timing replays the graph
that supplied the scored samples. These checks are outside scored intervals
for both baseline and candidate.

Task validation checks that the second oracle rejects the first state's
output under the fixed numerical rule. It also exercises a zero-activation
identity and a deliberately wrong output. A pair that cannot distinguish stale
output must be replaced by a different authentic capture, without weakening
the tolerance.

## Evaluation and runtime

The seven standard Arena actions are implemented in `scripts/evaluate.py`:

```text
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Every action emits an `ARENA_EVAL_RESULT` envelope and fails nonzero on error.
Compilation executes the real callable and its JIT path. Timing uses Arena's
canonical benchmark helper, with the declared event/graph method checked.
Measured outputs and replay outputs are checked; complete input backing
storages are checked for mutation outside timing.

The runtime binding declares the scoring image, build evidence, dispatch
environment, required baseline native artifacts and input assets. Runtime
checks compare every declared dispatch value or absence before importing AITER,
without changing the environment. Baseline actions re-observe their required
native artifacts after loading. File identity does not establish dispatch or
explain performance. Candidate artifacts need not match baseline artifacts.
The protected binding now fixes the scoring image, the four input assets and
both extensions observed during calibration: `module_aiter_core` and
`module_gemm_a8w8_blockscale_bpreshuffle_cktile`. Their file hashes agree with
the retained produced JIT cache. The default bpreshuffle table and all ten model
tables are pinned through `source_identity.json`; no selected internal kernel
or table row is claimed. Separate HSACO or profiler evidence is not required
for this declared extension-file identity.

The four authentic states passed the saved calibration checks. The historical
calibration remains `OBSERVED_UNBOUND`; `runtime_binding/build_evidence.json`
records the prospective binding derived from those observations and the durable
cache. Binding these dependencies permits the standard task actions; it does
not itself establish a framework-finalized task-validator PASS or a performance
result. Qualification is recorded separately against the final task source.

An isolated weighted-time improvement is a workload proxy. Integration and a
complete run under the same serving settings must establish end-to-end gain.

## Source attribution

[ROCm/AITER](https://github.com/ROCm/aiter), MIT license, is materialized at the
pinned commit under `upstream/`. The reference follows `run_torch` in
`op_tests/test_gemm_a8w8_blockscale.py`; the baseline call order follows
SGLang's `aiter_w8a8_block_fp8_linear` in
`python/sglang/srt/layers/quantization/fp8_utils.py`.
