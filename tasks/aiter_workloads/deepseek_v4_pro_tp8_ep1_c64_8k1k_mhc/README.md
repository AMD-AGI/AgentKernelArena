# DeepSeek V4 Pro mHC post+pre boundary, TP8/EP1, concurrency 64, 8192/1024

## Task contract

This task contains authentic captured inputs and measured call frequencies
for the selected rank-0, layer-0 site. The runtime, upstream source and observed
native artifacts are bound independently of the editable implementation.
Correctness, timing and graph replay are checked by the task's protected
evaluation actions and the shared Arena task validator.

`workloads.json:runtime_binding.input_assets` binds the four input carriers.
The scoring image (`image_ref`), runtime build evidence and native artifact
manifest are pinned under the same runtime binding. It also records
`dispatch_environment`: every setting in
`scripts/runtime.py:DISPATCH_ENVIRONMENT` (architecture/CU selection and AITER
import/JIT roots) as a string value or `null` for absence. The runtime compares
the effective environment with it before importing AITER and never writes or
scrubs it.

## Workload

DeepSeek V4 Pro served by SGLang 0.5.20 (commit 94602c9c), TP8/EP1/DP1,
64 concurrent requests sent at once, 8192 input and 1024 output tokens,
with requests supplying token IDs (`workload.input_mode: token_ids`),
chunked prefill 8192, eager prefill, CUDA-graph decode, `SGLANG_USE_AITER=1`,
gfx950, AITER 2cb3d175. This task selects the attention-output to FFN-input
boundary at `model.layers.0`, rank 0, calling the public
`aiter.ops.mhc.mhc_fused_post_pre`. It does not pool other layers' weights or
assume that this site's measured benefit applies to all mHC calls.

## Operation

One call combines the mHC post step of the finished sub-block with the mHC
pre step of the next sub-block (hc_mult 4, hidden 7168):

1. `next_residual = comb_res_mix^T @ residual_in + post_layer_mix * layer_input`
   (per token; fp32 math, bf16 result).
2. From `next_residual`: RMS-scaled projection by `fn` (24 mixes), scaled by
   `hc_scale` and shifted by `hc_base`; sigmoid pre/post mixes (post scaled by
   2.0); a 4x4 combination mix normalized by 20 Sinkhorn iterations.
3. `layer_input_out` = pre-mix-weighted sum of the 4 residual streams, then
   RMSNorm with `norm_weight`.

Returns `(post_mix fp32 [M,4,1], comb_mix fp32 [M,4,4],
layer_input_out bf16 [M,7168], next_residual bf16 [M,4,7168])`.

## Boundary and freedom

The entry point is `source/kernel.py:mhc_fused_post_pre`, called as
`mhc_fused_post_pre(layer_input, residual_in, post_layer_mix, comb_res_mix,
fn, hc_scale, hc_base, rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
hc_post_mult_value=2.0, sinkhorn_repeat=20, norm_weight=..., norm_eps=1e-6)`.
The starting implementation delegates to the pinned upstream call. The
baseline is the pinned upstream call itself.

Everything inside this call may be reorganized, replaced or rewritten in any
backend, provided the call keeps its signature and contract. Inputs are
read-only. Weights (`fn`, `hc_scale`, `hc_base`, `norm_weight`) arrive in the
layout shown below and any transformation of them is charged inside the call.
Each invocation must compute from its current dynamic inputs. Do not reuse
dynamic-input-dependent results or intermediates from previous invocations,
warmup or graph capture. Static compilation is allowed. This task has no
candidate load-time weight-transform phase: charging candidate weight
transformations per call limits the reformulations tested, equally for both
search methods. A task with separately charged static preparation would be a
different contract.

| Tensor | Shape | dtype | Kind |
|---|---|---|---|
| layer_input | [M, 7168] | bf16 | activation |
| residual_in | [M, 4, 7168] | bf16 | activation |
| post_layer_mix | [M, 4] | fp32 | activation |
| comb_res_mix | [M, 4, 4] | fp32 | activation |
| fn | [24, 28672] | fp32 | weight |
| hc_scale | [3] | fp32 | weight |
| hc_base | [24] | fp32 | weight |
| norm_weight | [7168] | bf16 | weight |

## Cases

| Case | M | Timing | Measured calls |
|---|---|---|---|
| `prefill_m8192` | 8192 | GPU events (eager prefill) | 64 |
| `decode_m64` | 64 | CUDA graph (decode graph) | 1,024 |

Weights are observed executed calls at this one site over the complete scored
request window, excluding warmup. The 1,024 decode calls are the recorded
execution count, not an estimate derived by subtracting one from requested
output tokens. The score compares `64 * prefill_ms + 1024 * decode_ms` for
upstream AITER and the candidate. This is a weighted isolated-call proxy;
end-to-end serving benefit requires a separate fixed-workload measurement.
Timing uses repeated warm inputs under the unchanged task timing policy.
Actual cache residency and its equivalence to serving are unmeasured; fitting
inputs within nominal cache capacity does not establish either. Both search
methods receive the same timing policy and complete-call boundary.

## Captured inputs

A case whose `inputs` field names a task-relative file and its SHA256 loads
that call's eight captured tensors (activations and the call's own layer
weights), keeping their shapes, strides, dtypes, storage offsets, backing-storage
sizes and storage sharing, all declared in `params.tensors`.

Large carriers may instead use the protected `runtime_binding.input_assets`
mapping in `workloads.json`. Each key is the same task-relative logical path
used by a case; its value declares a canonical container-visible absolute
`path`, `sha256` and `size_bytes`. The operator supplies these files through a
read-only mount outside the copied task, with no writable alias exposed to the
worker. The loader rejects local shadows, symlinks in the asset path, a writable
mount, wrong size or a digest that differs from the case. It hashes the opened
file before loading tensors, just as for bundled inputs. Missing assets fail;
there is no download, environment-based lookup or implicit fallback. Under
Arena's native GEAK adapter, the operator also declares these exact files in
the existing `runtime_artifacts` binding. Other agents use the same task runner
and declared read-only files. The mapping changes storage placement only.

Both cases bind actual activations and prepared weights captured from the
declared workload. The timing point is the first scored occurrence in each
observed stratum; the second compatible occurrence supplies correctness
coverage. Their layouts, storage identities and prepared-weight seals are
recorded in `capture_binding.json`. Counts come from the separate complete
accounting run; diagnostic tensor-export counts do not replace them.
Synthetic generation remains available for source fixtures but no current
case selects it and synthetic cases cannot receive a workload score.
Enrollment additionally requires a frequency provenance with
`status: measured`, `enrolled_for_scoring: true` and `case_counts` rows
equal to `case_weights`. It also opens the task-relative `capture` record by
its SHA256 and requires its `exported_cases` rows to match every scored case's
identity, count and input carrier. Changing `binding_status` alone enrolls
nothing.

Every captured case also requires `inputs.current_value_replay: {path, sha256}`,
binding a second authentic carrier containing all eight public tensors. The
protected capture receipt must establish the same caller, rank, layer,
prepared-weight seals, call constants and complete tensor layout for both
points. The backing storages of all four prepared weights must be byte-identical,
including bytes outside visible views. Dynamic and prepared storages must be
disjoint. This second point adds correctness coverage without a scoring weight.
Synthetic preflight cases retain their existing behavior and cannot enroll for
scoring or stand in for either captured state.

## Correctness

The protected reference (`scripts/reference.py`) is the upstream AITER torch
reference from `op_tests/test_mhc.py` (post then pre, in that order). Every
output must have the exact shape, dtype and device, be contiguous and finite,
and match elementwise with rtol = atol = 1e-2 (the upstream test defaults).
The whole backing storage of every input is compared bytewise before and
after each input state. For captured cases, the original storage is checked
before the harness copies the second point's four dynamic backing buffers into
the original public storage. Every tensor object, address, offset and alias is
retained; prepared weights are checked and remain unchanged. The harness takes
a fresh snapshot, computes a fresh oracle, and checks all four outputs from
the same callable again. Under graph timing this calls `TimedRun.rerun()` on
the exact graph that supplied the scored samples. The copy, oracle and extra
execution occur outside scored intervals for both baseline and candidate.

Task validation first checks that both oracle results are valid and have the
same output geometry. The pair must reject the previous oracle output tuple
under the fixed four-output numerical check against the second oracle; at
least one output must differ beyond its declared tolerance. The
`per_output_moved` diagnostic records which outputs meet that condition. A pair
that accepts the previous oracle tuple needs a different authentic point.
Invalid oracle results do not count as separation. Tolerance regions can
overlap, so this does not exclude every approximation that passes both states;
no candidate-dependent old-output veto is added.

Baseline actions end by re-observing the baseline's own
declared native artifacts against the protected manifest
(`runtime_binding.native_identity`, captured with `scripts/native_identity.py`). Task validation
also runs a known-answer identity and rejects a deliberately wrong row in
each output.

This task measures an isolated call at one captured site. An improvement in its
weighted time is a workload proxy; deployment and a complete run under the same
serving settings are required to establish end-to-end benefit.
