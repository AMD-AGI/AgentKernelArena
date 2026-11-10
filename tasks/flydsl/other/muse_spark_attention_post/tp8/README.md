# Muse Spark K2: attention output to expert input, one TP8 / EP8 rank (8 GPUs, in-kernel all-reduce)

One persistent 256-CTA launch per rank takes the attention output of the rank's 16 query heads through the
experts' input for one DFlash2 verify step of `rows` rows (rows / 8 requests of 8 draft tokens): the per-head
output norm-gate, the rank's o_proj K shard, the tensor-parallel all-reduce (every rank's bf16 partial rows
stored into every rank's symmetric buffer over XGMI and summed in rank order), the attention boundary
(weightless post norm, the gated fp32 residual stream updated in place, the pre-FFN norm), the sigmoid router's
top-8 of 256 experts and the pooler (8192 -> 4096) with the pre-expert norm. There is no host-side collective.
This is a multi-GPU task (`distributed.world_size: 8`).

**Model support is semi-experimental.** Muse Spark is registered in sgl-amd-mono's `SPECS` but absent from
the repository README's model table; it landed in one PR (#26), has no end-to-end server test, its default
served widths are 8 and 16 rows (`gfx950/config.py`: "32 and 64 rows are under test"), and the repository has
no 8-rank numerics test of K2 (`test_muse_spark_k2.py` runs a group of one and only compiles the TP8 build).

## Production origin

sgl-project/sgl-amd-mono at `52486703fbf27e77f05b573e4d931d4eec87118d` (Apache-2.0; `common/` files adapted
from ROCm/ATOM, MIT): `sgl_amd_mono/muse_spark/gfx950/build_k2.py::build_k2` with its stages
`stages/post_attn.py::run_k2` (which import `stages/ffn.py` and `stages/mxfp4.py`), `config.py`, `layout.py`,
`device.py`, the FlyDSL 0.3.2 bridge `compat.py` and `common/{guard,epoch,ops,plan,sync,mx}.py`. SGLang's ROCm
images pin sgl-amd-mono `e52803f9` (11 commits older); the baseline is vendored from `52486703`. The vendored
modules are those of `tasks/flydsl/other/muse_spark_layer/tp8`.

The build is `MuseMonoRunner.k2_kernel(rows)` with the runner's default environment: bf16 pooler
(`SGLANG_AMD_MONO_MUSE_FP8_DENSE` off), the row-sharded pooler (`PSHARD=1`, which the runner enables only for
rows <= 16: each rank computes 512 pooled rows and pushes them to every rank), no deferred K1 and no
pre-quantized expert input (both only in the merged layer launch). This is the runner's non-merged path
(`SGLANG_AMD_MONO_MUSE_LAYER=0`, or a step the merged launch does not take); by default the same stage code
runs inside the whole-layer launch (`tasks/flydsl/other/muse_spark_layer/tp8`). `baseline/kernel.py::run`
passes `MuseMonoRunner.launch`'s K2 arguments.

Vendoring changes (no kernel or stage code changed; details in `task.yaml` provenance): one flat package with
relative imports; the 0.3.2 bridge as plain functions called by `ops.global_ptr`/`global_load`,
`epoch.gstore`, `guard.stop_launch` instead of patching modules at import; frozen dataclasses / a slotted
class instead of `msgspec`; no `plan.kernel_sources` source digest; a local `kernel_signature`; environment
reads replaced by their defaults; `buffer_ops.py` from ROCm/aiter 652f2d4b and `dpp_utils.py` from ROCm/FlyDSL
dd837334.

**Scratch placement.** The runner keeps K2's mailbox scratch in ordinary device memory and moves the epoch word
on after every step, so mailbox tags never repeat. Here the epoch is a fixed input (`guard`, epoch 1) and the
scratch region sits in the framework's `symmetric` buffer after the peer regions (`symmetric_bytes` =
47,190,016 peer bytes + 15,134,720 scratch bytes), which the framework zeroes before every call. That buffer is
uncached memory (the same allocation as the runner's `PeerBuffer`), so the baseline's local scratch traffic is
uncached where production's is not; the kernel's hand-offs already use device-scope accesses, so the cost is
expected to be small but is unmeasured.

## Semantics (reference.py, float64)

The math of the ROCm model's ops K2 replaces, as `test/kernels/test_muse_spark_k2.py::_reference` states it,
keeping the model's materialized bfloat16 tensors and everything else in float64 (o and g are the logical
all-head tensors; the reference's o_proj is the whole sum the all-reduce forms):

- og = bf16(o rstd(o) sigmoid(g)) per 64-wide head (eps 1e-5); attn = bf16(og w_o^T);
- nb = bf16(attn rstd(attn)) (eps 1e-8); res' = alpha res + beta nb (the mutable stream);
- x = bf16(res' rstd(res') pre_w) (eps 1e-5);
- s = sigmoid(0.17677669529663687 x (router_hi + router_lo)^T); ids = top-8 of s + expert_bias, best first (a
  tie to the lower expert); scores = s[ids] / (sum + 1e-15);
- xin = bf16(x w_pool^T) rstd xn_w (eps 1e-5).

The kernel rounds each rank's o_proj partial to bf16 before the rank-ordered fp32 sum (one rounding of the sum,
as aiter's one-stage all-reduce); the reference keeps the exact sum.

## Inputs, shards and outputs

`inputs.py` documents every tensor. The generator returns logical tensors (o / g of all 128 heads, the whole
o_proj); `shard_inputs` gives rank r o / g columns 1024 r .. and w_o input columns 1024 r .. (its 16 heads) and
the replicated rest (the pooler whole: the kernel reads its rank's rows at S <= 16). Every output is replicated:
`ids`, `scores`, `xin` and the mutable `res`, all in `ranks_agree` (each rank sums the partials in rank order,
so the ranks agree bitwise). `guard` is the runner's epoch buffer (epoch 1, a 30 s deadline, clean error words):
read-only unless a wait fails, so a failed wait fails correctness.

Data: o, g unit normal; weights N(0, 0.02) bf16; norm weights 1 + N(0, 0.1); alpha U(0.5, 1), beta U(0, 0.5)
(the repository test's distributions).

**Routing margins.** The top-8 must not flip between reference and kernel. Each token's 8 experts are planted
as in the whole-layer task: expert e owns hidden columns 32e..32e+31; the residual stream carries levels 85,
75, .., 15 along random signs of its experts' blocks; the router column reads its block divided by alpha *
pre_w; noise N(0, 5e-4) and a bias U(-1e-3, 1e-3). Measured on CPU (float64, the framework's seeds of cases r8,
r24, r64): the smallest gap between consecutive biased scores among a token's top 8 is 0.0127 and between the
8th and 9th 0.0205 (58 / 137 / 232 distinct experts picked at 8 / 24 / 64 rows). The kernel's routing key keeps
24 bits of the biased score; the repository's own near-tie tolerance is 1e-3.

## Workloads

Phase: decode / verify only (the monokernels serve no prefill). K2's work does not depend on the context (no
context axis exists), and Muse Spark fixes every other dimension at TP8 / EP8, so the task uses the
**production-range span exemption** (`workloads.json` `performance_span`, `workload_evidence.json`):

- Observed production range: rows 8..64, the compiled widths 8, 16, 32, 64 (1, 2, 4, 8 requests; a smaller
  batch is assumed to run padded to the next width, as spec.py's check mode treats rows with
  `out_cache_loc <= 0` as padding). Default serving uses 8 and 16; 32 and 64 need
  `SGLANG_AMD_MONO_MUSE_WIDTHS=8,16,32,64`.
- Perf range: rows 8..64, span **8x** (all four widths estimated >= 10 us).
- Frequency (AgentX `cc-traces-weka-062126`, the per-session in-flight request distribution at concurrency 1,
  2, 4, 8, equal weight per concurrency point, padded to the widths): 8 rows 22.6%, 16 rows 21.8%, 32 rows
  21.4%, 64 rows 34.3%. **Derived from corpus statistics; not yet confirmed by a live capture.**

Performance cases: r8, r16, r32, r64 (one per width; perf-case span 8x). Correctness-only: r24 (the
non-power-of-two boundary; rows 8 is the minimum). Five compiled widths (8, 16, 24, 32, 64): four performance
cases on the one var axis need four distinct widths, and the non-power-of-two boundary a fifth, more than the
two to three the 10-minute budget guidance asks for.

Estimated baseline time per rank (bytes at ~5 TB/s plus ~10-15 us of in-kernel hand-offs; the o_proj shard 16.8
MB, the router 8.4 MB, the pooler 8.4 MB a rank sharded at 8 / 16 rows or 67 MB whole at 32 / 64 rows):
r8 ~20 us, r16 ~22 us, r32 ~35 us, r64 ~38 us. The 8- and 16-row cases are the closest to the 10 us floor.

## Comparison

- `ids` exact.
- `scores`: tolerance rtol 1e-3, atol 1e-5. Each row has 8 elements, so `baseline_relative`'s row RMS is an
  8-sample estimate of a ~1e-6 relative error (the whole-layer task measured a correct float32 implementation
  failing it on one of 512 elements). Float32 evaluations of the reference (one GEMM, or the kernel's eight bf16
  rank partials) peak at 0.07 of this bound (8e-5 relative), with the routing ids exactly equal; the kernel's x
  is a few 1e-4 from the reference's (the repository's note). The bound rejects normalizing the biased scores
  instead (the bias is ~2e-3 of a score): 22-24% of the elements fail.
- `xin`: `baseline_relative` with `max_violation_fraction: 1e-3`: the pooled row is a materialized bf16 tensor,
  and a correct implementation with another accumulation order flips some of its roundings, a one-ulp error at
  elements where the baseline's rounding agreed with the reference. A float32 evaluation with one exact o_proj
  sum, or with the kernel's bf16 rank partials in another order, against the kernel-like partial-rounding
  evaluation as baseline: at most 3.1e-5 of the elements over the bound, peak 1.36x.
- `res`: `sqnr` >= 55 dB. The stream is fp32 but carries beta times the bf16-rounded post norm nb, so wherever
  one implementation's nb rounds differently from the reference's its error is thousands of times the other's
  fp32-level error at that element: the float32 evaluations failed `baseline_relative` on up to 1.9e-4 of the
  elements (peaks 2-5x for kernel-like baselines, ~1e5x against an exact-sum one). Their SQNR: 88-94 dB with
  an exact o_proj sum, 66 dB with the kernel's bf16 rank partials (the all-reduce's numerics); 55 dB keeps an
  11 dB margin for the latter and rejects a missing rank's o partial or a dropped attention branch (estimated
  ~23 dB and ~14 dB, not measured). `xin` sees the stream at bf16 resolution through x.

## Unverified

Nothing has run on a GPU: FlyDSL 0.3.2 compile of the flattened package (the vendored modules are the
whole-layer task's, whose GPU QA was pending when this was written), numerics and the bounds against the real
baseline, timings and the 10 us floor of r8 / r16, the bitwise rank agreement, the standalone TP8 K2 (the
repository runs it only on a group of one), the uncached-scratch cost, compile time per width (the main risk
for the 10-minute budget, with five widths), and the workloads (corpus statistics, no live capture).
