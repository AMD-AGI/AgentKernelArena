# Muse Spark DFlash2 draft D-MLP: one draft layer's MLP half, one TP8 rank (8 GPUs, in-kernel all-reduce)

One persistent 256-CTA launch per rank runs one layer's MLP half of Muse Spark's DFlash2 draft model for a
draft step of `rows` rows (rows / 8 requests, a block of 8 rows each): the post-attention fused add and RMSNorm,
the convolution's kernel projection (per-token coefficients of 2 sides x 2 taps x 512 groups of 16 channels),
the grouped two-tap convolution within each 8-row block, the rank's 3584 gate and 3584 up rows of the
28672-wide MLP on bf16 MFMA GEMVs, silu(g) u, the rank's down K shard, the tensor-parallel all-reduce (every
rank's bf16 partial rows stored into every rank's symmetric buffer over XGMI, summed in rank order) and the
finish convolution with side 1's coefficients. There is no host-side collective. This is a multi-GPU task
(`distributed.world_size: 8`).

**Model support is semi-experimental.** Muse Spark (and its DFlash2 draft, RadixArk/Muse-Spark-Open-DFlash2)
is registered in sgl-amd-mono's `SPECS` but absent from the repository README's model table; it landed in one
PR (#26), has no end-to-end server test, its default served widths are 8 and 16 rows (`gfx950/config.py`: "32
and 64 rows are under test"), and the repository tests D-MLP only on a group of one (`test_muse_spark_dmlp.py`;
the TP8 build is only compiled).

## Production origin

sgl-project/sgl-amd-mono at `52486703fbf27e77f05b573e4d931d4eec87118d` (Apache-2.0; `common/` files adapted
from ROCm/ATOM, MIT): `sgl_amd_mono/muse_spark/gfx950/draft_mlp.py::build_dmlp` (`run_dmlp`), which imports
`stages/post_attn.py` (`sigmoid`, and with it `stages/ffn.py` and `stages/mxfp4.py`), with `config.py`,
`layout.py`, `device.py`, the FlyDSL 0.3.2 bridge `compat.py` and `common/{guard,epoch,ops,plan,sync,mx}.py`.
SGLang's ROCm images pin sgl-amd-mono `e52803f9` (11 commits older); the baseline is vendored from `52486703`.
The vendored modules other than `draft_mlp.py` are those of `tasks/flydsl/other/muse_spark_layer/tp8`.

The build is `DraftRunner.kernel(rows)` = `DMLPBuild(tokens=rows, npes=8)` (every stage), the launch
`muse_spark/draft.py::DraftRunner.launch` makes for each of the draft's 5 layers (stage 1 and stage 2, the
default). `baseline/kernel.py::run` passes its arguments.

Vendoring changes (no kernel or stage code changed; details in `task.yaml` provenance): one flat package with
relative imports; the 0.3.2 bridge as plain functions called by `ops.global_ptr`/`global_load`,
`epoch.gstore`, `guard.stop_launch` instead of `compat.install()` patching modules at import; frozen dataclasses
/ a slotted class instead of `msgspec`; no `plan.kernel_sources` source digest; a local `kernel_signature`;
environment reads replaced by their defaults; `buffer_ops.py` from ROCm/aiter 652f2d4b and `dpp_utils.py` from
ROCm/FlyDSL dd837334.

**Scratch placement.** The runner keeps D-MLP's mailbox scratch in ordinary device memory and moves the epoch
word on after every step, so mailbox tags never repeat. Here the epoch is a fixed input (`guard`, epoch 1) and
the scratch region sits in the framework's `symmetric` buffer after the peer region (`symmetric_bytes` =
16,781,312 peer bytes + 108,929,024 scratch bytes), which the framework zeroes before every call. That buffer is
uncached memory (the same allocation as the runner's `PeerBuffer`), so the baseline's local scratch traffic
(the GEMV partials, ~100 MB of mailbox regions sized for 64 rows, of which a step touches its rows' share) is
uncached where production's is not; the cost is unmeasured.

## Semantics (reference.py, float64)

The SGLang draft layer's MLP half as `test/kernels/test_muse_spark_dmlp.py::_reference` states it, over the
logical (whole, TP1) MLP, keeping the bfloat16 tensors that both the layer and the kernel materialize and
everything else in float64:

- res' = bf16(res + attn_out) (output `res_out`); h = bf16(res' rstd(res') w_post) (eps 1e-5);
- delta = h w_kp^T as [rows, side, tap, group]; conv_s(x)[t, c] = (base[s, 0, c] + delta[t, s, 0, c // 16])
  x[t, c] + (base[s, 1, c] + delta[t, s, 1, c // 16]) x[t - 1, c], the second term only within t's 8-row block;
- hc = bf16(conv_0(h)); g, u = bf16(hc w_gate^T), bf16(hc w_up^T); a = bf16(bf16(silu(g)) u);
- d = a w_down^T (the all-reduce's sum, unrounded); out = conv_1(d).

The kernel keeps the coefficients in float32 (SGLang's bf16 Linear rounds them) and sums the ranks' bf16
partials of d in float32 before conv_1 (one rounding of each partial); the reference keeps both exact. The
repository test checks the kernel against both chains (its own rounding points at cosine 0.9999, SGLang's bf16
chain at 0.999).

## Inputs, shards and outputs

`inputs.py` documents every tensor. The generator returns logical tensors (the whole 28672-wide gate / up and
down projections, 1.3 GiB); `shard_inputs` gives rank r gate rows 3584 r .. and up rows 28672 + 3584 r .. (gate
first, as SGLang's MergedColumnParallelLinear shard) and down input columns 3584 r ..; the rest is replicated.
Both outputs are replicated: `res_out` and `out`, in `ranks_agree` (every rank sums the partials in rank order).
`guard` is the runner's epoch buffer (epoch 1, a 30 s deadline, clean error words): read-only unless a wait
fails, so a failed wait fails correctness.

Data: the repository test's distributions (res unit normal, attn_out 0.5 N(0, 1), w_post 1 + 0.1 N, w_kp
0.01 N, base tap 0 1 + 0.05 N and tap 1 0.05 N, gate / up 0.02 N), with the down projection at 0.01 N (the
test's 0.02 is over its one-rank 3584 columns; the logical sum is over 28672).

## Workloads

Phase: draft steps of the speculative decode only (a block of 8 rows a request, run as a TARGET_VERIFY batch;
no prefill). D-MLP's work does not depend on the context (no context axis exists), and the draft model fixes
every other dimension at TP8, so the task uses the **production-range span exemption** (`workloads.json`
`performance_span`, `workload_evidence.json`):

- Observed production range: rows 8..64, the compiled widths 8, 16, 32, 64 (1, 2, 4, 8 requests; a smaller
  batch is assumed to run padded to the next width). Default serving uses 8 and 16; 32 and 64 need
  `SGLANG_AMD_MONO_MUSE_WIDTHS=8,16,32,64`.
- Perf range: rows 8..64, span **8x** (all four widths estimated >= 10 us).
- Frequency (AgentX `cc-traces-weka-062126`, the per-session in-flight request distribution at concurrency 1,
  2, 4, 8, equal weight per concurrency point, padded to the widths): 8 rows 22.6%, 16 rows 21.8%, 32 rows
  21.4%, 64 rows 34.3%. **Derived from corpus statistics; not yet confirmed by a live capture.**

Performance cases: r8, r16, r32, r64 (one per width; perf-case span 8x). Correctness-only: r24 (the
non-power-of-two boundary; rows 8 is the minimum). Five compiled widths (8, 16, 24, 32, 64): four performance
cases on the one var axis need four distinct widths, and the non-power-of-two boundary a fifth, more than the
two to three the 10-minute budget guidance asks for.

Estimated baseline time per rank (the 209 MB of weights a rank stream once: the kernel projection 33.6 MB, gate
/ up 117 MB, down 58.7 MB, at ~5 TB/s, plus ~15 us for the seven mailbox hops): r8 ~57 us, r16 ~58 us, r32
~60 us, r64 ~65 us (the work is weight-bound: the rows add little).

## Comparison

- `res_out`: the default `baseline_relative` bound (bf16(res + attn_out) is exact up to its one rounding).
- `out`: `baseline_relative` with `max_violation_fraction: 1e-3`. Two correct float32 evaluations of the
  reference that both use the kernel's bf16 rank partials, in different summation orders, each as the other's
  baseline, put up to 2.0e-4 of the elements over the default bound at 64 rows (peak 1.9x): the bf16
  intermediates (hc, g, u, a) round differently at a few elements. An exact-sum evaluation against the
  kernel-like one: 4e-6 over (peak 1.14x). Skipping the finish convolution fails 99% of the elements (peak
  ~750x); every admitted element stays within the framework's 16x peak limit.

## Unverified

Nothing has run on a GPU: FlyDSL 0.3.2 compile of the flattened package (D-MLP is vendored here for the first
time; its other modules are the whole-layer task's), numerics and the bound against the real baseline, timings,
the bitwise rank agreement, the TP8 D-MLP on eight ranks (the repository runs it only on a group of one), the
uncached-scratch cost, compile time per width (the main risk for the 10-minute budget, with five widths), and
the workloads (corpus statistics, no live capture).
