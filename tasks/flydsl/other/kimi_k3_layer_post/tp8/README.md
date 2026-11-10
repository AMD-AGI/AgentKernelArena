# Kimi-K3 layer post (o_proj to MoE output), one TP8 rank (8 GPUs, in-kernel collectives)

One persistent 256-CTA launch per rank takes a Kimi-K3 decoder layer from the attention's core output to
the layer output for a narrow step of `rows` <= 8 token rows: o_proj and its tensor-parallel all-reduce, the
residual add and the MLP attention-residual (AttnRes) seam, the router's top-16 of 896 experts, the latent
down projection, the shared experts, the routed experts on MXFP4 weights (SiTUv2), the latent all-reduce and
RMSNorm, the latent up projection and the final all-reduce. The ranks exchange partial rows inside the kernel
through each other's symmetric buffers over XGMI; there is no host-side collective. This is a multi-GPU task
(`distributed.world_size: 8`). The test-only standalone MoE launch of the same repository (`stages/moe.py`'s
`moe`, which "runs in no forward") runs the same MoE stages and is covered by this task, not a separate one.

## Production origin

sgl-project/sgl-amd-mono at `52486703fbf27e77f05b573e4d931d4eec87118d` (Apache-2.0; the shared device
primitives adapted from ROCm/ATOM, MIT): `sgl_amd_mono/kimi_k3/gfx950/layer.py` (K2, "As ATOM's
layer_post") with its stages `stages/{moe,oproj,seam,gemv,dims}.py`, `stamps.py`, `tag.py`, the model
constants of `kimi_k3/__init__.py` (`model.py`), and `common/{ops,sync,plan,guard,mx}.py`. SGLang's ROCm images
pin sgl-amd-mono `e52803f9` (11 commits older); the baseline is vendored from `52486703`. Kimi-K3 runs this
launch on every MoE layer of a step of at most 8 rows (decode, or one request's DSpark verify); Kimi-K3.1
runs it for its steps of one request (DSpark / DFLASH verify of L = 6 or 8 rows), and its wide launch above
that.

The build is `kimi_k3/spec.py::_weights`' `K2Build` for a covered middle layer that is not a block-write
layer: `nblocks` 4 live bank rows, `reset=False`, the serving defaults `fold` (the reduce writes
MoE output + updated prefix) and `reduce`, `guint` (the gate / up interleaved expert layout `spec.py`
requires), eps 1e-5 (K2Build's defaults; the checkpoint's values are not in the source), and the OPT bits
`layer.K2_OPT` **minus the three sentinel all-reduces** (`OPT_SENTS`): their peer regions must be armed with
0xFF bytes once (`init_peer`) and the readers re-arm them, which the framework's zeroing of the symmetric
buffer before every call would undo. Upstream's `check_k2.py` checks every OPT bit bit for bit against the
launch without it, so the baseline computes the same values through tagged mailbox pairs; it is the
production launch minus that optimization (a framework request below would allow it).

Vendoring changes (no kernel or stage code changed; details in `task.yaml` provenance): one flat package
with relative imports; the FlyDSL 0.3.2 copies of `common/` from `tasks/flydsl/other/muse_spark_layer/tp8`
(the 0.3.2 bridge `compat.py` called directly, `msgspec` replaced by frozen dataclasses / a slotted class,
no `plan.kernel_sources` digest); `K2Build` / `MoeBuild` / `AttnPostBuild` as frozen dataclasses; the route
table's LDS `fx.llvm.atomic_or` (FlyDSL >= 0.3.4) through a new `compat.lds_atomic_or` on the llvm dialect
(the form of upstream Muse Spark's `device.lds_atomic_add`); `tag_offset()` sized over this launch's own
scratch (upstream also counts Kimi-K3's K1 and MLA-front launches, not in this package); the host-side
sentinel arming (`init_peer`, a module global), the `k2_launch` wrapper, the test-only standalone MoE
launch and the never-called standalone `attn_post` launch removed. `buffer_ops.py` is ROCm/aiter 652f2d4b's
(sgl-amd-mono imports `aiter.ops.flydsl.kernels.buffer_ops`) and `dpp_utils.py` ROCm/FlyDSL dd837334's.

**Placement.** `symmetric` (10,821,632 bytes) holds the two epoch-parity peer halves sized for the widest
width and the guard flags (`layer.flags_offset()`), then, from byte 10,051,584, the runner's launch scratch
(`K3MonoRunner.scratch`, `layer.scratch_alloc_bytes()` = 770,048 bytes, the OPT_TAG_X rows included). In
production that scratch is also an uncached one-rank `PeerBuffer`, so the placement matches; the runner
never zeroes it and bumps the epoch once a step instead, while here the epoch is a fixed input (`guard`,
epoch 1) and the framework zeroes the buffer before every call, so a call's mailbox tags never meet a
previous call's words.

## Semantics (reference.py -> k3_numerics.layer, float64)

Composed from the repository's references (`test/kernels/kimi_k3/check_k2.py::seam_reference` and
`check_common.moe_reference`, whose routed experts are aiter `fused_moe` with SiTUv2 and per-1x32 MXFP4),
rewritten in float64 with the production tensors' bfloat16 hand-offs and both MXFP4 quantizations exact:

- **o_proj**: each rank's bf16 partial `core_r w_o_r^T`; `attn = bf16(sum_r partials)`;
  `prefix = bf16(prefix_in + attn)`.
- **seam**: sources = the 4 live bank rows and `prefix`; score_s = sum(rmsnorm(src_s) ares_nw ares_qk);
  softmax; `x = bf16(rmsnorm(sum_s p_s src_s) in_nw)`.
- **router**: `sig = sigmoid(x w_gate^T)`, the top-16 of `sig + bias` (ties to the lower expert),
  weights `sig / sum`.
- **latent**: `lat = bf16(x w_ld^T)`, MXFP4 a block of 32: code = aiter's `_e8m0_roundup(amax, 6)` as the
  kernel computes it (float32 `amax * float32(1/6)`, exponent rounded up when a mantissa bit is set, <= 254),
  E2M1 round-to-nearest-even of `lat / 2^(code - 127)`.
- **shared experts** (rank r's 768 columns): `g, u = bf16(x w_sgu_r^T)`, `h = bf16(SiTU(g, u))`,
  `sd_r = bf16(h w_sd_r^T)`; SiTU = `(2 tanh(g/4)) (1 + tanh(g/2)) 25 tanh(u/25)`.
- **routed experts** (rank r's 384 intermediate columns of each picked expert): `g, u` = the MXFP4 latent
  against the dequantized E2M1 x E8M0 gate / up rows (exact in float64), `v = SiTUv2(g, u)` (the same
  function), MXFP4 of `v` a block of 32 (same rule), `y = v_q w2_r^T`; the rank's partial
  `bf16(sum_k w_k y_k)`.
- **latent norm and up**: `lat_sum = bf16(sum_r partial_r)`, `ln = bf16(rmsnorm(lat_sum) ln_w)`; rank r's
  896 owned hidden columns `bf16(sd_r + ln w_up_r^T)`, the other columns `sd_r`;
  `out = bf16(sum_r final_r) + prefix`.

Returned values are those before their final bfloat16 rounding. The kernel's own routed sum is a bf16
fold in top-k order (a rounding an add); the reference sums the picks in float64 and rounds once (measured
effect below).

## Inputs, shards and outputs

`inputs.py` documents every tensor. Logical tensors are the TP group's: `core` [S, 12288], `w_o`
[7168, 12288], `w_sgu` [12288, 7168] (gate rows then up rows), `w_sd` [7168, 6144], `w_up` [7168, 3584];
`shard_inputs` gives rank r its 1536 core / o_proj columns, its 768 shared gate and up rows and down
columns, its 896 up rows, and the replicated rest (`prefix_in`, the bank, the router, the latent down, the
norms, `guard`). Outputs `out` and `prefix` are replicated (`ranks_agree`).

**Routed experts.** The 896 experts' MXFP4 bytes are 15.7 GB for the group. Instead of drawing them on every
rank, the generator draws one key a tensor (`w13`, `w13s`, `w2`, `w2s` are int64 [1] logical values) and every
byte is a fixed integer hash of (key, expert, row, column) (`k3_numerics.py`: the `lowbias32` mixer):
rank r's shard computes only its slice (2.1 GB of kernel inputs) and the reference only the picked experts,
identically on any device. Codes are uniform E2M1 bytes; scales E8M0 118..122 (2^-9 .. 2^-5), the
repository check's synthetic range. The kernel-facing layout is aiter's A16W4 (`shuffle_weight_a16w4(w13,
16, gate_up=True)`, `shuffle_scale_a16w4(w13s, E, True)`, `shuffle_weight_a16w4(w2, 16, False)`,
`shuffle_scale_a16w4(w2s, E, False)` with w2s's 12 scales a row padded to 16 with 0x7F), transcribed from
ROCm/aiter 652f2d4b (MIT). Checked on CPU: the transcription equals aiter's `shuffle.py` byte for byte, and
the kernel's address arithmetic (`fp4_tile_load`, `w13_scale_word` with the `guint` row-group remap,
`scale_index`) reads back the logical byte at 20,160 sampled lane loads.

Data: core N(0, 1), w_o N(0, 0.001) (the o_proj output adds ~0.11 rms to the prefix), sources N(0, 0.35)
off the planted columns, w_ld / w_sgu / w_up N(0, 0.02), w_sd N(0, 0.008), norm weights N(1, 0.1), ares_qk
N(0, 0.02). The bank's rows 4..7 are allocated and never read, as in production.

**Routing margins.** The top-16 must not flip between reference and kernel, and the score-correction bias
must decide picks (otherwise an implementation that ignores it passes), so the routing is planted. Expert e
owns hidden columns 8e..8e+7; every live seam source carries a level L times the expert's random signs on
the owned columns of the token's planted experts, and the router row reads them at 0.061 / in_nw plus
N(0, 0.001) everywhere, so the router logit is about 1.0..1.4 L. Bias classes: 224 experts +0.12, 134
experts -0.30, the rest U(-0.01, 0.01). Each token plants 12 plain winners (bias ~0, L 2.0..3.1), 4 boosted
winners (bias +0.12, L 1.0..1.15: sigmoid ~0.75, picked only through the bias) and 4 decoys (bias -0.30, L
1.6..1.9: sigmoid ~0.86, above the boosted winners without the bias, below every winner with it). Measured on CPU
(framework seeds, first draw of each case): the gap between the 16th and 17th biased scores is 0.213 /
0.156 / 0.138 / 0.104 at 1 / 2 / 6 / 8 rows (at least 0.139 over 30 draws of 1 row, 0.106 over 4 draws of 8
rows); the float32 evaluation of the reference picks the same experts in every draw; ignoring the bias
changes 4 picks of every token; 16 / 31 / 87 / 120 distinct experts are picked. Consecutive picks can be
close (gap down to 1e-5): their order sets only the kernel's fold order, which the reference does not model.

## Workloads

Monokernel steps are decode and speculative verify only; prefill does not apply. The launch has no context
axis (it starts after the attention) and every other dimension is fixed by the model at TP8, so `rows` (the
compiled width, 1..8) is the only var axis and its >= 10 us production range spans 8x: the task declares the
**production-range span exemption** (`workloads.json` `performance_span`, evidence in
`workload_evidence.json`). Rows are weighted from AgentX (`cc-traces-weka-062126`, per-session in-flight
requests k=1 90.0%, 2 5.4%, 3 2.1%, ...) under two serving modes weighted equally, plain decode (one row a
request in flight) and DSpark verify (8 rows a request, so only one request in flight fits), each over
concurrency points C = 1, 2, 4, 8: rows 1 13.3%, 2 12.8%, 4 10.5%, 8 56.9%, the rest 6.5%. **Derived from
corpus statistics and the serving configuration; not yet confirmed by a live capture.** An end-to-end
AgentX replay of Kimi-K3.1 needs its DFLASH / DSpark serving setup with ReplaySSM
(`--enable-linear-replayssm-spec`, `kimi_k31/spec.py`); it has not been run.

Four performance cases, one compiled build each: rows 1, 2 and 8 (82% of the weighted steps) and 6 (the
non-power-of-two boundary; also Kimi-K3.1's L = 6 verify of one request). Four is the minimum: the
contract needs four performance cases on distinct axis values and this launch compiles per row count, one
more than the two-to-three builds the 10-minute budget guidance asks for. No unaligned case is required
(the axis domain is 8 units).

Estimated baseline time per rank (not measured; ~69 MB of dense weights a rank plus 2.2 MB a distinct picked
expert at ~5 TB/s after the cold-cache flush, plus ~15-25 us of hand-offs over about ten in-kernel
exchanges): r1 ~35-45 us (16 experts), r2 ~45-55 us (32), r6 ~70-80 us (~87), r8 ~85-95 us (~120). All are
above the 10 us floor.

## Comparison

- `prefix`: the default `baseline_relative` bound (it is a bf16 add of the bf16 all-reduce of bf16
  partials; the float32 evaluation of the reference agrees at 100 dB or better and passes the bound).
- `out`: `sqnr` >= 18 dB, a gross-error bound. `out` lies downstream of both MXFP4 quantizations, and they
  are chaotic: a bf16 rounding flip of the seam output `x` (an implementation with float32 arithmetic makes
  a few a row at other elements than the reference) flips bf16 latent values, about 1% of which cross an E2M1
  rounding boundary; one such latent code changes every picked expert's gate / up by ~1%, which re-rounds ~3%
  of the SiTUv2 outputs' E2M1 codes, so that token's whole routed output moves by about -25 dB. Measured on
  CPU (framework seeds; the float32 evaluation of the reference stands in for a correct float32 kernel):
  its `out` SQNR at 1 row is 31 dB in 1 of 30 draws and 54 dB or more (median 68 dB) in the others, 36-66 dB at
  8 rows over 4 draws; a forced latent scale-code flip (an amax crossing a power of two) gives 24-25 dB at 1
  row; an emulation of the kernel's bf16 top-k fold (instead of the reference's single rounding) gives 45 dB.
  Wrong implementations, first draw (1 / 8 rows): the shared experts missing 4.1 / 4.4 dB, the bias ignored
  in the selection (4 picks a token change) 6.7 / 6.7 dB, SiLU(g) u instead of SiTUv2 11.6 / 11.9 dB, no
  latent MXFP4 quantization 15.2 / 15.5 dB, one pick of one token dropped 16.0 / 24.9 dB. **Not caught**:
  skipping the MXFP4 requantization of the SiTUv2 output (20.5 dB), weights renormalized from the biased
  scores (27 dB), one dropped pick at 2 or more rows (19-25 dB), and anything smaller; a per-element or
  per-row bound would reject correct implementations at the rate above. `baseline_relative` uses a per-row
  baseline error and would fail a correct implementation whose flips fall in other rows than the
  baseline's (the Muse Spark layer task measured that). The QA controls (shifted, fp8-rounded, half-written,
  zeroed, stale) are rejected through `prefix` regardless.
- A uniform scaling of the routing weights is invisible by construction: the latent RMSNorm follows the
  weighted sum.

## Unverified

Nothing has run on a GPU: the FlyDSL 0.3.2 compile of the flattened package (in particular
`compat.lds_atomic_or` and the scaled-FP4 conversion intrinsics), the build without the sentinel all-reduces
against the production build, numerics and the bounds against the real baseline, timings, and the 10-minute
budget with four compiled widths of a large persistent kernel (compile time per width is unknown and is the
main risk). The `out` bound was set with a float32 evaluation of the reference standing in for a correct
kernel; the kernel's seam uses hardware `rsq` / `exp2` approximations and may flip more bf16 values of `x`
(more ~25 dB rows; 18 dB keeps ~6 dB below a forced latent scale flip). Per-rank input generation costs
~10 s a rank on the minipc CPU (hashing 2.1 GB of expert bytes), much less on a GPU.

## Framework requests

- A task-declared fill pattern for `symmetric` regions (here: 0xFF for K2's sentinel regions), so the
  production OPT_SENT* all-reduces could be timed.
