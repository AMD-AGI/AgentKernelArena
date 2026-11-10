# Muse Spark K3: MXFP4 experts to the next attention input, one TP8 / EP8 rank with TP experts (8 GPUs)

One persistent 256-CTA launch per rank takes a DFlash2 verify step of `rows` rows (rows / 8 requests of 8 draft
tokens) from K2's routing and expert input to the next layer's attention input: the expert input quantized to
MXFP4, gate / up on the scaled E2M1 x E2M1 MFMA against the rank's slices of every picked expert,
bf16(silu(gate) up) re-quantized to MXFP4, the mids exchanged with every rank, the down projection's output
columns of the rank, the finalize (per-(token, slot) post norm, score-weighted slot sum) over values every rank
pushes, the unpooler (4096 -> 8192, the rank's rows pushed to every rank) and the FFN boundary. The ranks
exchange data inside the kernel through each other's symmetric buffers over XGMI; there is no host-side
collective. This is a multi-GPU task (`distributed.world_size: 8`).

**Model support is semi-experimental.** Muse Spark is registered in sgl-amd-mono's `SPECS` but absent from
the repository README's model table; it landed in one PR (#26), has no end-to-end server test, and its default
served widths are 8 and 16 rows (`gfx950/config.py`: "32 and 64 rows are under test"). The repository's 8-rank
K3 test (`test_muse_spark_k3_tp8.py`) covers TP experts at 8 and 16 rows only.

## Production origin

sgl-project/sgl-amd-mono at `52486703fbf27e77f05b573e4d931d4eec87118d` (Apache-2.0; `common/` files adapted
from ROCm/ATOM, MIT): `sgl_amd_mono/muse_spark/gfx950/build_k3.py::build_k3` with its stages
`stages/ffn.py::run_k3` and `stages/mxfp4.py`, `config.py`, `layout.py`, `device.py`, the FlyDSL 0.3.2 bridge
`compat.py` and `common/{guard,epoch,ops,plan,sync,mx}.py`. SGLang's ROCm images pin sgl-amd-mono `e52803f9`
(11 commits older); the baseline is vendored from `52486703`. The vendored modules are those of
`tasks/flydsl/other/muse_spark_layer/tp8`.

The build is `MuseMonoRunner.k3_kernel(rows)` with the runner's default environment: TP experts
(`SGLANG_AMD_MONO_MUSE_TPX=1`: each rank holds its 1/8 slices of all 256 experts, built from EP's 32 local
experts a rank with `weights.tp_expert_copies`), a bf16 unpooler (`..._MUSE_FP8_DENSE` off), a middle layer
(`last` = 0). The standalone launch has no pre-quantized expert input (`xq_pre`) or dynamic ug queue
(`ug_dyn`): those exist only in the merged layer launch. This is the runner's non-merged path
(`SGLANG_AMD_MONO_MUSE_LAYER=0`, or a step the merged launch does not take); by default the same stage code runs
inside the whole-layer launch (`tasks/flydsl/other/muse_spark_layer/tp8`). `baseline/kernel.py::run` passes
`MuseMonoRunner.launch`'s K3 arguments (`local_start` = 32 rank, which TP experts do not read).

Vendoring changes (no kernel or stage code changed; details in `task.yaml` provenance): one flat package with
relative imports; the 0.3.2 bridge as plain functions called by `ops.global_ptr`/`global_load`,
`epoch.gstore`, `guard.stop_launch` instead of patching modules at import; frozen dataclasses / a slotted
class instead of `msgspec`; no `plan.kernel_sources` source digest; a local `kernel_signature`; environment
reads replaced by their defaults; `buffer_ops.py` from ROCm/aiter 652f2d4b and `dpp_utils.py` from ROCm/FlyDSL
dd837334.

**Scratch placement.** The runner keeps K3's mailbox scratch in ordinary device memory and moves the epoch word
on after every step, so mailbox tags never repeat. Here the epoch is a fixed input (`guard`, epoch 1) and the
scratch region sits in the framework's `symmetric` buffer after the peer regions (`symmetric_bytes` =
47,190,016 peer bytes + 14,163,968 scratch bytes), which the framework zeroes before every call. That buffer is
uncached memory (the same allocation as the runner's `PeerBuffer`), so the baseline's local scratch traffic is
uncached where production's is not; the cost is expected to be small but is unmeasured.

## Semantics (reference.py, float64)

The compact-A4 chain of the ROCm model K3 replaces, as `test/kernels/test_muse_spark_k3.py::_reference` and
its 8-rank form state it, over the logical (whole) experts, keeping the MXFP4 quantization and the materialized
bfloat16 tensors exactly and everything else in float64:

- per pick: gu = MXFP4(xin) . dequant(w13[e])^T (aiter's `_mxfp4_quant_op` rule on the bf16 input: the E8M0
  scale from the 32-block max rounded at mantissa bit 21, E2M1 round half to even, saturating at 6);
  mid = MXFP4(bf16(silu(gate) up)); y = bf16(mid . dequant(w2[e])^T); v = bf16(y pen_w);
  out += score v / sqrt(mean(v^2) + 1e-8);
- part = bf16(out); branch = bf16(part w_unpool^T); post = bf16(branch pfw); nb = bf16(post rstd) (eps 1e-8);
  res' = alpha res + beta nb (the mutable stream); attn_out = res' rstd npw (eps 1e-5).

## Inputs, shards and outputs

`inputs.py` documents every tensor. The generator returns logical tensors (every expert whole: 6.4 GiB of
MXFP4 weights); `shard_inputs` gives rank r the TP slices of every expert (`weights.tp_expert_copies` /
`test_muse_spark_k3_tp8.py`: gate rows 512 r .., up rows 4096 + 512 r .., down rows 512 r .., with their
scales; 0.8 GiB a rank) and the replicated rest (the unpooler whole: the kernel reads its rank's rows). Both
outputs are replicated: `attn_out` and the mutable `res`, in `ranks_agree` (every rank finalizes the whole rows
from the same pushed values in the same order). `guard` is the runner's epoch buffer (epoch 1, a 30 s deadline,
clean error words): read-only unless a wait fails, so a failed wait fails correctness.

Data: the routing is an input (K2 makes it), so no discrete decision is taken inside the launch: 8 distinct
experts a token uniformly at random (59 and 229 distinct experts in cases r8 and r64; all 256 are reachable),
scores U(0.5, 0.75) normalized to sum 1 (K2's normalized sigmoids); xin unit normal; MXFP4 codes uniform bytes
with E8M0 scales 118..121 (2^-9 .. 2^-6); res unit normal; norm weights 1 + N(0, 0.1); alpha U(0.5, 1), beta
U(0, 0.5); the unpooler N(0, 0.02).

**Why the scale range matters.** The mids are re-quantized to MXFP4, so a tiny difference in gate / up can flip
an E2M1 code and turn an accumulation-order difference into a visible one. Every E2M1 x E2M1 product is a
multiple of 2^-2 of its two block scales; with the input's per-block scales within a few powers of two and the
weights' within 2^3, every gate / up partial sum fits float32's 24-bit significand, so the gate / up dot
products are exact in float32 whatever the summation order, and the mids differ from the reference's only
where the kernel's approximate sigmoid moves bf16(silu(gate) up) across a rounding boundary. Measured on CPU
(float32 evaluations of the reference with two summation orders, the framework's seeds of r8 and r64): every
gate / up dot product was exact in float32 in both orders, and the mids differed from the float64 reference's in
0 (r8) and 1 (r64, of 2.1 million) elements.

## Workloads

Phase: decode / verify only (the monokernels serve no prefill). K3's work does not depend on the context (no
context axis exists; it grows with the distinct experts the step picks), and Muse Spark fixes every other
dimension at TP8 / EP8, so the task uses the **production-range span exemption** (`workloads.json`
`performance_span`, `workload_evidence.json`):

- Observed production range: rows 8..64, the compiled widths 8, 16, 32, 64 (1, 2, 4, 8 requests; a smaller
  batch is assumed to run padded to the next width, as spec.py's check mode treats rows with
  `out_cache_loc <= 0` as padding). Default serving uses 8 and 16; 32 and 64 need
  `SGLANG_AMD_MONO_MUSE_WIDTHS=8,16,32,64`.
- Perf range: rows 8..64, span **8x** (all four widths estimated >= 10 us).
- Frequency (AgentX `cc-traces-weka-062126`, the per-session in-flight request distribution at concurrency 1,
  2, 4, 8, equal weight per concurrency point, padded to the widths): 8 rows 22.6%, 16 rows 21.8%, 32 rows
  21.4%, 64 rows 34.3%. **Derived from corpus statistics; not yet confirmed by a live capture.** The routing is
  uniform at random; production routing (with its load-balancing bias) is not captured either.

Performance cases: r8, r16, r32, r64 (one per width; perf-case span 8x). Correctness-only: r24 (the
non-power-of-two boundary; rows 8 is the minimum). Five compiled widths (8, 16, 24, 32, 64): four performance
cases on the one var axis need four distinct widths, and the non-power-of-two boundary a fifth, more than the
two to three the 10-minute budget guidance asks for.

Estimated baseline time per rank (bytes at ~5 TB/s plus ~15 us of in-kernel hand-offs; 3.34 MB of TP-expert
slices a distinct expert, ~57 / 102 / 163 / 222 distinct experts at 8 / 16 / 32 / 64 rows, the unpooler's
8.4 MB of rank rows): r8 ~55 us, r16 ~85 us, r32 ~125 us, r64 ~165 us.

## Comparison

- `attn_out`: the default `baseline_relative` bound. With the exact gate / up products above, a correct
  implementation with float32 accumulation in another order stays within it: the two float32 evaluations, each
  as the other's baseline, passed at r8 and r64 with peaks of 0.43-0.50 of the bound.
- `res`: `sqnr` >= 60 dB. The stream is fp32 but carries beta times the bf16-rounded post norm nb, so wherever
  one implementation's nb rounds differently from the reference's (a one-ulp difference of a bf16 tensor) its
  error is thousands of times the other's fp32-level error at that element: the float32 evaluations failed
  `baseline_relative` on 3-6e-5 of the elements with peaks of 400-6800x the bound, while their SQNR was 90 dB
  (r8) and 77.6 dB (r64). `attn_out` sees the same branch at bf16 resolution.
- Controls (CPU): dropping one slot of every token gives 17.5 dB on both outputs and fails `attn_out` on 89% of
  its elements; losing one rank's 512 intermediate columns of gate gives 17.8 dB; one wrong expert in one
  token's weakest slot (r8) gives 23.9 dB and fails `attn_out` on that token's row (11% of the elements).

## Unverified

Nothing has run on a GPU: FlyDSL 0.3.2 compile of the flattened package (the vendored modules are the
whole-layer task's, whose GPU QA was pending when this was written), whether the scaled MFMA's accumulation is
exact as the CPU emulation assumes, numerics and the bound against the real baseline (the kernel's sigmoid and
its in-kernel orders), timings, the bitwise rank agreement, the standalone TP-experts K3 at 24 / 32 / 64 rows on
eight ranks (the repository runs 8 and 16), the uncached-scratch cost, compile time per width (the main risk for
the 10-minute budget, with five widths), the CPU / GPU cost of drawing 6.4 GiB of logical experts a draw on
every rank, and the workloads (corpus statistics, no live capture).
