# GLM-5.3-Flash sparse-attention (DSA) decoder layer, TP4

One decode step of one GLM-5.3-Flash sparse-attention layer on each of 4 tensor-parallel ranks, as the
single persistent launch `flash_ffn_kernel` of sgl-amd-mono builds it with `FlashBuild(dsa=True)`
(256 CTAs x 512 threads a rank; the TP reductions run inside the kernel through symmetric peer memory).

## Operator

Per step row (one decode request; pad rows fill the compiled width):

1. **Attention-side mHC**: the previous layer's owed FFN post and comb applied to its 4 residual streams,
   24 mix projections, a 20-iteration Sinkhorn gate, RMSNorm (input_layernorm).
2. **Projections**: q_a | kv_a (FP8, 128x128 block scales), indexer wk | weights_proj and the kpool gate
   (bf16), q_a / kv_a RMSNorms, the latent written to the MLA cache, the index key LayerNorm.
3. **kpool**: a row whose token completes a 4-token pool compresses it (softmax(gate + ape)-weighted key,
   bf16, Hadamard-128, bf16, UE8M0 FP8) into the index cache; its (k, gate) go to the request's ring.
4. **Sparse MLA**: q_b (FP8, this rank's 16 heads), W_UK absorption, the index query (Hadamard, UE8M0 FP8,
   head weights), index scores `scale(p) * sum_h w_h relu(q8_h . k8_p)` of the row's complete pools, exact
   top-512 selection (rows with more than 512 complete pools; others attend to every token), split-softmax
   NoPE attention over the selected 2048 tokens plus the row's tail, W_UV, o_proj (FP8), summed over ranks.
5. **FFN**: FFN-side mHC (-> `res_out`, `post_out`, `comb_out`, owed to the next layer), the bf16 router,
   top-8 of 288 by sigmoid + bias (weights renormalized x 2.5) plus the shared expert, MXFP8 activations,
   MXFP4 experts (this rank's 512 of 2048 intermediate columns), SwiGLU clamped at 10, MXFP8 mid, the
   MoE TP reduce -> `x_out`.

The float64 definition is `dsa_layer_math.py`; `reference.py` returns its outputs and final caches.

## Multi-GPU contract (`distributed`, world size 4)

* The generator draws the **logical** layer (all 64 heads, the whole 2048-wide expert intermediate)
  once per seed; `shard_inputs` gives rank r heads `16r .. 16r + 15` (q_b rows, kv_b rows, W_UK^T,
  o_proj columns and their block scales) and expert intermediate columns `512r .. 512r + 511` (w13 gate
  and up rows, w2 columns, scale columns); everything else (activations, mHC, q_a | kv_a, indexer,
  router, caches, tables, guard) is replicated. `to_kernel_layout` then applies the production load-time
  layouts (AITER (16, 16) shuffle of the FP8 projections and MXFP4 experts, e8m0_shuffle of the scales,
  the router's MFMA packing).
* `rank_expected`: every rank writes the whole outputs (the TP sums run in place, in rank order) and its
  own replica of the caches, so each rank's expected tensors are the logical outputs. `ranks_agree`
  lists the outputs and the kpool rings (the large caches are checked per rank against the reference,
  not gathered to rank 0).
* Reserved inputs: `peers` and `symmetric` are the kernel's `peers` / `sym` arguments; `rank` its rank.
  `symmetric_bytes = (rows + 1) * 4 MiB` holds the kernel's exchange regions and guard flags
  (`layout.layout`'s symmetric part, 1 MiB a row) and, past them at 4 KiB alignment, its cross-CTA
  scratch mailboxes (3.6 MiB a row with `max_pools = 262144`). Production allocates the scratch as a
  separate uncached buffer of the same kind (`hipExtMallocWithFlags(..., uncached)`); placing it in the
  framework-zeroed symmetric buffer keeps mailbox words out of the compared state.
* `guard` [8] int32 is the launch's guard block: epoch 0 (every mailbox tag is 1, which zeroed buffers
  never hold), a 10 s wait deadline, zero error words. It is a read-only input: a wait that times out
  writes an error word (and stops the launch), which the read-only check would also catch.

## Reference and its assumptions

There is no upstream torch reference for the DSA attention, the kpool compression or the mHC gate
(`test/kernels/glm53_flash/check_layer.py` checks routing and the key selection against torch, the rest
only for bitwise self-consistency); the reference was derived from the kernel source. Assumptions, each
read from the kernel:

* mHC: `R'[j] = sum_i comb[i][j] R[i] + post[j] x`; mixes over the stream-major 4 x 4096 row with
  `rinv = rsqrt(mean(R'^2) + 1e-5)`; pre = sigmoid(...) + 1e-6, post = 2 sigmoid(...), comb = softmax over
  the last axis + 1e-6, a column normalization, then 19 (row, column) normalizations, each sum + 1e-6.
* The FP8 block projections are exactly dequantized weights times bf16 activations (no activation
  quantization). FWHT is Sylvester order scaled by float32(1/sqrt(128)), its result rounded to bf16 before
  UE8M0 (scale = 2^ceil(log2(max(amax, 1e-4) / 448))). Index head weight =
  `iw * q_scale * 128^-0.5 * 32^-0.5` with `iw` unrounded. No RoPE (qk_rope_head_dim 0).
* MXFP8 (FFN input and mid): E8M0 = exponent of float32(amax * float32(1/448)) rounded up when its
  mantissa is nonzero, clamped to 1..254; codes E4M3 RNE. MXFP4: E2M1 low nibble first.
* Router keys use the bias rounded to bf16; route weights use the unrounded sigmoid scores.
* Pad rows (position -1) write no cache, attend to nothing (x_attn = 0) and still run the FFN; their
  outputs are compared. A scored row attends to the 512 pools' tokens and then `4 * bound .. pos`; when
  its token completes a pool, that token is attended only through its pool's selection.
* Precision: float64 everywhere except the MLA cache, held in float32 (exact for its bf16 rows) because a
  float64 copy of a long-context cache is large; quantizations are emulated from the float64 value
  rounded once to float32. The pool compression also rounds to bf16 where the kernel stores bf16.

## Planted margins (inputs.py)

The two discrete decisions must not flip between the reference and a correct implementation:

* **Top-512 pool selection.** The indexer head weights are made positive (weights_proj rows solved so
  that `weights_proj . a` is U(0.5, 2) on the rows' own attention inputs), so every unscaled pool score is
  nonnegative; each scored row's pools then get power-of-two scales placing its 512 best pools in
  [2T, 4T) and every other pool below T / 4, using the reference's own index query (a factor-8 gap; the
  pool this step completes keeps the kernel's own scale, and T puts it at 3T, a winner).
* **Router top-8.** The router rows are solved so that each row's logits are U(3, 6) on 8 random experts
  and N(0, 0.25) clipped to +-0.75 elsewhere (bias U(-0.1, 0.1)); the 8th and 9th keys differ by > 0.07.
* **Pool compression.** A completing request's three earlier ring entries share one key on the 1/64 grid
  (|k| in [0.25, 4)) and gates 16 above the completing token's, so the pooled key is that key exactly in
  bf16 and its Hadamard is exact in float32: the compressed FP8 codes and scale are bit-exact, which is why
  `index_cache` is compared `exact`. (A float32 emulation of the kernel's compression, hardware-exp
  style, matched the reference bitwise in 2000 of 2000 perturbed trials on CPU.)

These are test-data structures (production shapes and magnitudes, synthetic weight structure), as the
MoE tasks plant router winners. On CPU, both framework seeds of the checked cases kept a router gap
>= 0.099 and a pool-score ratio (512th / 513th) >= 8.0; perturbing the index query, head weights, the
attention output and the FFN input by bf16-scale noise flipped no decision.

## Comparison bounds

* `x_out`: `baseline_relative` with `k: 4`, `max_violation_fraction: 0.01`. The MoE output goes through
  two MXFP8 quantizations (the FFN input and the mid); a correct implementation whose bf16 rounding of
  the FFN input or of the mid differs from the baseline's moves individual E4M3 codes by one step and
  occasionally a whole E8M0 block scale by 2x, which the baseline's own element error does not cover
  (the `fused_moe_a8w4` task uses the same bound for the same reason).
* `index_cache`: `exact` (packed FP8 codes and float32 scale bytes), achievable because of the planted ring.
* Everything else: the default.

## Workloads

Decode steps only (the monokernel serves decode and MTP verify; prefill shapes do not apply). Rows =
requests at the compiled widths 2 (1-2 requests) and 8 (5-8 requests); two compiled widths keep the
evaluation at two compiles per implementation. Contexts are AgentX decode-step-weighted quantiles
(semianalysisai/cc-traces-weka-062126 rev 23f152f6: p10 52K, p25 108K, p50 220K, p75 426K, p90 632K,
plus 16K as the low end); they are **derived from corpus statistics and not yet confirmed by a live
capture**. The leading request holds `ctx` tokens, the others ctx/2 to ctx. Correctness adds the first
token (ctx 1), the attend-everything limit 2051, the first scored size 2052 (with expert scales that
make the SwiGLU clamp fire), odd widths, a short context and the 1M cap of the compiled launch
(`max_pools = 262144`; AgentX's longest context is 990K).

## Baseline

`baseline/flash_kernel.py` is sgl-amd-mono's `glm53_flash/gfx950/kernel.py` at 52486703 (SGLang ROCm
images pin e52803f9, 11 commits older) with: package-relative imports of the vendored helpers (ops,
guard, sync, epoch, plan, config, layout, tiles; AITER's buffer_ops and FlyDSL's dpp_utils), a fixed
source digest, the sparse-attention argument table passed as 27 kernel arguments instead of a host-built
device table (the candidate rules forbid `torch.tensor` in a call), and FlyDSL 0.3.2 stand-ins for
`fx.memory_fence`, pointer `fx.atomic_add` and `fx.generic_store` (`compat.py`, llvm dialect, as
sgl-amd-mono's own Muse Spark 0.3.2 bridge does, but without patching `flydsl.expr`). `kernel.py::run` is
`FlashLayerOp.launch` for one sparse-attention layer with SGLang's in-place layouts (FP8 projections
AITER-shuffled, the shared expert as fused slot 288). Licenses: `baseline/LICENSE` (Apache-2.0 and the
upstream NOTICE with ROCm/ATOM's MIT text).

## What is unverified

Nothing here has run on a GPU: the baseline's compile on FlyDSL 0.3.2 (the compat stand-ins, the
`tab_*` argument change, `fx.coop`, permlane and scaled-MFMA availability), its numerics against the
reference, the timing (each performance case is estimated at 25-70 us from bytes moved, above the 10 us
floor), the multi-rank evaluator path (`distributed`) and the 10-minute budget (two compiles of a
5,000-line persistent kernel per implementation). The CPU checks run the generator, the reference,
shard, layout and expected for every rank.
