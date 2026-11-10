# Qwen3.8-Flash-Next verify launch: 3 Gated DeltaNet + 1 QSA layers with MXFP4 MoE (monokernel)

One launch of the Qwen3.8-Flash-Next monokernel of
[sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) on one MI355X: four whole decoder
layers (Gated DeltaNet, Gated DeltaNet, Gated DeltaNet, QSA sparse attention, each followed by a 512-expert
MXFP4 MoE) of a NEXTN speculative-verify step, in one persistent kernel of 256 CTAs x 512 threads. A verify
step of SGLang's Qwen4-Exp model (TP 1) runs 13 such launches; launches 2 to 11 have exactly this form and
launch 12 adds the model-level mixer ([launch plan](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/sgl_amd_mono/qwen38_flash_next/gfx950/runner.py),
[kernel guide](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/sgl_amd_mono/qwen38_flash_next/README.md)).
The stages hand values to each other through tagged mailboxes in global memory instead of kernel boundaries;
the kernel guide describes them stage by stage.

## The operator

A step has `requests` requests of 4 verify tokens; row `t` is request `t // 4`, token `t % 4`, at position
`seq_lens[r] + t % 4`. The residual state of a row is 4 hyper-connection streams of 2560. Layer `i` does:

1. **Attention-side mix** (SGLang `GatedResidual`): `n_b = gemma_rmsnorm(h_b)` per stream,
   `x = mean_b(sigmoid(W_up silu(W_down n / 4))_b * n_b)` and the combine gates
   `g = 2 sigmoid(W_inj n / 4)` (4 per row).
2. **Attention block** on `x`:
   - **Gated DeltaNet** (layers 0-2): `[q | k | v | z] = x W_qkvz`, `[b | a] = x W_ba`; a depthwise causal
     conv1d of width 4 over `[q | k | v]` (the request's committed window, then its 4 tokens) and SiLU;
     per head `q = q / sqrt(|q|^2 + 1e-6) / sqrt(128)`, `k = k / sqrt(|k|^2 + 1e-6)`; for each of the 48
     value heads (key head `h // 3`) and token in order, with `beta = sigmoid(b)` and
     `decay = exp(neg_a * softplus(a + dt_bias))`: `S = decay S`, `S += (beta (v - S k)) k^T`,
     `o = S q`, from the request's committed state `S [V, K]`; then
     `o * rsqrt(mean(o^2) + 1e-6) * norm_w * sigmoid(z)` and `out_proj`.
     The state after every token goes to `gdn_issm[layer, request, token]` and the conv window after every
     token to `gdn_iconv` (row `[c, j : j + 3]` after token `j`), as SGLang's verify scratch.
   - **QSA** (layer 3): `qkv_proj` gives 24 heads of query and output gate and 2 KV heads; Gemma RMSNorm and
     Neox RoPE on the first 64 of 256 dimensions (interleaved MRoPE sections 11 / 11 / 10, from `cos_sin`);
     the tokens' K / V rows go to `out_cache_loc`. The index projection gives 4 index-query heads of 128
     (Gemma RMSNorm, RoPE) and one raw index key, which goes to the request's 8-slot ring with its position;
     when a token completes a block of 4, the mean of the block's 4 keys, Gemma-normalized and rotated at the
     block's first position, goes to `index_comp[req_to_token[req, first] // 4]`. Row `t` sees the complete
     blocks `b < (pos + 1) // 4` (those completed inside the step included) scored by
     `sum_heads relu(iq . k_b) / sqrt(128)`, keeps the top 512 (all of them when there are at most 512) and
     attends, per head, to their tokens plus its own uncompressed tail `4 * nblocks .. pos` with scale
     1/16; the output is gated by `sigmoid(gate)` and projected by `o_proj`.
3. **Combine**: `h_b += g_b * attention_out`.
4. **MoE-side mix** (same form, own weights), then the **MoE**: `softmax` over the 512 router logits, the
   top 10 renormalized, plus the shared expert (index 512) gated by `sigmoid` of the last router row; every
   expert is `(silu(x W_gate) * (x W_up)) W_down` on MXFP4 weights.
5. **Combine** into the streams.

`h_out` is the streams after layer 3 and `h_out_ssq` their sums of squares over every 16 consecutive
elements (the next launch's mixer reads them). The other written tensors are the mutable inputs above; the
committed GDN states and every pool row of other requests are left alone. The math follows SGLang's
`python/sglang/srt/models/qwen4_exp.py`, `layers/hyperconnection.py`, `models/qwen3_5.py`
(GDN, gated attention) and `layers/attention/qsa/` (indexer, compression, top-k and expansion) at
[`5d8e90a0f96e6ba35603f9b27f2f1039268c810e`](https://github.com/sgl-project/sglang/tree/5d8e90a0f96e6ba35603f9b27f2f1039268c810e/python/sglang/srt);
`reference.py` implements it in float64 on the dequantized weights.

## Scenario and layouts

- Model geometry: `amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8` as the kernel is built for it
  (`HF_CONFIG` / `INDEXER_CONFIG` in
  [`qwen38_flash_next/__init__.py`](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/sgl_amd_mono/qwen38_flash_next/__init__.py)):
  hidden 2560, 4 streams, mixer rank 320; GDN 16 key / 48 value heads of 128, conv width 4; QSA 24 heads,
  2 KV heads, head 256, partial rotary 0.25; indexer 4 heads of 128, blocks of 4, budget 2048 tokens;
  512 experts of 640, top 10, one shared expert of 640.
- Serving setup: NEXTN with topk 1, 3 steps, 4 draft tokens, CUDA-graph batches of 1 to 8 requests and a
  bf16 SSM state (`SGLANG_MAMBA_SSM_DTYPE=bfloat16`), the flags of the upstream serving test
  ([`test/e2e/qwen38_flash_next_server.py`](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/test/e2e/qwen38_flash_next_server.py));
  the 8-slot index-key ring of [sgl-project/sglang#47051](https://github.com/sgl-project/sglang/pull/47051)
  (the upstream kernel test's pool); text requests, so the 3 MRoPE axes agree.
- Kernel-facing layouts are SGLang's after loading on gfx950: AITER `shuffle_weight(w, (16, 16))` FP8
  projections with float32 per-channel scales; FusedMoE MXFP4 experts with the intermediate dimension padded
  from 640 to 768 (gate rows 0..639, up rows 768..1407), (16, 16)-shuffled, and AITER-shuffled E8M0 scales;
  the bf16 mixer and router matrices in the kernel's 16 x 64 MFMA tiles. `inputs.py` draws the logical
  (row-major) weights the reference reads and `to_kernel_layout` builds these layouts outside timing.
- Pools: 64-slot KV pages scattered over the pool (`req_to_token`), requests and GDN states at scattered
  slots, the verify scratch rows of request `r` at row `r`.

### Inputs that keep the discrete selections well separated

Two decisions inside the launch are discrete: each row's 10 routed experts in every layer and each QSA
row's 512 blocks. The kernel computes them from activations it quantizes to FP8, so if the 10th and 11th
router logits (or the 512th and 513th block scores) were close, the kernel and the float64 reference would
pick different experts or blocks for some rows, and a correct candidate could fail. With random weights such
near ties are the rule. `inputs.py` therefore gives the residual streams one direction shared by every row
and one per row (each about a fifth of a stream's energy), points ten random experts of every layer's
router at each row's own direction, points the non-rotated half of every index-query head at the shared
direction, and gives exactly 512 of a request's compressed blocks (when it has more) keys along the index
direction and the others keys against it. The routes still differ per row: 32 rows of 10 random experts
cover about 240 of the 512 experts per layer (the kernel serves up to 321).

Random keys spread each row's attention over up to 2048 tokens, so its output is far smaller than a value
row: with ordinary `o_proj` scales, attending to every block instead of the top 512 changes `h_out` less than
the baseline's own error does (41.6 dB against 28 dB at 2 x 3000 tokens). `o_proj`'s scales therefore carry a
factor of 32. With these inputs, wrong variants of the reference differ from it by much more than the baseline does
(SQNR of the layers' contribution to `h_out`, 3 x 40001 and 2 x 131072 tokens): attending to every block
7.2-7.5 dB, the first 512 blocks instead of the top 512 4.2-4.6 dB, 9 instead of 10 experts 15.4-16.0 dB,
no shared expert 0.8-0.9 dB, no RoPE 11.7-12.6 dB (7.3-7.8 dB on the new K rows), a zero committed GDN
state 3.2-3.4 dB; the baseline is at 27.7-28.0 dB.

## Baseline

`baseline/` is the upstream launch at sgl-amd-mono
[`52486703`](https://github.com/sgl-project/sgl-amd-mono/tree/52486703fbf27e77f05b573e4d931d4eec87118d)
(`qwen38_flash_next/gfx950/kernel.py` and the modules it uses), adapted to this task's stateless call:

- **Arguments instead of a pointer table.** Upstream reads its weights, pools and step tensors through an
  int64 table SGLang's runner builds once per batch size. Here every tensor's base address is a kernel
  argument, re-read from the kernarg segment where a stage needs it (`ops.KernArgs`), and a layer's slice of
  a stacked tensor is a compile-time offset.
- **One zero-filled scratch per call.** Upstream keeps its mailbox scratch and advances the step epoch on the
  host after every step, so a stale tag never matches. A call here allocates its scratch with `torch.zeros`
  (step 0), and only the mailboxes these four layers use are laid out (17 MiB for 1 request to 389 MiB for 8).
  The memset adds about 2 us (0.4 %) at 1 request and 48 us (3 %) at 8 requests; production pays it once.
- **Constant deadline.** Every wait stops the launch after 20 s (upstream: half of the server's watchdog).
- **FlyDSL 0.3.2.** The task's runtime ships FlyDSL 0.3.2; `compat.py` builds the global loads and stores,
  the null-page stop and the LDS atomic that upstream takes from FlyDSL 0.4 on the llvm dialect, as
  sgl-amd-mono's own Muse Spark bridge does. AITER's FlyDSL `buffer_ops.py` and `dpp_utils.py` are vendored.

`kernel.py` keeps the upstream stages of the other launch kinds (token embedding and n-gram prelude, PLE,
model-level mixer, MTP input fusion); this build never traces them. `task.yaml` lists every change. On
identical inputs the vendored launch matches the upstream launch (run on
the same image through the same bridge) bit for bit in every output and written pool, at 1 x 64, 2 x 3000 and
8 x 1500 tokens; its kernel time matches too (361 vs 363 us at 1 request, 1529 vs 1499 us at 8, CUDA graph,
MI355X). The kernel compiles for each request count at its first call (35 s for 1 request to 125 s for 8).

The baseline is deterministic for these unpadded batches; upstream notes that padded graph batches are not,
because their pad rows race on KV slot 0.

## Accuracy and the bound

Every floating output and pool uses the default `baseline_relative` bound. `index_ring_pos` holds integer
positions and compares `exact` (the auditor's `QA-BOUND-OVERRIDE` warning names it). The baseline's own SQNR against the float64 reference (MI355X, the workloads' shapes): `h_out`
27.3-30.3 dB (the layers' contribution 27.0-28.0 dB), `h_out_ssq` 33.2-36.9 dB, `gdn_issm` 33.2-33.6 dB,
`gdn_iconv` 32.3-32.5 dB, the new K / V rows 27.8-28.3 dB, the new ring rows 27.5-28.3 dB and the new
compressed keys 29.6-32.9 dB. Its error is dominated by the per-token FP8 quantization of the projection
inputs (W8A8), which the reference does not perform; an implementation that keeps those activations in
higher precision is more accurate than the baseline and passes. The reference evaluated in float32 passes
every bound on every case (the auditor's calibration probe).

## Workloads

Performance cases are verify steps of 1, 2, 4 and 8 requests over 2k to 128k context tokens (the longest
request; the others hold half to all of it). The kernel serves up to 8 requests (32 rows); contexts above
2048 tokens select 512 of up to 32768 blocks, over up to 4 index segments of 8192 blocks, and those above
32768 tokens exercise the merge of segment winners. The span of `requests * ctx` is 128x. The baseline takes
0.37 ms (1 request) to 1.61 ms (8 requests) per case (cold-cache CUDA graph, MI355X); the request count sets
most of it, through the experts the rows route to. Correctness-only cases
add the minimum (1 request, 16 tokens), odd request counts (3, 5) and contexts that are not a multiple of 4,
16 or 64. Context lengths of 2045 to 2047 tokens are moved to 2044 by the generator, so a request either has
at most 512 blocks in all its rows or at least 512 before its verify window.

## Not covered

- The other launch kinds of a verify step (layer 0 with the token embedding and the n-gram prelude, layers 1-3
  with the PLE block, the model-level mixer of the last launch), the MTP draft launches, padded graph
  batches, multimodal MRoPE positions, the 4-slot index ring and an f32 SSM state.
- The other sgl-amd-mono monokernels run their layer across 2 to 8 GPUs with the tensor-parallel all-reduces
  inside the launch, which a single-GPU kernel task cannot express.

## Provenance

- Every `baseline/` file keeps its upstream SPDX header (Apache-2.0). `ops.py` and `plan.py` adapt
  ROCm/ATOM (MIT); `baseline/LICENSE` holds that notice and `baseline/LICENSE-NOTICE` is sgl-amd-mono's
  NOTICE file. `buffer_ops.py` and `dpp_utils.py` are from ROCm/aiter
  [`285579b6`](https://github.com/ROCm/aiter/tree/285579b6211b0da30f35b91a890c29c823597cd1/aiter/ops/flydsl/kernels)
  (Apache-2.0, FlyDSL Project Contributors).
- `inputs.py` ports AITER's `shuffle_weight` / `e8m0_shuffle` permutations and sgl-amd-mono's bf16 tile
  packing; `reference.py` is written from SGLang's model code cited above.
