# Qwen3.8-Flash-Next verify launch: 3 Gated DeltaNet + 1 QSA layers with MXFP4 MoE (monokernel)

This task is one launch of the Qwen3.8-Flash-Next monokernel from
[sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono). It runs on one MI355X.

- The launch runs four whole decoder layers of a NEXTN speculative-verify step.
- The layers are Gated DeltaNet, Gated DeltaNet, Gated DeltaNet and QSA sparse attention.
- Each layer ends with a 512-expert MXFP4 MoE.
- One persistent kernel of 256 CTAs x 512 threads runs all four layers.
- The stages send values to each other through tagged mailboxes in global memory, not through kernel
  boundaries. The
  [kernel guide](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/sgl_amd_mono/qwen38_flash_next/README.md)
  describes each stage.

A verify step of SGLang's Qwen4-Exp model (TP 1) has 13 launches
([launch plan](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/sgl_amd_mono/qwen38_flash_next/gfx950/runner.py)).
Launches 2 to 11 have exactly this form. Launch 12 also runs the model-level mixer.

## The operator

A step has `requests` requests, and each request has 4 verify tokens. Row `t` is token `t % 4` of request
`r = t // 4`. Its position is `seq_lens[r] + t % 4`. The residual state of a row is 4 hyper-connection
streams of 2560 values. Each layer does these operations:

1. **Attention-side mix** (SGLang `GatedResidual`):
   - `n_b = gemma_rmsnorm(h_b)` for each stream `b`.
   - `x = mean_b(sigmoid(W_up silu(W_down n / 4))_b * n_b)`.
   - The combine gates are `g = 2 sigmoid(W_inj n / 4)`, 4 for each row.
2. **Attention block** on `x`: Gated DeltaNet in layers 0-2 and QSA in layer 3. The next two sections
   describe them.
3. **Combine**: `h_b += g_b * attention_out`.
4. **MoE-side mix**: the same form as the attention-side mix, with its own weights.
5. **MoE**:
   - `softmax` over the 512 router logits. The layer keeps the top 10 and renormalizes them.
   - The `sigmoid` of the last router row gates the shared expert (index 512).
   - Each expert computes `(silu(x W_gate) * (x W_up)) W_down` with MXFP4 weights.
6. **Combine**: `h_b += g_b * moe_out`, with the gates of the MoE-side mix.

### Gated DeltaNet (layers 0-2)

1. `[q | k | v | z] = x W_qkvz` and `[b | a] = x W_ba`.
2. A depthwise causal conv1d of width 4 over `[q | k | v]`, then SiLU. The conv input is the committed window
   of the request, then the 4 tokens of the request.
3. For each head: `q = q / sqrt(|q|^2 + 1e-6) / sqrt(128)` and `k = k / sqrt(|k|^2 + 1e-6)`.
4. The recurrence starts from the committed state `S [V, K]` of the request. It runs for each of the 48 value
   heads and for each token in order. Value head `h` uses key head `h // 3`. With `beta = sigmoid(b)` and
   `decay = exp(neg_a * softplus(a + dt_bias))`, each token does `S = decay S`, `S += (beta (v - S k)) k^T`
   and `o = S q`.
5. `o * rsqrt(mean(o^2) + 1e-6) * norm_w * sigmoid(z)`, then `out_proj`.
6. The state after each token goes to `gdn_issm[layer, request, token]`. The conv window after token `j` goes
   to row `[c, j : j + 3]` of `gdn_iconv`. This layout is the verify scratch of SGLang.

### QSA (layer 3)

1. `qkv_proj` gives 24 heads of query and output gate, and 2 KV heads.
2. Gemma RMSNorm, then Neox RoPE on the first 64 of 256 dimensions. RoPE uses the interleaved MRoPE sections
   11 / 11 / 10 from `cos_sin`.
3. The new K and V entries of the tokens go to the slots `out_cache_loc`.
4. The index projection gives 4 index-query heads of 128 (Gemma RMSNorm, RoPE) and one raw index key. The raw
   key and its position go to the 8-slot ring of the request.
5. When a token completes a block of 4, the layer computes the mean of the 4 keys of the block. It applies
   Gemma RMSNorm to this mean, then RoPE at the first position of the block. The result goes to
   `index_comp[req_to_token[req, first] // 4]`.
6. Row `t` sees the complete blocks `b < (pos + 1) // 4`. These also include the blocks that complete inside
   the step. The score of block `b` is `sum_heads relu(iq . k_b) / sqrt(128)`.
7. Row `t` keeps the top 512 blocks, or all blocks when there are at most 512.
8. Each head attends with scale 1/16 to the tokens of these blocks and to the uncompressed tail
   `4 * nblocks .. pos` of the row.
9. `sigmoid(gate)` gates the output, and `o_proj` projects it.

### Outputs and source

`h_out` holds the streams after layer 3. `h_out_ssq` holds their sums of squares over each group of 16
consecutive elements. The mixer of the next launch reads these sums. The launch also writes the 7 mutable
inputs: `gdn_issm`, `gdn_iconv`, `k_cache`, `v_cache`, `index_ring`, `index_ring_pos` and `index_comp`. It
does not change the committed GDN states or the pool slots of other requests.

The math follows these SGLang files at
[`5d8e90a0f96e6ba35603f9b27f2f1039268c810e`](https://github.com/sgl-project/sglang/tree/5d8e90a0f96e6ba35603f9b27f2f1039268c810e/python/sglang/srt):

- `python/sglang/srt/models/qwen4_exp.py`
- `layers/hyperconnection.py`
- `models/qwen3_5.py` (GDN, gated attention)
- `layers/attention/qsa/` (indexer, compression, top-k and expansion)

`reference.py` implements this math in float64 on the dequantized weights.

## Scenario and layouts

**Model geometry.** The kernel uses the geometry of `amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`
(`HF_CONFIG` / `INDEXER_CONFIG` in
[`qwen38_flash_next/__init__.py`](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/sgl_amd_mono/qwen38_flash_next/__init__.py)):

| Part | Geometry |
|---|---|
| Residual and mixer | hidden 2560, 4 streams, mixer rank 320 |
| GDN | 16 key heads and 48 value heads of 128, conv width 4 |
| QSA | 24 heads, 2 KV heads, head 256, partial rotary 0.25 |
| Indexer | 4 heads of 128, blocks of 4, budget 2048 tokens |
| MoE | 512 experts of 640, top 10, one shared expert of 640 |

**Serving setup.** The task uses the flags of the upstream serving test
([`test/e2e/qwen38_flash_next_server.py`](https://github.com/sgl-project/sgl-amd-mono/blob/52486703fbf27e77f05b573e4d931d4eec87118d/test/e2e/qwen38_flash_next_server.py)):

- NEXTN with topk 1, 3 speculative steps and 4 draft tokens
- CUDA-graph batches of 1 to 8 requests
- a bf16 SSM state (`SGLANG_MAMBA_SSM_DTYPE=bfloat16`)

The index-key ring has 8 slots, as in [sgl-project/sglang#47051](https://github.com/sgl-project/sglang/pull/47051).
The upstream kernel test uses the same pool. All requests are text requests, so the 3 MRoPE axes are equal.

**Kernel layouts.** The kernel reads the layouts that SGLang makes when it loads the weights on gfx950:

- The FP8 projections use AITER `shuffle_weight(w, (16, 16))` and float32 per-channel scales.
- FusedMoE pads the intermediate dimension of the MXFP4 experts from 640 to 768 (gate rows 0..639, up rows
  768..1407). The weights use the (16, 16) shuffle, and the E8M0 scales use the AITER shuffle.
- The bf16 mixer and router matrices use the 16 x 64 MFMA tiles of the kernel.

`inputs.py` draws the logical (row-major) weights that the reference reads. `to_kernel_layout` builds the
kernel layouts outside the timed region.

**Pools.**

- The KV pages of 64 slots are at scattered positions in the pool (`req_to_token`).
- Requests and GDN states use scattered slots.
- The verify scratch rows of request `r` are at row `r`.

### Inputs that keep the discrete choices separated

Two choices inside the launch are discrete:

- the 10 routed experts of each row in every layer
- the 512 blocks of each QSA row

The kernel makes these choices from activations that it quantizes to FP8. If the 10th and 11th router logits
are close, the kernel and the float64 reference can select different experts for some rows. The same applies
to the 512th and 513th block scores. Then a correct candidate can fail. With random weights, such near ties
are common.

So `inputs.py` adds these margins:

- It gives the residual streams one direction that all rows share and one direction for each row. Each
  direction has about a fifth of the energy of a stream.
- For each row, it aligns ten random experts in the router of every layer with the direction of that row.
- The index projection maps the shared direction to one index direction in the non-rotated half of every
  index-query head.
- When a request has more than 512 compressed blocks, exactly 512 of these blocks get keys along the index
  direction. The other blocks get keys against it.

The routes still differ between rows. 32 rows of 10 random experts use about 240 of the 512 experts in each
layer. The kernel serves up to 321 experts in each layer.

Random keys spread the attention of each row over up to 2048 tokens. So the attention output is much smaller
than one V entry. With normal `o_proj` scales, the attention has a small effect on `h_out`. Then a wrong
variant that attends to every block instead of the top 512 has a higher SQNR than the baseline. At
2 x 3000 tokens (2 requests over 3000 context tokens), this variant has 41.6 dB and the baseline has 28 dB.
So `inputs.py` multiplies the `o_proj` scales by 32.

With these inputs, each wrong variant of the reference has a much lower SQNR than the baseline. The table
gives the SQNR of the layers' contribution to `h_out` at 3 x 40001 and 2 x 131072 tokens. At these shapes,
the baseline has 27.7-28.0 dB.

| Wrong variant | SQNR |
|---|---|
| Attend to every block instead of the top 512 | 7.2-7.5 dB |
| The first 512 blocks instead of the top 512 | 4.2-4.6 dB |
| 9 experts instead of 10 | 15.4-16.0 dB |
| No shared expert | 0.8-0.9 dB |
| No RoPE | 11.7-12.6 dB (7.3-7.8 dB on the new K entries) |
| Zero committed GDN state | 3.2-3.4 dB |

## Baseline

`baseline/` is the upstream launch at sgl-amd-mono
[`52486703`](https://github.com/sgl-project/sgl-amd-mono/tree/52486703fbf27e77f05b573e4d931d4eec87118d):
`qwen38_flash_next/gfx950/kernel.py` and the modules that it uses. These changes adapt it to the stateless
call of this task:

- **Arguments instead of a pointer table.** Upstream reads its weights, pools and step tensors through an
  int64 pointer table. SGLang's runner builds this table once per batch size. Here, the base address of each
  tensor is a kernel argument. Each stage reads the address again from the kernarg segment at the point of
  use (`ops.KernArgs`). The slice of a layer in a stacked tensor is a compile-time offset.
- **One zero-filled scratch per call.** Upstream keeps its mailbox scratch between steps. The host advances
  the step epoch after each step, so an old tag never matches. Here, each call allocates its scratch with
  `torch.zeros` (step 0). The scratch holds only the mailboxes of these four layers: 17 MiB for 1 request to
  389 MiB for 8 requests. The memset adds about 2 us (0.4 %) at 1 request and 48 us (3 %) at 8 requests.
  Production pays this cost only once.
- **Constant deadline.** Every wait stops the launch after 20 s. Upstream uses half of the server's watchdog.
- **FlyDSL 0.3.2.** The runtime of the task has FlyDSL 0.3.2. Upstream takes these features from FlyDSL 0.4:
  global loads and stores, the null-page stop and the LDS atomic. `compat.py` builds these features on the
  llvm dialect. The Muse Spark bridge in sgl-amd-mono uses the same method. The task also vendors
  `buffer_ops.py` and `dpp_utils.py` from the FlyDSL kernels of AITER.

`kernel.py` keeps the upstream stages of the other launch kinds: token embedding and n-gram prelude, PLE,
model-level mixer and MTP input fusion. This build never traces these stages. `provenance.modifications` in
`task.yaml` lists every change.

On identical inputs, the vendored launch and the upstream launch give bitwise identical results. This is true
for every output and every mutable input, at 1 x 64, 2 x 3000 and 8 x 1500 tokens. Both launches ran on the
same image with the same bridge. Kernel times (CUDA graph, MI355X):

| Requests | Vendored | Upstream |
|---|---|---|
| 1 | 361 us | 363 us |
| 8 | 1529 us | 1499 us |

The kernel compiles for each request count at its first call. Compilation takes 35 s for 1 request to 125 s
for 8 requests.

The baseline is deterministic for these unpadded batches. Upstream notes that padded graph batches are not
deterministic, because their pad rows race on KV slot 0.

## Accuracy and the bound

Every floating-point output and mutable input uses the default `baseline_relative` bound. `index_ring_pos`
holds integer positions and uses `exact`. The auditor's `QA-BOUND-OVERRIDE` warning names this override.

The SQNR of the baseline against the float64 reference (MI355X). The first four rows are for the 13 workload
cases. The other rows are for the new entries only, at the earlier shapes (1 to 8 requests over 2k to 128k
context tokens).

| Tensor | SQNR |
|---|---|
| `h_out` | 27.6-30.7 dB (the layers' contribution at 2k to 128k: 27.0-28.0 dB) |
| `h_out_ssq` | 33.6-37.1 dB |
| `gdn_issm` | 33.2-33.7 dB |
| `gdn_iconv` | 32.3-32.7 dB |
| New K and V entries | 27.8-28.3 dB |
| New ring entries | 27.5-28.3 dB |
| New compressed keys | 29.6-32.9 dB |

The baseline error comes mainly from the per-token FP8 quantization of the projection inputs (W8A8). The
reference does not quantize these inputs. So an implementation that keeps these activations in higher
precision is more accurate than the baseline, and it passes. The reference passes every bound on every case
when it runs in float32. The calibration probe of the auditor checks this.

## Workloads

The contexts come from AgentX corpus statistics (`semianalysisai/cc-traces-weka-062126` rev `23f152f6`, 98,827
requests). When each request is weighted by its decode steps, the context is 52k tokens at p10, 108k at p25,
220k at p50 and 426k at p75. A live capture did not confirm these shapes.

- 43.6 % of the AgentX decode steps have more than 256k context tokens. The model holds at most 262,144
  positions. So the largest cases have 262,140 context tokens and 4 verify tokens.
- The kernel compiles once for each request count. So the cases use two request counts: 1 request (4 rows, one
  interactive user) and 8 requests (32 rows, the largest CUDA-graph batch).
- The context length of a case is that of its first request. The other requests hold half to all of it.
- Contexts above 2048 tokens select 512 of up to 65,536 blocks. The selection runs over up to 8 index segments
  of 8192 blocks. Contexts above 32,768 tokens also test the merge of the segment winners.

The performance cases (cold-cache CUDA graph, MI355X):

| Requests | Context tokens | AgentX decode steps | Baseline |
|---|---|---|---|
| 1 | 16,384 | 4k to 64k: 9.2 % | 371 us |
| 1 | 65,536 | 64k to 128k: 17 % | 396 us |
| 1 | 131,072 | 128k to 256k: 25.9 % | 401 us |
| 1 | 225,280 | p50 | 405 us |
| 1 | 262,140 | the model limit | 400 us |
| 8 | 65,536 | 64k to 128k: 17 % | 1601 us |
| 8 | 131,072 | 128k to 256k: 25.9 % | 1597 us |
| 8 | 262,140 | the model limit | 1640 us |

The span of `requests * ctx` is 128x. The request count sets most of the time, through the experts that the
rows route to.

The correctness-only cases add these shapes:

- the minimum: 1 request and 16 tokens
- the top-512 boundary: 2049 tokens, so the rows have 512 and 513 complete blocks
- the boundary between one and two index segments: 32,769 tokens, so the rows have 8192 and 8193 blocks
- an unaligned context: 70,001 tokens
- short ragged contexts: 8 requests of 1500 to 3000 tokens. Some requests attend to every block, and the others
  select 512 blocks.

The generator moves context lengths of 2045 to 2047 tokens to 2044. So a request either has at most 512 blocks
in all its rows, or it has at least 512 blocks before its verify window.

One evaluation of one implementation (compile, correctness and performance) takes about 7 minutes on MI355X.
The QA audit evaluates the baseline and a candidate, and it takes 14 minutes.

## Not covered

- The other launch kinds of a verify step:
  - layer 0 with the token embedding and the n-gram prelude
  - layers 1-3 with the PLE block
  - the model-level mixer of the last launch
- The MTP draft launches, padded graph batches, multimodal MRoPE positions, the 4-slot index ring and an f32
  SSM state.
- Request counts 2 to 7. Each request count compiles its own kernel (35 s for 1 request to 125 s for 8), and
  two counts keep one evaluation near 7 minutes.
- The other sgl-amd-mono monokernels. They are separate tasks, and 14 of them run on 4 or 8 GPUs with the
  tensor-parallel collectives inside the launch. The release notes list them.

## Provenance

- Every `baseline/` file keeps its upstream SPDX header (Apache-2.0).
- `ops.py` and `plan.py` adapt ROCm/ATOM code (MIT). `baseline/LICENSE` holds the MIT notice.
- `baseline/LICENSE-NOTICE` is the NOTICE file of sgl-amd-mono.
- `buffer_ops.py` and `dpp_utils.py` come from ROCm/aiter
  [`285579b6`](https://github.com/ROCm/aiter/tree/285579b6211b0da30f35b91a890c29c823597cd1/aiter/ops/flydsl/kernels)
  (Apache-2.0, FlyDSL Project Contributors).
- `inputs.py` ports the `shuffle_weight` and `e8m0_shuffle` permutations of AITER and the bf16 tile packing of
  sgl-amd-mono.
- `reference.py` follows the SGLang model code that [The operator](#the-operator) cites.
