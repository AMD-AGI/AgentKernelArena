# Qwen3.8-Flash-Next decoder launch: verify step, layers gdn, gdn, gdn, qsa

One launch of the Qwen3.8-Flash-Next (SGLang `Qwen4-Exp`) monokernel for a NEXTN target-verify step at
tensor parallelism 1: R requests x 4 tokens (row t = request t // 4, draft step t % 4), four decoder
layers in the production middle plan `gdn,gdn,gdn,qsa`. SGLang runs this launch 11 times per verify
step (launches 2-12, layers 4-47 of 48); this task is one such launch on synthetic weights.

Per layer, on the 4 hyper-connection residual branches (2560 each):

1. attention-side mixer (RMS norm of the branches, low-rank 320 down + SiLU, up + sigmoid gates, gated
   branch mean; combine gates `2 sigmoid(W_inj Hn / 4)`);
2. a Gated DeltaNet layer (W8A8 FP8 `in_proj_qkvz` / `in_proj_ba` with per-token E4M3 activation
   quantization, causal 4-tap conv1d over the committed conv state, l2-normalized q / k, the gated delta
   rule over the request's 4 tokens from its committed SSM state with the state and conv window written
   after every token, gated RMS norm, W8A8 `out_proj`), or the QSA layer (W8A8 q / gate / k / v and
   indexer projections, Gemma q / k norms, NeoX RoPE on 64 of 256 dims, the 8-slot index-key ring and
   the group compression, index logits over every complete 4-token block, exact top-512 block selection,
   sparse attention with GQA 24 / 2, sigmoid output gate, W8A8 `o_proj`);
3. the combine into the residual branches, the MoE-side mixer, the router (top-10 of 512 with
   renormalized softmax, plus the shared expert at index 512 with a sigmoid gate) and the MXFP4 experts
   (width 640, SiLU-gated), then the MoE combine.

`reference.py` (float64 PyTorch) documents the exact semantics; `task.yaml` lists every input and
output. Outputs are the new residual streams, their sum-of-squares partials (what the next launch's
first mixer reads), and every state write of the launch: the GDN verify scratch (SGLang's
`intermediate_ssm` and the deduplicated `intermediate_conv_window`, rows 0..R-1), the new QSA K / V
rows of the step's tokens, the compressed index key of the 4-position group the step completes, and
the index-key ring with its RoPE positions (mutable).

## Baseline and provenance

`baseline/layers.py` is sgl-amd-mono's FlyDSL launch kernel (`sgl_amd_mono/qwen38_flash_next/gfx950/kernel.py`
at 52486703, Apache-2.0; SGLang's ROCm images pin e52803f9, 11 commits older): one persistent grid of
256 CTAs x 512 threads whose stages hand values over through tagged-pair mailboxes in a global scratch.
The helpers it imports are vendored beside it (`ops`, `guard`, `plan`, `layout`, `config`, `device`,
and ROCm/aiter's `buffer_ops` / `dpp_utils` at dbd8bf5bd). `baseline/kernel.py` is new: it builds the
int64 pointer table of the launch (one row a layer plus the global row) as `runner.VerifyRunner` does
for one middle launch, allocates the scratch and the epoch / guard block, and launches on the current
stream. Modifications (listed in `task.yaml` provenance and in each file's header):

* The AKA runtime ships FlyDSL 0.3.2. `fx.generic_load` / `fx.generic_store` (FlyDSL >= 0.3.4) are
  written on llvm-dialect ops the way `sgl_amd_mono/muse_spark/gfx950/compat.py` bridges them (without
  patching `flydsl`, which the candidate rules forbid), and the radix-select histogram's
  `fx.atomic_add` (FlyDSL >= 0.3.3) goes through `ops.lds_atomic_add` (`llvm.atomicrmw` on 0.3.2).
* No `msgspec`, environment or source-file reads: build keys are dataclasses, the JIT key's source
  digest is a constant tag, the fault injection is removed.
* **Output buffers for the new K / V rows and the completed compressed key.** Production writes them
  into SGLang's KV cache at `out_cache_loc` and into the compressed-key pool. Here they go to `k_new` /
  `v_new` ([rows, 2, 256]) and `index_key_new` ([R, 128]): the launch never reads those rows back from
  memory (attention and the index logits take the step's own K / V / key from its mailboxes), and
  comparing multi-gigabyte mutable caches on every timed sample would dominate the evaluation. The KV
  cache, compressed keys, `req_to_token` and committed GDN states are read-only inputs; the bytes
  written are the same as in production.
* Scratch and epoch: the runner keeps one scratch and advances the epoch after each step, so stale
  mailbox pairs never carry the current tag. `run()` allocates them zeroed per call instead (a memset
  of 49 MB at R = 1 and 662 MB at R = 8, inside the timed graph); every call's tag is then 1.

## Inputs

Random weights in SGLang's post-load gfx950 layouts, as sgl-amd-mono's kernel test draws them, at the
model's real dimensions (512 + 1 MXFP4 experts a layer, about 6.4 GB of expert weights for the four
layers). The layout transform (`to_kernel_layout`) applies AITER's `shuffle_weight(w, (16, 16))` to the
FP8 projections and MXFP4 experts, `e8m0_shuffle` to the expert scales and sgl-amd-mono's
`packing.pack_bf16` to the mixer and router weights; these permutations were checked on the CPU to be
inverted by sgl-amd-mono's `packing.unshuffle16` / `unshuffle_moe_expert` and to equal `pack_bf16`.

Request 0 has the case's context; the other requests draw theirs uniformly in [ctx / 2, ctx]. KV slots
are 64-slot pages in random order (page 0 unused), so `req_to_token` is a real page table.

The two discrete decisions are planted with clear margins (see the `inputs.py` docstring):

* routing: 10 large per-row code channels in the residual streams and one router weight per expert on
  its code channel. On the CPU (R=1 at ctx 1, 2049, 4093; R=8 at ctx 3000) the 10th routed logit is
  >= 4.0 and the 11th <= 1.03 in every row and layer; 32 rows route to roughly 240 experts a layer;
* top-512 blocks: a shared large channel feeds index queries of a known direction (heads 1 / 3 negate
  heads 0 / 2, so a block's logit is `(|q0 . k| + |q2 . k|) / sqrt(128)`), the compressed keys are 512
  planted winners and losers of equal norm, and the key the step compresses is a loser. On the CPU the
  512th score is >= 11.0 and the 513th <= 4.8 (ratio >= 2.37).

In all four CPU cases the routes and block selections were unchanged when every mixer output was
perturbed by 1 % relative noise and rounded to bf16, far more than the kernel's own rounding.

## Workloads

Derived from AgentX corpus statistics (`semianalysisai/cc-traces-weka-062126` rev 23f152f6, 98,827
requests; decode-step-weighted context p10 52K, p25 108K, p50 220K, p75 426K) and the serving
configuration (R <= 8, max_position_embeddings 262,144); **not yet confirmed by a live capture**.
Contexts are clipped at 262,140 (+ 4 verify positions): the 43.6 % of AgentX decode steps above 256K
(30 % of requests exceed 262,144) cannot use this model. Monokernels serve decode / verify only; prefill
shapes do not apply. Two compiled widths, R = 1 (4 rows, the interactive single request) and R = 8 (32
rows, the largest compiled batch): every R is a separate FlyDSL build of a 4-layer unrolled kernel.
Performance: R = 1 at 16K, 64K, 128K, 220K and the cap; R = 8 at 64K, 128K and the cap (var-axis product
span 128x). Correctness only: context 1 (no complete block), 2049 (512 / 513 blocks), 32769 (one / two
selection segments), 70001 (unaligned) and R = 8 at 3000 (short ragged contexts).

Estimated baseline time from bytes moved at ~5 TB/s: about 0.9 GB a launch at R = 1 (~41 experts x 4
layers, FP8 projections, mixers, GDN states, scratch memset), ~180 us plus the latency of ~60
mailbox-synchronized stages; about 4.4 GB at R = 8 (~240 experts a layer, 0.38 GB of GDN states and
verify scratch, the 0.66 GB scratch memset), ~0.9 ms. Every performance case is far above 10 us.

## Comparison

Every floating output uses `baseline_relative` with `k: 4` and `max_violation_fraction: 0.01`; the ring
positions compare exactly. The reference is float64 with exact E4M3 activation quantization; the
baseline rounds intermediate values to bf16 (mixer outputs, projections, conv outputs, residual
streams, attention probabilities) and quantizes per token from those bf16 values, so a correct
implementation that rounds differently flips a different ~1-2 % of the E4M3 activation codes. Its error
then has the baseline's size but not its element pattern, and with the default `k: 2` a few percent of
its elements would exceed twice the baseline's error. `k: 4` with 1 % of elements allowed over the bound
(none beyond 16x) admits that, as for the repository's other W8A8 / MXFP4 MoE tasks.

## Unverified

Nothing here has run on a GPU: FlyDSL compilation of the vendored kernel on FlyDSL 0.3.2 (including the
llvm-dialect bridges and the atomic), the redirected stores, the pointer-table wiring, the baseline's
SQNR against the reference, compile time per width and timing. The CPU checks covered the contract, the
baseline's candidate-rule audit, the inputs and reference (CPU float64, shapes, finiteness), the layout
permutations against sgl-amd-mono's packing code, the scratch layout and the planted margins. Compile
time is the main budget risk: if the two widths do not compile within the 10-minute evaluation budget,
the smallest compliant set keeps R = 1 only (performance at 2560, 16K, 64K, 128K and the cap context, a
102x span; correctness at 1, 2049 and 70001).
