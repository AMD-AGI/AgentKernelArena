# Kimi-K3.1 MLA (DSA) projection front, TP8 rank slice

The projection front of one Kimi-K3.1 MLA layer (the full-attention layers, with a DSA indexer) for a
speculative verify step, on one tensor-parallel rank of 8 (12 of the 96 query heads): every
projection SGLang's MLA forward and its indexer read, from the attention residual stream.

1. **AttnRes seam.** Each token's sources are its `nblocks` AttnRes bank rows and its prefix. A
   source's score is the pseudo-query `attn_res_query` against the source RMS-normalized with
   `attn_res_norm` (eps 1e-5); the softmax over the sources mixes them, and `input_layernorm`
   (eps 1e-5) gives x.
2. **Projections of x.** `fused_qkv_a` (2112 rows: q latent 1536, kv latent 512, rope key 64), the
   attention output gate `g_proj` (1536 rows a rank) and the DSA indexer's `wk` (128) and
   `weights_proj` (32). `latent` is the raw kv latent and rope key.
3. **Latent norms.** q latent and kv latent RMSNorms (eps 1e-6, as aiter's fused_qk_rmsnorm); the
   normed kv latent is `k_nope`.
4. **Projections of the normed q latent.** `q_b` (12 heads x (128 nope | 64 rope)) and the indexer's
   `wq_b` (32 heads x 128).
5. **Absorb.** `q_nope_absorbed` = q_nope W_UK per head into the 512-wide kv_lora space (SGLang's
   `q_nope_out`, from the bf16 `w_kc`).

Every output is bfloat16. `reference.py` (float64 PyTorch) states the exact semantics; `task.yaml`
lists every input and output. The operator has no KV cache and no context dependence: RoPE, the KV
write, the indexer's top-k and the sparse attention run after it on SGLang's path.

## Production origin and variant

sgl-amd-mono (`sgl-project/sgl-amd-mono` at `52486703`; SGLang's ROCm images pin `e52803f9`, 11
commits older) serves Kimi-K3.1 at TP8 with DSpark (block size 7, 8 rows a request) or DFLASH (the
documented launch: `--speculative-num-draft-tokens 6`, 6 rows). A narrow step, a verify of one
request (`kimi_k31/spec.py`: L = 6 or 8), runs Kimi-K3's launches; on each of the 23 DSA layers that
is this launch (`kimi_k3/gfx950/attention/mla_pre.py`, built by `kimi_k3/spec.py` `_mla_build`,
launched by `_mla_core`). This task is the build of a layer that is not a block-write layer (16 of
the 23; layers 12, 24, ..., 84 also store the prefix as a bank row, a separate build) with the
absorb on: the checkpoint's attention weights are bf16, so `w_kc` is bf16 with `w_scale` 1 and the
launch writes `qn`, which SGLang's ROCm MLA forward reads. Without DCP only (under DCP SGLang
computes q itself). The kernel is gfx950 only (bf16 MFMA 16x16x32).

## Baseline

`baseline/mla_pre.py` is sgl-amd-mono's FlyDSL kernel: one persistent grid of 256 CTAs x 512 threads.
CTAs 0..55 run the seam on 128 hidden columns each; stage 1 runs the 238 16-row GEMV groups of x
(weights issued at launch, a group a CTA on CTAs 56..255 and the 38 extra groups on seam CTAs); every
CTA then sums the q groups' squares in group order, normalizes the q latent into LDS and runs stage 2
(q_b and wq_b, 400 groups); 48 CTAs run the absorb (a head and 128 kv_lora columns each) once their
head's q_b groups are out, and CTA 255 the kv norm. Stages hand values over through tagged-pair
mailboxes in a global scratch (`sync.py`). The helpers it imports are vendored beside it (`gemv`,
`seam`, `dims`, `tag`, `stamps`, `ops`, `plan`, `sync`, and ROCm/aiter's `buffer_ops` / `dpp_utils`
at dbd8bf5bd).

`baseline/kernel.py` is new: `_mla_core`'s host side for one layer. It builds the key `_mla_build`
makes (`rows` tokens, `nblocks` bank rows, no bank write, `qn` on, eps 1e-5 / 1e-5 / 1e-6 / 1e-6),
hands the absorb its weight as `[heads x kv_lora, nope]` (the input is already transposed as
`_mla_build` transposes SGLang's `w_kc` once at load) and allocates the mailbox scratch and epoch word
zeroed for every call (the runner keeps one scratch and bumps the epoch once a step on the device; a
zeroed scratch makes every tag of the call current exactly once). The zeroing is inside the timed
call, as in the QA-passed Qwen3.8-Flash-Next task.

FlyDSL 0.3.2: the kernel uses no API newer than the runtime's apart from `ops.global_load` (written on
llvm-dialect ops in the vendored `ops.py`, the bridge of `muse_spark/gfx950/compat.py`) and
`stamps.stamp_flush` (`fx.generic_store`, only traced in a timeline build, which the task never makes).
`ops.py`, `buffer_ops.py` and `dpp_utils.py` are the files of the QA-passed
`tasks/flydsl/other/qwen38_flash_next_decoder/verify_gdn3_qsa1` baseline. `sync.py` keeps only the
scratch-space mailbox and the unguarded wait (these launches open no guard scope). Every modification
is listed in `task.yaml` and in each file's header. Licenses: `baseline/LICENSE` (sgl-amd-mono and
aiter Apache-2.0; ROCm/ATOM MIT for `ops.py`, `plan.py`, `sync.py` and `stamps.py`).

## Workloads

Derived from the serving configuration, the model's layer structure and the AgentX corpus statistics
(`semianalysisai/cc-traces-weka-062126` rev 23f152f6); not yet confirmed by a live capture.
Monokernel steps are verify only; prefill shapes do not apply.

- `rows`: the narrow launches serve verify steps of one request (2+ requests are 12+ rows and take
  the wide path), 8 rows with DSpark block 7 and 6 with DFLASH; no usage split between the two is
  known, so they are weighted equally. In AgentX one request is in flight for 90% of a session's busy
  time. The kernel is compiled for widths 1..8; rows 1 and 3 are correctness-only cases.
- `nblocks`: a verify step runs this launch on each of the 16 non-block-write DSA layers; layer i
  reads i // 12 + 1 bank rows (`prev_valid_blocks`), two layers a depth from 1 (layers 4, 8) to 8
  (layers 88, 92).

**Production-range span exemption: used.** Every non-batch dimension is fixed by the model and the
kernel has no context axis. Observed range: rows 6..8, nblocks 1..8 (observed var-axis span 10.7x);
every production point reads the 76 MB of projection weights and takes the baseline well over 10 us,
so the perf range equals the observed range. Performance cases r8_nb1, r8_nb8, r6_nb1, r6_nb8 (the
corners) span 10.7x in var-axis size (`workload_evidence.json` holds the source, frequencies and
derivation). The run time is nearly the same at every point: the bank adds at most 1 MB to the 77 MB a
launch moves.

Estimated baseline time (not measured): about 77 MB moved (fused_qkv_a 30 MB, g_proj 22 MB, wq_b 13
MB, q_b 7 MB, wk / weights_proj 2 MB, w_kc 1.6 MB, bank and prefix), 15 us at 5 TB/s, plus the seam ->
stage 1 -> norm -> stage 2 -> absorb hand-off chain: 22-32 us per performance case. An evaluation
builds six kernels per implementation (rows 1, 3, 6, 8 x the cases' bank depths); expected wall time
4-8 min, dominated by FlyDSL compiles (not measured). Dropping the two correctness-only cases saves two
builds per implementation.

## Comparison

`q`, `index_q` and `q_nope_absorbed` use `baseline_relative` (`k` = 2) with `max_violation_fraction`
= 0.001; the other outputs use the default bound. The bounds were calibrated on CPU with an emulation
of the baseline's arithmetic (float32 accumulation, bf16 at the kernel's hand-offs: x, the fused_qkv_a
latent, the normed q latent and q) against the float64 reference, through the evaluator's comparison
(rows / depths 8 / 4, 8 / 8, 6 / 1, 3 / 2, 1 / 1):

- a candidate that rounds at the same points but accumulates differently (float64) flips a few bf16
  hand-offs. `latent`, `gate`, `index_k`, `index_weights` (after the x hand-off) and `k_nope` (after
  x and the latent) stay at 0.4-0.8x the default bound; `q` and `index_q` (after x, the latent and
  the normed q latent) reach 0.59-0.95x; `q_nope_absorbed` (after q as well) reaches up to 1.12x on
  0.002-0.006% of its elements, which the 0.1% violation allowance (16x peak cap) admits;
- a candidate with no intermediate rounding peaks at 0.33x;
- a q_b GEMV that adds its K chunks in bfloat16 fails (`q` 2.3-2.6% of elements over the bound, peak
  2.1-2.6x; `q_nope_absorbed` 3.4-4.2%), as do fp8 activations (62-77% of elements, peak 10-26x), a
  q latent norm without its weight and a GEMV that drops a 128-wide K chunk (far outside the bound).

Emulated baseline SQNR against the reference: 48.4-52.8 dB.

## Unverified

Nothing here has run on a GPU. Unverified: the vendored kernel tracing and compiling on the FlyDSL
0.3.2 runtime (checked against the 0.3.2 sources by inspection only: `fx.union`, `SharedAllocator`,
`s_waitcnt`, `buffer_load` / `buffer_store` with zero-range descriptors), its numerics against the
reference (the bounds above come from a CPU emulation, not the kernel), the timing estimates and the
compile time. sgl-amd-mono's own test (`test/kernels/test_kimi_k3_mla_pre.py`) checks rows 1, 6 and 8
(the absorb at 6 and 8) by cosine only. The persistent grid needs all 256 CTAs resident at once (one a
CU on MI355X).
