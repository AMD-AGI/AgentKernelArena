# Kimi-K3 KDA layer front, TP8 rank slice

The front half of one Kimi-K3 Kimi Delta Attention (KDA) layer for a speculative verify step of one
request, on one tensor-parallel rank of 8 (12 of the 96 KDA heads, head dim 128), from the attention
residual stream to the gated core output that o_proj reads:

1. **AttnRes seam.** Each token's sources are its `nblocks` AttnRes bank rows and its prefix (the
   current block's partial sum). A source's score is the pseudo-query `attn_res_query` against the
   source RMS-normalized with `attn_res_norm` (eps 1e-5); the softmax over the sources mixes them,
   and `input_layernorm` (eps 1e-5) gives x.
2. **in_proj.** x times the merged in_proj weight (6288 rows a rank: q, k, v and the output gate g,
   1536 each; the forget-gate input f_a, 128; beta, 12; 4 pad rows).
3. **Short conv.** A 4-tap causal depthwise conv over the request's committed window (the last three
   conv inputs) and the step's q / k / v rows, then SiLU; q and k l2-normalized per head (eps 1e-6),
   q scaled by 128^-1/2.
4. **Gates.** decay = exp(-5 sigmoid(exp(A_log) (f_b f_a + dt_bias))) per head and key column (the
   lower-bound gate), beta = sigmoid(b).
5. **Delta rule.** From the request's committed fp32 state S[head, v, k], token by token:
   S = S diag(decay); S += ((v - S k) beta) k^T; o = S q. The state after every token is an output.
6. **Gated RMSNorm.** o rmsnorm(o) o_norm sigmoid(g) (eps 1e-5) is `core_out` [rows, 1536].

`reference.py` (float64 PyTorch) states the exact semantics; `task.yaml` lists every input and output.
The operator has no KV cache and no context dependence: its state is the fixed-size recurrent state.
The KDA tasks under `tasks/triton/state_space/kda_*` are chunked prefill; this is the decode / verify
step, a new operator.

### Inputs and outputs that mirror SGLang's state

- `conv_state` (mutable) is K1's conv buffer of the request, token-major `[rows + 3, 4608]` (SGLang's
  pool order): rows 0..2 hold the committed window and rows 3.. the previous step's conv inputs.
  The launch shifts the buffer by one token (final rows 0..2 = initial rows 1..3) and writes the
  step's conv inputs, the bf16 q / k / v projections, to rows 3..; SGLang's hand-back
  (`k1.finish`) reads rows 3.. . A candidate must reproduce both.
- `ssm_states[state_rows[t]]` is the state after token t (SGLang's verify scratch,
  `intermediate_ssm`; the accept step commits the accepted token's row). `state_rows` is ascending
  in production and a random permutation here.
- `ssm_state` (the committed state) and `prefix` are read only.

## Production origin and variant

sgl-amd-mono (`sgl-project/sgl-amd-mono` at `52486703`; SGLang's ROCm images pin `e52803f9`, 11
commits older) serves Kimi-K3 at TP8 with `--speculative-algorithm DSPARK
--speculative-dspark-block-size 7 --enable-amd-monokernel`. Every narrow verify step runs one K1
launch (`kimi_k3/gfx950/attention/kda.py`, host side `kimi_k3/gfx950/k1.py`) on each of the 69 KDA
layers. K1 engages only on verify steps of exactly one request, with `qlen` = the draft tokens (8 for
block size 7). This task is the snapshot build Kimi-K3 serves (`ring` off, initial state read from
the committed pool, per-token states written). Kimi-K3.1 requires `--enable-linear-replayssm-spec`,
under which K1's ring build writes ReplaySSM ring rows instead of per-token states; that variant is
not this task. No KDA layer is a block-write layer (those are layers 12, 24, ..., 84, all full
attention), so the bank is never written. The kernel is gfx950 only (bf16 MFMA 16x16x32).

## Baseline

`baseline/kda.py` is K1's FlyDSL kernel: one persistent grid of 256 CTAs x 512 threads in three
roles. CTAs 0..55 run the seam on 128 hidden columns each (slice, gate, mix, `input_layernorm`),
CTAs 56..252 run the in_proj GEMV (two 16-row groups each, one `mfma_f32_16x16x32_bf16` per 32 K,
weights issued at kernel start), and 96 KDA tasks (12 heads x 8 chunks of 16 state rows) run the
conv, gates, recurrence and gated norm. Stages hand values over through tagged-pair mailboxes in a
global scratch (`sync.py`). The helpers it imports are vendored beside it (`seam`, `dims`, `tag`,
`stamps`, `ops`, `plan`, `sync`, and ROCm/aiter's `buffer_ops` / `dpp_utils` at dbd8bf5bd).

`baseline/kernel.py` is new: `k1.launch`'s host side for one layer. It builds the key `k1.build`
makes (tokens = qlen = rows, `nblocks` bank rows, no delta, no bank write, `init_committed`, eps 1e-5,
lower bound -5), hands the conv buffer over as a view with a zero slot stride (as `k1.py` expands its
own buffer, so the slot number the kernel reads folds onto the request's window), pins `num_acc` at 1
and allocates the mailbox scratch and epoch word zeroed for every call (the runner keeps one scratch
and bumps the epoch once a step on the device; a zeroed scratch makes every tag of the call current
exactly once). The scratch zeroing and one 4-byte fill are inside the timed call, as in the
QA-passed Qwen3.8-Flash-Next task.

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

- `rows`: K1 runs only on verify steps of one request (a step of 2+ requests is 16+ rows and takes
  SGLang's attention), at 8 rows with the documented DSpark block size 7. In AgentX one request is
  in flight for 90% of a session's busy time, so these are the steps the launch serves. The kernel
  is compiled for widths 1..8; rows 1 and 3 are correctness-only cases.
- `nblocks`: a verify step runs K1 once on each KDA layer; layer i reads i // 12 + 1 bank rows
  (`prev_valid_blocks`; layer 92 reads 8), so depths 1..7 each cover 9 of the 69 KDA layers and depth
  8 covers 6.

**Production-range span exemption: used.** Every non-batch dimension is fixed by the model and the
kernel has no context axis. Observed range: rows 8..8, nblocks 1..8 (observed var-axis span 8x);
every production point reads the 90 MB in_proj weight and takes the baseline well over 10 us, so the
perf range equals the observed range. Performance cases r8_nb1, r8_nb4, r8_nb6, r8_nb8 span 8x in
var-axis size (`workload_evidence.json` holds the source, frequencies and derivation). The run time
is nearly the same at every point: the bank adds at most 1 MB to the 98 MB a launch moves.

Estimated baseline time (not measured): about 98 MB moved (90 MB weight, the 6.3 MB of per-token
states, the committed state, bank and prefix), 20 us at 5 TB/s, plus the seam -> GEMV -> recurrence
hand-off chain: 25-35 us per performance case. An evaluation builds six kernels per implementation
(rows 1, 3, 8 x the cases' bank depths); expected wall time 4-8 min, dominated by FlyDSL compiles
(not measured). Dropping the two correctness-only cases saves two builds per implementation.

## Comparison

`core_out` and `ssm_states` use `baseline_relative` with `k` = 4 and `max_violation_fraction` =
0.001; `conv_state` uses the default (`k` = 2). The bounds were calibrated on CPU with an emulation of
the baseline's arithmetic (float32 accumulation, bf16 at the kernel's hand-offs: x, the projections,
the conv output, f_b f_a and o) against the float64 reference, through the evaluator's comparison
(cases r8_nb4, r8_nb8, r6_nb1, r3_nb2, r1_nb1):

- a candidate that rounds at the same points but accumulates differently (float64) flips a few
  bf16 hand-offs, and every flipped q / k / decay element moves a whole state column: at `k` = 2 it
  fails `ssm_states` (peak 1.9-5.3x the bound on 0.009-0.034% of elements) and `core_out` (peak
  1.1-1.4x on up to 0.043%); at `k` = 4 its peak is at most 2.7x on at most 0.011% of elements,
  which the 0.1% violation allowance (16x peak cap) admits;
- a candidate with no intermediate rounding peaks at 0.33x (`core_out`) and 0.005x (`ssm_states`);
- a bf16 recurrent state fails (`ssm_states` 8-19% of elements over the bound at `k` = 4, peak
  17-49x), as do a bf16 decay (3.5-13%, peak 15-17x), fp8 activations, a missing conv window, a zero initial
  state, beta = 1, no decay, and states written in token order instead of `state_rows` (all far
  outside the bound).

Emulated baseline SQNR against the reference: 47-49 dB (`core_out`), 52-60 dB (`ssm_states`).

## Unverified

Nothing here has run on a GPU. Unverified: the vendored kernel tracing and compiling on the FlyDSL
0.3.2 runtime (checked against the 0.3.2 sources by inspection only: `fx.union`, `SharedAllocator`,
`s_waitcnt`, `buffer_load` of bf16), its numerics against the reference (the bounds above come from a
CPU emulation, not the kernel), the timing estimates and the compile time. Risks for the GPU audit:
rows 1 and 3 are compiled in production (SGLang warms every width) but sgl-amd-mono's K1 test checks
only 8 rows; the persistent grid needs all 256 CTAs resident at once (one a CU on MI355X).
