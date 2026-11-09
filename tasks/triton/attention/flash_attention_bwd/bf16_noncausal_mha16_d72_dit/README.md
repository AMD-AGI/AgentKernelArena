# Flash attention backward, non-causal MHA 16 heads, bfloat16, head_dim 72 (DiT-XL/2)

Backward pass of full (non-causal) multi-head attention over the patch tokens of an image, in the BSHD
layout: `q`, `k`, `v`, `o`, `do` are `[batch, seqlen, 16, 72]`, `lse` is `[batch, 16, seqlen]` float32.
The inputs are what the forward pass saves (`o`, the bfloat16 output of `softmax(sm_scale * q k^T) v`, and
`lse`, the natural-log row log-sum-exp of the scaled scores) plus the upstream gradient `do`; the operator
returns `dq`, `dk` and `dv` (bfloat16). The backward recomputes `P = exp(sm_scale * q k^T - lse)`, uses
`delta = rowsum(do * o)` and forms `dS = P * (do v^T - delta)`.

## Production origin of the baseline

Same baseline as `tasks/triton/attention/flash_attention_bwd/bf16_causal_gqa32x8_d128` (AITER's Triton MHA
one-kernel backward, `ROCm/aiter` at `652f2d4b768239aefd515c929e84817681ff598d`, MIT; the file list, kept
headers and modifications are given in that task's README and in `task.yaml`),
here with `bwd_kernel_noncausal` and `causal=False`. `baseline/LICENSE` is the AITER MIT license. The
debug-print blocks of the kernels were deleted (always false, printing is forbidden in a candidate). The head
dimension 72 is not a power of two: the wrapper pads it to 128 (`HEAD_DIM = 128`, `ACTUAL_HEAD_DIM = 72`)
and masks the tail, exercising the kernel's padded-head path. The causal task's README also describes the
kernel's relation to AITER's `flash_attn_triton_amd` backward, the AMD Triton backend that flash-attention also
ships; it applies to this kernel as well.

## Scenario

bfloat16, 16 heads of dimension 72 (hidden size 1152), `sm_scale = 1/sqrt(72)`. Citations: DiT-XL/2
(`facebook/DiT-XL-2-256` and `facebook/DiT-XL-2-512`, `transformer/config.json`: `num_attention_heads` 16,
`attention_head_dim` 72, `patch_size` 2, `sample_size` 32 and 64, 28 layers). With patch size 2, the 32 x 32 and
64 x 64 latents give 16 x 16 = 256 and 32 x 32 = 1024 patch tokens (a 384 px image, 24 x 24 = 576 tokens, is a
correctness case). The var axes are the images per micro-batch (1-128) and the tokens per image (1-1024).

## Inputs

`q`, `k`, `v` and `do` are standard normal bfloat16; `inputs.py` derives `o` and `lse` from `q`, `k`, `v`
(float32 attention, `o` rounded to bfloat16). The reference recomputes the attention and its exact gradient
in float64 from `q`, `k`, `v` and `do`, ignoring the saved `o` and `lse`.

## Workloads

Performance cases run 1 to 128 images at 512 px and 16 to 64 images at 256 px (1k to 131k tokens, a 128x
span). The shortest, one 1024-token image, is 5 x 2 x 1024^2 x 72 x 16 = 1.2e10 FLOP: 12 us at
1 PFLOP/s plus fixed cost, and the baseline is far from peak with 16 heads x 8 key blocks = 128 programs, so it is
estimated at 30 us or more; 16 images of 256 px is also 1.2e10 FLOP. The largest, 128 x 1024, is 1.5e12 FLOP
(about 2 ms at peak). Smaller batches of 256-token images are correctness-only, as are the boundary
cases: two tokens, 17, 127, 129, 200, 333, 576 and 577 tokens and batches of 1-5, and a case with
`sm_scale = 0.125`.

## Why the sequence axis starts at 2

With one token per sequence every query attends to exactly one key, so `P = 1` and `dS = P * (dP - delta) = 0`: no
gradient flows through the scores, `dq` and `dk` are identically zero, and the baseline-relative sanity check has no
signal. At `seqlen = 2` (non-causal) both queries attend to both keys, so `dS` is nonzero everywhere. The axis minimum is therefore 2 and
the boundary case is `c_b1_s2`. Every other correctness case has `seqlen >= 17`.

## Error bound

`baseline_relative` with `k = 3` for all three gradients (default 2). In the CPU emulation described in the
causal task's README (1024 tokens, 8 heads of dimension 72, non-causal, scripted with the framework's
`compare_tensor`), a more accurate implementation that keeps `P` and `dS` in float32 peaked at 0.66x of the
bound at `k = 2`, 0.48x at `k = 3` and 0.38x at `k = 4` with no violating element, and the baseline's SQNR was 52 dB for the
three gradients. `k = 3` keeps a margin of 2x for the accumulation-order variants of a different
tiling while still rejecting a 1% (peak 2.2x) and a 0.3% (peak 1.03x at `k = 4`) scale error and a sequence shift of `dv`.
