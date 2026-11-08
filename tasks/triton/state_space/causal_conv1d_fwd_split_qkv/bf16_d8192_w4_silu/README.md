# Causal depthwise conv1d forward with q | k | v split (varlen), bfloat16, 8192 channels, width 4, SiLU

The short convolution in front of a Gated DeltaNet prefill (Qwen3-Next): packed variable-length
sequences go through a causal depthwise convolution of width 4 with bias and SiLU, and the 8192
output channels are written directly as three tensors `q` (`n_tokens x 2048`), `k` (`n_tokens x 2048`)
and `v` (`n_tokens x 4096`) instead of one `n_tokens x 8192` tensor. Each sequence starts from its
slot in the conv-state cache when `has_initial_state` is set and from zeros otherwise, and its slot is
overwritten in place with the last 3 inputs it saw (which includes the previous state when the
sequence has fewer than 3 tokens). The cache pool holds twice as many slots as there are sequences;
`cache_indices` assigns distinct slots and the slots no sequence references must stay untouched.

Layout: `x` is `(n_tokens, dim)` with the channels packed `[Q | K | V]`, `conv_states` is
`(cache_lines, 3, dim)`, i.e. channel-last, and the baseline addresses transposed views. Sequence
lengths are random (about 256 tokens on average, `n_seqs = ceil(n_tokens / 256)`, every sequence
non-empty), so short sequences exercise the state-shift paths. Padded sequences
(`cache_indices == -1`) and prefix-cache block states are not part of this scenario.

## Baseline

AITER's 2D-tiled Triton prefill kernel `_causal_conv1d_fwd_split_qkv_tile_kernel`
(`aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/causal_conv1d_fwd_split_qkv.py`, ROCm/aiter
at `652f2d4b768239aefd515c929e84817681ff598d`), vendored unchanged, launched as its public wrapper
`causal_conv1d_split_qkv_triton_tile_fn` does (`aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py`;
`block_m` 64, `block_n` 32, 4 warps, 1 stage, SiLU on, bias on, prefix caching off). The upstream file has
no SPDX or copyright line, so the AITER repository `LICENSE` (MIT) applies; it is copied to `baseline/LICENSE`.

**Origin: vLLM.** The AITER tile kernel is a derivative of vLLM's `_causal_conv1d_fwd_kernel`
(`vllm/model_executor/layers/mamba/ops/causal_conv1d.py` of vllm-project/vllm at
`edbc4969a76b25c531f7b9bb16b79984e65ef023`): the prefix-caching code (`n_block_to_fill`,
`seq_completed_offset`, `last_full_block_token_index`, `B_size`), the state-shift paths (`VAL = state_len -
seqlen`) and the argument names (`conv_state_batch_coord`, `sequence_start_index`) are the same, with an
added 2D `[feature, token]` tile. The AITER file names none of this: it has no header, and its history
(first commit "[Triton] Support GDR prefill and decode (#1594)") does not name it either; the sibling file
`aiter/ops/triton/_triton_kernels/conv/causal_conv1d.py` is likewise header-less and carries the comment "added to
support vLLM larger cache lines". The vLLM file's
header reads `SPDX-License-Identifier: Apache-2.0`, `SPDX-FileCopyrightText: Copyright contributors to the vLLM
project` and "Copyright (c) 2024, Tri Dao. Adapted from
https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py".
`baseline/LICENSE-vllm` carries those lines verbatim and vLLM's Apache-2.0 LICENSE at that commit, so the
baseline is MIT (AITER) and Apache-2.0 (vLLM).

The 1D per-token kernel of the same file is not used. aiter's wrapper builds the (sequence, chunk)
program map on the host (`query_start_loc.diff().to("cpu")`), which a graph-captured call cannot do;
production precomputes it once per batch in the attention metadata. Here a small Triton kernel
(`_build_launch_map_kernel`, adapted from the one of the `causal_conv1d_fwd` task of this repository)
builds the same map on the device and is part of the timed call, so baseline and candidate pay for it
equally. The grid is sized from the shapes (`ceil(n_tokens / 64) + n_seqs` chunk programs per channel
block) and the surplus programs exit through the pad id. The kernel never reads `x` of another
sequence: chunks after the first read the previous three tokens of their own sequence.

## Scenario and config citations

Qwen3-Next-80B-A3B-Instruct, tensor-parallel 1. Source: the public `config.json` of Qwen/Qwen3-Next-80B-A3B-Instruct on Hugging Face (`https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json`): `linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_key_head_dim` 128, `linear_num_value_heads` 32, `linear_value_head_dim` 128. The same values are the defaults of `Qwen3NextConfig` (`vllm/transformers_utils/configs/qwen3_next.py` in vLLM at `edbc4969a76b25c531f7b9bb16b79984e65ef023` and `python/sglang/srt/configs/qwen3_next.py` in SGLang), which corroborates them.

- `linear_conv_kernel_dim` 4 gives `width` 4 (and the 3-token state).
- `linear_num_key_heads` 16 and `linear_key_head_dim` 128 give `key_dim` 16 x 128 = 2048 for q and for k.
- `linear_num_value_heads` 32 and `linear_value_head_dim` 128 give `value_dim` 32 x 128 = 4096.
- `dim` = 2 x 2048 + 4096 = 8192, the same convolution width the AITER test
  `op_tests/test_causal_conv1d_prefill_split_qkv.py` uses (`CONV_DIM = 8192`, `K_DIM = 2048`,
  `V_DIM = 4096`, `WIDTH = 4`).

## Inputs

`x ~ N(0, 1)`, `weight ~ N(0, 0.5^2)`, `bias ~ N(0, 0.1^2)`, conv states `~ N(0, 1)`; `inputs.py` draws the
packed sequence boundaries (random cut points, mixed lengths), distinct cache slots and a
`has_initial_state` flag per sequence (Bernoulli 0.5). Nothing is constant or zero.

## Reference

Float64, a loop over the sequences: prior state (or zeros) concatenated with the tokens, a width-4
shifted-sum with bias, then SiLU, then the last three rows of that concatenation as the new state. It
shares no code with the kernel and is independent of the aiter wrapper.

## Comparison

`q`, `k` and `v` use the default `baseline_relative` bound. The state is made only of copies of
inputs (no arithmetic), so `conv_states` is compared `exact`, which is stricter than the default,
not a relaxation. No bound is overridden.

## Workloads

Total tokens of chunked-prefill batches from 1280 to 131072 (102x span; the upper end is bounded
by the memory of the float64 reference and its compare temporaries at 8192 channels). Per token the
kernel reads 16 KB (`x`) and writes 16 KB (`q`, `k`, `v`), so the perf cases move 42 MB to 4.3 GB and
the smallest is estimated at about 13 us on MI355X (42 MB / 4.5e12 B/s + 4 us). Production prefill
steps are typically 2048 to 16384 tokens, and long-context prefill chunks reach tens of thousands.
Batches below 1280 tokens run under 10 us and are correctness-only, together with the boundary cases 1
(minimum), 3 (the state length), 48 (not a multiple of 64, multiple of 16), 300 (two short sequences) and
1000 (not a power of two nor a multiple of 16).

## Baseline note

The kernel uses `v_rcp_f32` inline assembly for the SiLU, which is AMD-only and valid on gfx950.
