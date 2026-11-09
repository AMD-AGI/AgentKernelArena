# HSTU attention forward, causal, bfloat16, 4 heads x 128, context tokens and candidates

HSTU (Hierarchical Sequential Transduction Unit) attention replaces the softmax of attention by a pointwise
SiLU: `out = silu(alpha * q k^T) v / max_seq_len` per sequence and head, in a jagged batch where sequence
`b` occupies tokens `seq_offsets[b] .. seq_offsets[b + 1] - 1` of `q`, `k` and `v` (`[n_tokens, 4, 128]`).

**Masking.** A sequence is `contextual_seq_len` context tokens, the user's interaction history, then its last
`num_targets[b]` candidate tokens. Each token gets an attention position: all context tokens are position 0,
the first history token is position 0 too, later history tokens count up, and all candidates share the position one
past the last history token (so candidates attend to the whole history and to themselves only). A query
attends to a key whose position is earlier (causal) or that is the query itself (the raw-index diagonal), and a
context token (position 0) additionally attends to every non-candidate token. No attention-span limit is
used. The sum is divided by `max_seq_len`, the longest sequence of the batch (a host integer, as in
production), and the product with `v` takes the SiLU weights rounded to bfloat16. The reference
(`reference.py`) states this in float64 PyTorch from the mask definition of the HSTU PyTorch reference; it
is not a copy of the Triton code.

`sort_by_length_indices` lists the sequences by decreasing length (`torch.sort(descending=True)` in
production); it only orders the kernel's work and does not change the result.

## Production origin of the baseline

AITER's Triton HSTU forward, `triton_hstu_attention_fwd` and `_hstu_attn_fwd`: `ROCm/aiter` at
`652f2d4b768239aefd515c929e84817681ff598d`, files
`aiter/ops/triton/_triton_kernels/attention/hstu_attention.py` and `aiter/ops/triton/attention/hstu_attention.py`
(both start with `Copyright (C) Advanced Micro Devices, Inc. All rights reserved.` and
`Copyright (C) 2024-2026, The vLLM team.` followed by the Apache-2.0 notice, kept verbatim),
`aiter/ops/triton/utils/_triton/kernel_repr.py` (no header upstream; the AITER repository license, MIT,
applies to it), `aiter/ops/triton/utils/common_utils.py` (only `prev_power_of_2`, inlined) and the gfx950
`aiter/ops/triton/configs/hstu_attn/gfx950-HSTU_ATTN_FWD.json` (inlined). The forward kernels are unchanged;
only the forward path is vendored (the backward kernel uses atomics and locks and is not part of this
task). `baseline/LICENSE` is the AITER MIT license, which covers `kernel_repr.py`; `baseline/LICENSE-generative-recommenders`
carries Meta's notice and the Apache-2.0 license text for the HSTU kernels (next section).

The tuned `gfx950` config table is selected by the batch size as upstream does it (`small_batch` below 512 after
rounding down to a power of two, `batch_512`, `large_batch`).

### Origin: Meta's HSTU kernels

The HSTU kernels derive from Meta's Triton HSTU attention in `meta-recsys/generative-recommenders`,
`generative_recommenders/ops/triton/triton_hstu_attention.py`, recorded here at commit
`acc36be02236750cd98d99246f7d7ef1412edd0e`. That file defines `_hstu_attn_fwd_one_block`, `_hstu_attn_fwd_compute`,
`_hstu_attn_fwd` and `triton_hstu_attention_fwd` under the same names, and its `@manual=//triton:triton` markers sit
on the import lines, as in the AITER file. Its header is `# Copyright (c) Meta Platforms, Inc. and affiliates.`
followed by the Apache-2.0 notice. The AITER copies name AMD and The vLLM team instead; they are kept verbatim,
and `baseline/LICENSE-generative-recommenders` carries Meta's header, copied verbatim from that commit, and the
repository's Apache-2.0 LICENSE. The baseline is therefore Apache-2.0 (the HSTU kernels) and MIT (`kernel_repr.py`).

## Scenario and workload origin

bfloat16, 4 heads, query/key/value width 128, causal, `alpha = 1/sqrt(128)`, no context tokens in the timed cases (`contextual_seq_len = 0`, the `STULayerConfig` value), up to 10 candidates
per sequence, no span limit. Citations: `meta-recsys/generative-recommenders`,
`generative_recommenders/dlrm_v3/configs.py` (`get_hstu_configs`: `hstu_num_heads=4`, `hstu_attn_qk_dim=128`,
`hstu_attn_linear_dim=128`), `generative_recommenders/modules/dlrm_hstu.py` (`DlrmHSTUConfig.max_seq_len = 16384`,
`max_num_candidates = 10`, `STULayerConfig(causal=True, max_attn_len=None, contextual_seq_len=0 by default,
sort_by_length=True, attn_alpha=None)`) and `generative_recommenders/modules/stu.py` (`attn_alpha` defaults to
`1.0 / (attention_dim ** 0.5)`). The dataset configs name one contextual feature (`user_id`) for MovieLens-20M, streaming
and KuaiRand-like data and five (`user_id`, `sex`, `age_group`, `occupation`, `zip_code`) for MovieLens-1M
(`contextual_feature_to_max_length`); the timed cases use `contextual_seq_len = 0`, as the `STULayerConfig` above, and correctness cases also run the contextual-prefix path with one and five context tokens. The
mapping from the number of contextual features to `contextual_seq_len` was not read in the model code, so these two values are
exercised for coverage, not claimed as the production value. The config's `hstu_ultra` stack uses `max_seq_len = 1536`, which is the longest sequence of most
performance cases.

`inputs.py` places one sequence of exactly `max_seq_len` tokens at a random position and splits the remaining
tokens over the other sequences at random, so lengths, offsets and the longest sequence move between draws, draws
1 to 10 candidates per sequence (capped so that one history token remains after the context), and sorts the
sequences by decreasing length. `q`, `k`, `v` are standard normal.

Performance cases: 1 to 256 sequences, 4k to 262k packed tokens, `max_seq_len` 1536 (most cases), 4096 and 16384.
Estimated baseline time, from `2 x H x D x sum(L^2)` FLOP for the two causal products (qk and pv, with
the L^2/2 causal pairs counted at 2 FLOP each) at 1 PFLOP/s plus about 4 us (the SiLU needs one exponential per
score, small against that): one 4096-token sequence is 1.7e10 FLOP, 17 us ideal, about 21 us with the fixed cost
(the smallest performance case); eight sequences of at most 1536 tokens (about 1e7 for sum(L^2)) are 1e10 FLOP,
10 us ideal, about 14 us, and the real kernel (64-row query tiles, 4 x 24 programs) is expected to run
well above that; one 16384-token sequence is 2.7e11 FLOP, 0.27 ms ideal; 256 sequences in 262144 tokens are
about 3.7e11 FLOP, 0.37 ms ideal (the largest case). Batches of tens of tokens (one token, 23, 40, 129, 333,
513, 1000, 1537, 2000) and the variants with 1 and 5 context tokens are correctness-only.

## Error bound

Default `baseline_relative`. The baseline's error is dominated by rounding the SiLU weights to bfloat16 before the
second matrix product, and by the `fast_expf`/`fast_dividef` approximation of the SiLU; any other correct tiling
produces independent noise of that size.
