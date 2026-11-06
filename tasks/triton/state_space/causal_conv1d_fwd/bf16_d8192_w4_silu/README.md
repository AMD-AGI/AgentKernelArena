# Causal depthwise conv1d forward (varlen), bfloat16, 8192 channels, width 4, SiLU

Packed variable-length sequences (continuous batching) go through a causal depthwise
convolution of width 4 with bias and SiLU, as in the short convolution of Mamba2 and
gated-delta-net layers (8192 channels is the conv width of Qwen3-Next at tensor-parallel 1).
Each sequence starts from its slot in the conv-state cache when `has_initial_state` is set and
from zeros otherwise, and its slot is overwritten in place with the last 3 inputs it saw
(which includes the previous state when the sequence has fewer than 3 tokens). The cache
pool holds twice as many slots as there are sequences, so the slots no sequence references
must stay untouched; `cache_indices` assigns distinct slots.

Layout: `x`/`out` are `(n_tokens, dim)` and `conv_states` is `(cache_lines, 3, dim)`, i.e. the
channel-last layout vLLM's kernel requires, viewed transposed by the baseline. Sequence lengths
are random (about 256 tokens on average, `n_seqs = ceil(n_tokens / 256)`), so short sequences
exercise the state-shift paths. Padded sequences (`cache_indices == -1`) and prefix-cache
block states are not part of this scenario.

## Baseline

vLLM's Triton `_causal_conv1d_fwd_kernel` (Apache-2.0; derived from Tri Dao's causal-conv1d),
vendored from `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` at the recorded revision.
The kernel body is unchanged except that the CUDA-only `launch_pdl` hooks are removed. vLLM's
wrapper builds the (sequence, chunk) program map with a host round trip, which a graph-captured
call cannot do; production normally precomputes it in the attention metadata. Here a small Triton
kernel (`_build_launch_map_kernel`) builds the same map on the device and is part of the timed
call, so the baseline and any candidate pay for it equally. The grid is sized from the shapes
(`ceil(n_tokens / 8) + n_seqs` chunk programs) and surplus programs exit through the pad id.

## Comparison

`out` uses the default `baseline_relative` bound. The state is made only of copies of inputs
(no arithmetic), so `conv_states` is compared `exact`; this is stricter than the default,
not a relaxation.

## Workloads

Total tokens of chunked-prefill batches from 1280 to 131072 (102x span; the upper end is
bounded by the memory of the float64 reference and its compare temporaries at 8192 channels). Perf cases move 33 MB to
4.3 GB. Batches below 1280 tokens run under 10 us and are correctness-only, together with the
boundary cases 1 (minimum), 3, 48 and 1000 (not a power of two nor a multiple of 16).
