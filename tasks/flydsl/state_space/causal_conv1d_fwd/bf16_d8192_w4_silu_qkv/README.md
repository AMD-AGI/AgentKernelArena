# Causal depthwise conv1d forward (varlen) with split q/k/v output, bfloat16, 8192 channels, width 4, SiLU (FlyDSL baseline)

Packed variable-length sequences (continuous batching) go through the causal depthwise convolution of width 4 with
bias and SiLU that precedes the gated delta rule in Gated DeltaNet layers (prefill). The 8192 channels are the
concatenation `q | k | v` of 2048 + 2048 + 4096 channels and the result is written directly split into
`query [n_tokens, 2048]`, `key [n_tokens, 2048]` and `value [n_tokens, 4096]`. Each sequence starts from its slot
of the conv-state cache when `has_initial_state` is set and from zeros otherwise, and its slot is overwritten in
place with the last 3 inputs it saw (which include the previous state when the sequence has fewer than 3 tokens).
The cache holds twice as many slots as there are sequences, so slots that no sequence references must stay
unchanged. This is the operator of `tasks/triton/state_space/causal_conv1d_fwd/bf16_d8192_w4_silu`, except that
AITER's FlyDSL kernel fuses the q/k/v split into the store and takes its inputs in the layout described below.

## Layout and the kernel's launch schedule

- `x` is channel-major `[dim, n_tokens]` (the kernel indexes `channel * stride + token`, so tokens must be
  contiguous). The sibling task passes the transpose of a `[n_tokens, dim]` tensor instead.
- `conv_states` is declared `[cache_lines, 3, dim]`; the baseline hands the kernel the `[cache_lines, dim, 3]` view
  (channel-contiguous), the layout of AITER's own test of this kernel.
- The kernel runs one program per (sequence, 64-token chunk). AITER takes that schedule from its prefill metadata
  (`CausalConvPrefillMetadata.get_chunk_grid`), which the serving stack builds on the host from the batch's
  sequence lengths, and the grid size is the host-known number of chunks. The function's own fallback builds the
  schedule with a CPU round trip, which a CUDA-graph capture cannot do. The task therefore carries the schedule as
  two metadata inputs, `chunk_sequence_ids` and `chunk_ids`, and the launch uses `chunk_sequence_ids.shape[0]` as
  its grid. The schedule is built outside the timed call (as in a serving step) and is the same for every
  implementation.
- To make that shape a function of the axes, the number of chunks is a derived axis,
  `n_chunks = ceil(n_tokens / 64) + (n_seqs - 1) // 2`, and `inputs.py` draws sequence lengths whose chunk counts
  add up to exactly `n_chunks` (it splits `n_chunks` into `n_seqs` positive counts, then picks each length inside
  its count so that the lengths add up to `n_tokens`; this is possible for every `n_tokens`). With about 256
  tokens per sequence (`n_seqs = ceil(n_tokens / 256)`) the lengths are mixed: 1 to about 1900 tokens, not tile
  multiples, with a few shorter than the state length at the larger batches. Padded sequences
  (`cache_indices == -1`) and prefix-cache block states are not part of this scenario.

## Scenario and config citations

Qwen3-Next dimensions, from the defaults of `Qwen3NextConfig` in `vllm/transformers_utils/configs/qwen3_next.py`
(pinned vLLM commit): `linear_conv_kernel_dim` 4, `linear_key_head_dim` 128, `linear_value_head_dim` 128,
`linear_num_key_heads` 16, `linear_num_value_heads` 32. The conv channels are
`2 * 16 * 128 + 32 * 128 = 8192` (key 2048, value 4096). AITER's own test of this kernel
(`op_tests/test_causal_conv1d_prefill_split_qkv.py`) uses the same `CONV_DIM` 8192, `K_DIM` 2048, `V_DIM` 4096,
`WIDTH` 4. The Hugging Face `config.json` of Qwen3-Next-80B-A3B-Instruct has the same fields (`linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_key_head_dim` 128, `linear_num_value_heads` 32, `linear_value_head_dim` 128); the vLLM class defaults corroborate it.

## Baseline

`baseline/causal_conv1d_flydsl.py` is the FlyDSL kernel builder `build_causal_conv1d_flydsl_module` of AITER
(`aiter/ops/flydsl/causal_conv1d_flydsl.py`, MIT, `baseline/LICENSE`) and its `functools.cache` wrapper, with the
optional-import guard turned into plain imports. The host wrapper `causal_conv1d_split_qkv_flydsl_fn` is
reproduced by `kernel.py` on its precomputed-schedule path (bias, SiLU, bfloat16, fixed 64 x 64 tile with 256
threads, as AITER dispatches it by default), writing into caller-provided outputs. The wrapper's dtype conversion,
the default-argument allocations and the CPU round trip are not part of the call. `baseline/buffer_ops.py` is
AITER's vendored copy of FlyDSL's `kernels/common/buffer_ops.py` (Apache-2.0, `baseline/LICENSE-FlyDSL`, header
kept verbatim).

Origin notes: the file carries only AITER's SPDX/copyright header and no "adapted from" text; the first commit of
the file (`git log --follow`) is "add multi-backend prefill causal_conv1d kernels for GDN" (the same change adds the
HIP and Triton variants). The Triton causal conv1d kernels elsewhere derive from the causal-conv1d project
(Tri Dao); no such statement exists in this FlyDSL file. No third-party origin was found.

## FlyDSL 0.3.2 adaptation

FlyDSL 0.3.2 adaptation: in buffer_ops.buffer_load and buffer_store the cache-policy operand is passed as the keyword-only aux integer attribute of rocdl.RawPtrBufferLoadOp / RawPtrBufferStoreOp (the runtime's op constructors take no positional aux value); no change to the kernel algorithm. The rest of the FlyDSL API used by the kernel (`flyc.kernel/jit/compile`, `fx.struct`, `fx.Array`, `fx.SharedAllocator`, `fx.ptr_store/ptr_load`, `fx.math.exp2`, `fx.range_constexpr`, `fx.const_expr`, `fx.gpu.barrier`, `fx.Int8/Int32/Int64/Float32`, `T.i8/i32/bf16`) was checked by name against `python/flydsl` of FlyDSL v0.3.2.

## Comparison

`query`, `key` and `value` use the default `baseline_relative` bound. The state only receives copies of inputs
(no arithmetic), so `conv_states` is compared `exact`; this is stricter than the default, not a relaxation.

CPU emulation (float32 taps in the kernel's order with `exp2`-based SiLU as the baseline, 2 draws of 300 and 2048
tokens): float32 `acc * sigmoid(acc)`, reversed tap order and PyTorch-style SiLU pass the default bound with no
violating element; the float64 reference itself is the same design. The wrong designs are rejected: dropping the
bias fails on 87% of the elements, rounding the pre-activation to bfloat16 on 0.2% (peak 1.6x), bfloat16
accumulation on 1.7% (peak 3.4x to 5x).

## Workloads

`n_tokens` 1280 to 131072 (102x span) are performance cases: chunked-prefill batches of about 256 tokens per
sequence. Each token moves 16 KiB in (`x`) and 16 KiB out, so 1280 tokens are 42 MB (9.3 us at 4.5 TB/s plus the
4 us fixed cost, about 13 us) and 131072 tokens are 4.3 GB (about 0.96 ms). Smaller batches run under 10 us and are
correctness-only, together with the boundary cases 1 (axis minimum), 2, 3 (shorter than and equal to the conv
state), 48 and 300 (not a power of two), 1000 (not a multiple of 16). The upper end is bounded by the 8.6 GB
float64 reference at 8192 channels and by the kernel's 32-bit byte offsets: `x` of 131072 tokens is 2^31 bytes.

## Kernel notes

- `has_initial_state` is passed as a `torch.bool` tensor (FlyDSL maps it to `i1`) and read by the kernel as bytes,
  as AITER's test does.
- The wrapper compiles on first call through `flyc.compile` and reuses the `CompiledFunction` for every shape,
  as AITER does; the tensors are dynamic-layout arguments.
- The program grid is `n_chunks` (up to 2303) by 128 feature tiles; the fast and slow paths of the kernel
  (interior chunks versus chunks at sequence ends or with conv-state blending) are both exercised by the mixed
  lengths.
