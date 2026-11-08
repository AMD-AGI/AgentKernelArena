# FP8 MQA logits, DeepSeek-V3.2 indexer (64 heads, head size 128), Gluon baseline

For each query token the lightning indexer scores every key in its window:
`logits[i, j] = sum_h weights[i, h] * relu(q[i, h, :] . kv[j, :] * kv_scales[j])`, with `-inf` outside
`[cu_seqlen_ks[i], cu_seqlen_ke[i])`. Q and KV are float8_e4m3fn (MI355X native format), the dot is
accumulated in float32, and the float32 logits feed a later top-k. The interface, reference and window
generator are those of the sibling task `tasks/triton/attention/fp8_mqa_logits/h64_d128`, so the operator
means the same in every baseline language.

**Production origin.** AITER's `fp8_mqa_logits` on gfx950, which dispatches to a Gluon kernel
(`aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py`) for Triton 3.6 and newer, used by
vLLM and SGLang for the DeepSeek-V3.2 sparse-attention indexer. DeepSeek-V3.2 `config.json`:
`index_n_heads` 64, `index_head_dim` 128 (and `index_topk` 2048, the number of keys the logits are later
reduced to).

**Scenario and workloads.** Prefill chunks: `seq_len` query tokens against `seq_len_kv` packed keys. The
window generator (`inputs.py`, shared with the sibling) packs two requests (one when a side is too small
to split) and gives each row its chunked-prefill causal window, so the ragged `cu_seqlen_ks/ke` offsets
and the start > 0 path are exercised. Differences from the sibling's workloads: `seq_len_kv` starts at 64
and every case has at least as many keys as two times the queries (or a single request), so no row has a
one-column window. A single-column window makes the baseline-relative bound degenerate (the sibling
README documents this and needs a violation allowance for it); avoiding those windows keeps the default
`baseline_relative` bound, with no override. Nine timed cases span 2.1M to 268M `seq_len * seq_len_kv`
(128x), from 512 queries against 4096 keys (estimated baseline time about 15-30 us: 16 Gflop of fp8
work plus an 8 MB logits write) to 8192 queries against 32768 keys (about 2-3 ms; 1 GiB of float32
logits, below the 2 GiB buffer-store limit). Correctness-only cases: the axis minimum (1 x 64), decode
against 4096 and 65537 keys, windows shorter than one 32-column tile (16 x 64) and one tile plus a
one-column tail (2 x 66), odd sizes (7 x 2000, 300 x 2000, 100 x 777) and 2048 x 4096.

## Baseline

`baseline/gluon_kernel.py` vendors AITER's gfx950 Gluon kernel file from ROCm/aiter at
652f2d4b768239aefd515c929e84817681ff598d unchanged, except that `strip_annotate` (a Python 3.14
compatibility helper that upstream imports from `aiter/ops/triton/utils/common_utils.py`; a no-op on
Python 3.12) is inlined from that file. Upstream has no SPDX or copyright header in either file, so the
repository license (MIT, `baseline/LICENSE`) applies. No third-party origin note exists in the vendored
files or their history; only AITER's test for this operator states that it is adapted from DeepGEMM's
`tests/test_attention.py`, and that test is not part of this task.

`baseline/kernel.py` is the gfx950 branch of the upstream Python wrapper behind `run()`: one program per
query row, `BLOCK_KV` 32, one wave, `waves_per_eu` 3, two buffers, the four-chain folded head reduction
when the Triton build supports constexpr-tuple permutes (the upstream probe is kept), buffer loads and
stores below the 2 GiB limit. Adaptations:

- **-inf fill.** The Gluon kernel writes only the in-window columns of a row. Production pre-fills the
  logits with `torch.full(-inf)`; `run()` does the same on the evaluator's preallocated buffer with
  `logits.fill_(-inf)`, a preparation operation the candidate rules allow and that is graph-capturable.
  The upstream Gluon kernel is the only kernel launched.
- **Layout probe.** Upstream decides with `inspect.getsource` whether `async_copy.global_load_to_shared`
  accepts a distributed layout (it returns True when the source mentions `DistributedLayout`) and then
  selects the padded shared layout. `inspect` is not available to implementations, so the result is a
  constant, `True`. In triton-lang/triton at the `release/3.8.x` branch
  (`python/triton/experimental/gluon/language/amd/cdna4/async_copy.py`) `global_load_to_shared` checks
  `isinstance(ptr.type.layout, DistributedLayout)`, so the probe returns True there. That was read from the
  branch, not from the exact build in the image (3.8.0+git4cff872c). `run()` launches once,
  with no fallback.
- **Output allocation.** Upstream allocates logits padded to a multiple of 256 columns and returns a
  view; here the evaluator provides the exact `[seq_len, seq_len_kv]` tensor. All kernel stores are masked
  to the window, so they never leave their row.
- Dropped: the Triton and gfx942 paths, the gfx1250 dispatch and the FN-to-FNUZ recast.

## Reference and bound

`reference.py` is the sibling's float64 evaluation of the dequantized products. The default
`baseline_relative` bound applies (`k = 2`, no violation allowance); the `-inf` positions must match
exactly. The scaled-MFMA accumulation is the baseline's dominant error, as in the sibling.
The sibling's analysis puts the error of a correct float32 candidate at 0.04-0.10x of the bound for many-column rows.
