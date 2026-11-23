# Paged attention decode, FP8 KV cache (per-tensor scales), 8 query heads / 1 KV head, head size 128, 1024-token pages

Single-token (decode) grouped-query attention over a paged FP8 KV cache with one scale per tensor.
For each sequence `s` and query head `h`,

```text
out[s, h] = softmax(softmax_scale * q[s, h] . K_s^T) . V_s
K_s[t] = key_code[t] * key_scale[0],  V_s[t] = value_code[t] * value_scale[0]
```

over the first `context_lengths[s]` cached tokens; the eight query heads share the single KV head,
`softmax_scale = 128 ** -0.5`, and the result is rounded to bfloat16. The float64 reference
dequantizes the codes exactly and is otherwise exact attention.

This is the per-tensor sibling of `paged_attention_decode/fp8kv_q8_kv1_d128_p1024` (per-token
scales). The kernel and worklist are the same; the scale layout, the baseline's probability
quantization and the workloads differ (this scenario's are weighted by a recorded serving workload
and allow contexts up to 256k).

## Production origin

AITER's persistent ("PS") paged-attention decode path serves FP8 KV caches in 1024-token pages with
per-token or per-tensor quantization (`op_tests/test_pa_ps.py`, `pa_persistent_fwd`). This task's
baseline is the FlyDSL port of that path, `kernels/pa_decode_fp8.py` from ROCm/FlyDSL, whose
upstream test runs both quantization modes. The cache is quantized the way AITER's
`per_tensor_quant` does it: `scale = max|x| / 448` over the whole key (or value) cache and
`code = float8_e4m3fn(x / scale)`. A per-tensor FP8 cache is the common production setup, with
one key and one value scale per layer. Eight query heads per KV head and head size 128 are the
per-rank attention of GQA 8:1 models such as Llama-3.1-70B at TP 8 or Qwen3-30B-A3B at TP 4.

## Interface and layouts

- `query[num_seqs, 8, 128]` bfloat16: one decode token per sequence.
- `key_cache[pages, 1, 8, 1024, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d // 16, slot, d % 16]` (16-byte key vectors).
- `value_cache[pages, 1, 64, 128, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is
  at `[page, kv_head, slot // 16, d, slot % 16]` (the PS kernel's transposed value layout).
- `key_scale[1]`, `value_scale[1]` float32: the per-tensor scales.
- `context_lengths[num_seqs]`, `kv_indptr[num_seqs + 1]`, `kv_page_indices[pages]`: token `t` of
  sequence `s` is in page `kv_page_indices[kv_indptr[s] + t // 1024]`, slot `t % 1024`. Every
  sequence owns `ceil(max_seq_len / 1024)` scattered pages and lists the first
  `ceil(context_lengths[s] / 1024)`. The longest sequence has `max_seq_len` tokens, the others between
  half of it and all of it. Slots past a sequence's length and the pages it does not list hold
  stale but finite codes; they must not reach the result.
- The worklist (`work_indptr`, `work_info`, `reduce_indptr`, `reduce_final_map`,
  `reduce_partial_map`): what AITER's `get_pa_metadata_v1` (`csrc/kernels/mla/metadata/v1_2_pa_device.cuh`,
  query length 1, one KV head) builds for 256 compute units, followed by `pa_decode_fp8.py`'s
  block-split expansion. Compute unit `c` handles work rows `work_indptr[c] .. work_indptr[c+1]-1`.
  A row `[sequence, slot, q_start, q_end, kv_start, kv_end, 0, q_head_range]` covers
  `kv_page_indices[kv_start:kv_end]` of one sequence. Its four 256-token sub-tile partials are rows
  `slot * 4 + sub_tile` of the partial buffers. Sequence group `g` merges the partial rows
  `reduce_partial_map[reduce_indptr[g]:reduce_indptr[g+1]]` into output token `reduce_final_map[g, 0]`.
  Production builds this once per decode step and shares it across all layers, so it is an input
  rather than timed work. `inputs.py` ports both the scheduler and the expansion. A candidate may
  use it or schedule the work itself.

## Workloads

The performance cases come from a recorded agentic serving workload: the AgentX corpus
(`semianalysisai/cc-traces-weka-062126`, agentic coding sessions) replayed with 32 concurrent sessions against
`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on SGLang 0.5.23 and MI355X (capture
`qwen3-235b-a22b-tp8-mi355x-v1`): TP 8, no DP attention, no speculative decoding, chunked prefill 16384, FP8
KV cache (one scale per tensor) in 16-token pages, aiter attention, and a 262144-token context (requests that
do not fit are skipped: 30% of the corpus requests, 67% of its input tokens). Each case's `weight` is how
often its shapes ran (calls per second of the steady state, TP rank 0), and the task scores the
serving-weighted speedup `sum(w*t_baseline)/sum(w*t_candidate)`, the time the workload spends in the operator
before and after (see Case weights in `docs/how-to/add-task.md`). `weight_provenance` in `workloads.json`
records the capture, the effective serving settings and the share of the corpus the context window dropped.
The 12 weighted cases cover 98.9% of the operator's recorded GPU time:

| `num_seqs` | `max_seq_len` | Phases | Share of the operator's time | Weight (calls/s) |
| ---: | ---: | --- | ---: | ---: |
| 24 | 123904 | decode | 0.6% | 38.63 |
| 24 | 126464 | decode | 9.0% | 556.9 |
| 24 | 129536 | decode | 0.7% | 41.35 |
| 24 | 133120 | decode | 0.9% | 55.96 |
| 24 | 134656 | decode | 0.5% | 35.02 |
| 24 | 135680 | decode | 0.5% | 35.9 |
| 32 | 129024 | decode | 0.5% | 28.24 |
| 32 | 133632 | decode | 0.8% | 39.15 |
| 32 | 136704 | decode | 2.5% | 116.4 |
| 32 | 139264 | decode | 2.6% | 131.5 |
| 32 | 140288 | decode | 0.8% | 33.79 |
| 32 | 147968 | decode | 79.4% | 3537 |

Qwen3-235B-A22B has 64 query heads and 4 KV heads of size 128, so a TP-8 rank runs 8 query heads over one
(replicated) KV head, this task's shape. Every case is a decode step: `num_seqs` is the running batch padded
to its CUDA-graph batch size (24 and 32 at this load), and `max_seq_len` is the step's mean KV length per
sequence from the scheduler log, in 512-token buckets (the task draws the other sequences between half of it
and all of it, so a case reads somewhat less KV than the recorded step). Agentic sessions keep long contexts:
recorded KV lengths reach about 161k tokens, so the `max_seq_len` axis allows 262144 (the deployment's
context); with the earlier 131072 limit 89% of the operator's time fell above the axis. The served attention
uses 16-token pages; the task keeps the PS kernel's 1024-token pages, which change the page arithmetic, not
the work. In the MI355X QA audit the baseline took 0.13-0.19 ms per case; the largest weighted case (32
sequences of up to 148k tokens) holds about 0.6 GB of FP8 KV per cache.

The earlier performance cases (batch sizes 1-256 with contexts of 8k to 128k, extended from AITER's PS test)
are correctness-only, together with the earlier boundary cases (`s3_l1027` and `s81_l1027`, the batches of the
upstream FlyDSL test, single-page and page-crossing contexts), the other recorded shapes one case with a
non-default `softmax_scale`, and 160 sequences at the axis maximum of 262144 tokens, whose 5 GiB caches put KV
addresses above 4 GiB (37 cases in total).

## Baseline

`baseline/kernel.py` vendors `compile_pa_decode_ps` and its helpers unchanged from ROCm/FlyDSL
`kernels/pa_decode_fp8.py` at commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47`; `dpp_utils.py`
comes from `kernels/dpp_utils.py`. `run()` is the persistent-scheduling branch of
`pa_decode_ps_launch` with no sliding window and bf16 queries. Scales of shape `[1]` select the
kernel's per-tensor path, exactly as upstream. It launches the PS kernel once, with grid
`(256 compute units, 1, 4 sub-tiles)`, to walk the worklist, then launches the reduce. The upstream
launch calls two AITER operators that cannot be vendored as FlyDSL source:

- `get_pa_metadata_v1` (a HIP kernel) and the host-side block-split expansion build the worklist.
  Both are ported to `inputs.py`, and the result is passed in.
- `pa_reduce_v1` merges the partials. It is replaced by `compile_pa_decode_ps_reduce`, a task-written
  FlyDSL kernel with the same merge, `out = sum_i exp(lse_i - lse) * partial_i`. It runs one
  workgroup per (sequence group, query head) and one thread per head-dimension element. Each wave
  step reads 64 partial LSEs, and a running maximum carries across steps. It is likely slower than
  AITER's reduce for sequences split many ways.

`baseline/` is identical to the per-token sibling's apart from the `run()` docstring.
`buffer_ops.py`, `vector.py` and `meta.py` are FlyDSL's `python/flydsl/expr` helpers from the same
commit, used only when the installed FlyDSL no longer ships them. Three changes were made: the
descriptor flags come from the detected architecture instead of an environment variable;
`vector.py` re-exports only the two raw ops the kernel uses instead of a star import; and `meta.py`
attaches unknown MLIR locations instead of reading the Python stack.

## Accuracy and the comparison bound

The baseline quantizes each query head to FP8 (`scale = max|q| / 240`). In the per-tensor path it
also quantizes the probabilities directly to FP8 in `[0, 1]`, relative to the maximum of each
256-token sub-tile, before the PV product. A CPU emulation of this path on the task's inputs gave
29-30 dB SQNR against the exact reference; the default sanity floor is 25 dB.

The default `baseline_relative` bound is kept. It accepts an implementation that reproduces the
baseline's FP8 quantization. It also accepts one that keeps the query and the probabilities in
bfloat16 or better, which the emulation placed at 0.11-0.14 of the bound. Normalizing the FP8
probabilities per 64 tokens instead peaked at 1.2-1.5 times the bound and fails. As in the
per-token sibling, the bound cannot be widened for such variants without also accepting an output
rounded to FP8 (2.0-2.5 times the bound at the default `k`), which the QA controls must reject.

## Not verified here

The kernels need FlyDSL and gfx950. The worklist port and the reference were checked on the CPU, and the
baseline's per-tensor numerics against an emulation of the PS kernel and the reduce. Contexts above 131072
tokens (the earlier axis limit) are exercised by the weighted cases (up to 148k) and by the correctness case
at the axis maximum.