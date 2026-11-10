# Paged MXFP4 MQA logits, DeepSeek-V4.1-Flash FP4 indexer, DSpark verify (32 heads, head size 128, 64-row pages)

For one DSpark verify step of R requests x 6 rows, every row scores the indexed keys of its request with the
lightning indexer, using MXFP4 queries and keys. Both are packed e2m1 vectors (two codes per byte, element `2i`
in the low nibble) with one E8M0 power-of-two scale per 32 elements: `dequant(x)[i] = e2m1(x[i]) *
2 ** (scale[i // 32] - 127)`. Row `t` (draft position `t % 6` of request `t // 6`) sees `bounds[t]` key columns;
column `c` is row `c % 64` of page `page_table[t, c // 64]` of SGLang's FP4 index plane:

```text
logits[t, c] = sum_h weights[t, h] * relu(dot(dequant(q[t, h, :]), dequant(key(c))))   c < bounds[t]
logits[t, c] = -inf                                                                      c >= bounds[t], up to ctx
```

Every row sees more than 512 columns. Accumulation and the output are float32. The logits feed SGLang's
top-512 kernel.

**Why the domain starts at ctx = 518.** A row that sees at most 512 columns is not scored in production:
SGLang's top-512 takes all of its columns without reading logits, and a step whose rows all see at most 512
does not launch K2i (`low_ratio_decode_rows_are_identity`). K2i does not decide this per row: a piece (the six
rows of a request) is scored up to its longest row's chunks, and each row's logits are written below its own
bound, so a row at or below 512 columns next to a longer one gets real logits that SGLang ignores. Those
elements are unspecified in production, so the task keeps them out of its domain: every request draws at least
518 columns, so its first row sees at least 513 and every row is scored. (An earlier revision kept such rows
with -inf logits; the MI355X audit, job 279635, showed the baseline's 2550 real logits there in `r1_c513`.)

## Production origin

sgl-amd-mono's K2i (`sgl_amd_mono/deepseek_v41/gfx950/indexer/k2i.py`, `score_fp4.py`), the FP4 scorer the
SGLang AMD monokernel plugin launches between K1 and K2 of every index-source layer of DeepSeek-V4.1-Flash
(`spec._select`, `K2iRunner.select` / `score`), on each TP rank. Its scorer comes from ROCm/ATOM's K2a
`index_score_fp4.py` by way of sgl-project/sglang#43984 and is "bit for bit with aiter's row-group kernel" in
arithmetic order (scaled `mfma_scale_f32_16x16x128_f8f6f4` with E2M1 operands, relu as `x + |x|` against half
the weight, a lane's heads in one FMA chain, then permlane swaps). The six rows of a request are one *piece*
of the step's walk plan: each CTA loads a 64-key step once and scores all six rows off it. The walk plan
(`walk_plan.py`, a one-CTA Triton kernel in the plugin) is built on the device from the step's bounds once per
step and compress ratio; it is part of `run()` here.

## A distinct task from `fp4_h64_d128_bs64`

`tasks/flydsl/attention/paged_mqa_logits/fp4_h64_d128_bs64` is the same operator in another scenario with
another baseline lineage: DeepSeek-V4(-Flash/-Pro)'s 64-head indexer on AITER's FlyDSL decode kernel
(`pa_mqa_logits_fp4`, one or two query tokens per sequence, AITER's preshuffled cache and persistent-grid
schedule). This task is DeepSeek-V4.1-Flash's 32-head indexer (`index_n_heads` 32) for DSpark verify steps of 6
rows a request sharing key loads, on SGLang's split FP4 index plane (payload and scales apart, 64-row pages) and
per-row page table and bounds, with sgl-amd-mono's ATOM-derived K2i as baseline. Under the schema-v3
target-language rules (`docs/how-to/add-task.md`: a scenario fixes dtypes, constant dimensions and layouts; two
scenarios of one operator are two tasks, filed under the baseline's language), a different scenario with a
different baseline lineage is a distinct task in the same operator directory.

## Layouts (SGLang's, the kernel ABI)

Inputs are generated directly in the production layouts; there is no layout transform.

- `q_fp4[rows, 32, 64]` uint8: head `h`, bytes `16 c .. 16 c + 15` hold elements `32 c .. 32 c + 31`.
- `q_scale[rows, 1, 4, 16, 4]` uint8: the scale of head `h`, group `c` is at `[t, 0, c, h % 16, h // 16]`; the
  last axis is padded to 4 (bytes 2 and 3 are unused, 127).
- `weights[rows, 32]` bfloat16: the rows' head weights with SGLang's head weight scale folded in (K2i applies
  no further scale).
- `kv_cache[pages, 1, 4, 64, 16]` uint8: group `c` of plane row `r` of a page at `[page, 0, c, r, :]`.
- `kv_scale[pages, 1, 4, 64]` uint8: the scale of group `c` of row `r = 16 j + l` at `[page, 0, c, 4 l + j]`
  (SGLang's `deepseek_v4_memory_pool` index-K scale order).
- `page_table[rows, ceil(ctx / 64)]` int32: SGLang's per-row page table; a request's six rows hold the same
  row, entries past its own pages are 0. The scorer reads the row `owners[t]` (a request's first row) for the
  whole piece; an entry is 2 ** 3 of its virtual 8-row pages.
- `bounds[rows]` int32: SGLang's compressed sequence length of each verify row (one fewer for each earlier
  draft row). `owners[rows]` int32: `t // 6 * 6`, the plugin's `K2iRunner.owners` table for 6 rows a request.
- `logits[rows, ctx]` float32 (the runner's buffer is as wide as SGLang's page table; here it is `ctx`, the
  longest row's bound).

## Scenario and constant dimensions

`num_heads` 32 and `head_size` 128: sgl-amd-mono's DeepSeek-V4.1-Flash `HF_CONFIG` (`index_n_heads` 32,
`index_head_dim` 128), the scorer's `index_plan.HEADS` / `DIM`. `rows_per_request` 6: DSpark verify (the
plugin's K2i test `PER_REQ = 6`, `k2i.MAX_ROWS = 48`, "8 DSpark requests of 6"), equal to the scorer's
`WALK_ROWS`. `page_size` 64 and E8M0 per 32: SGLang's FP4 index plane (`indexer/runner.py::check_plane`).
DeepSeek-V4.1-Flash has 8 index-source layers (2, 8, 14 at compress ratio 2; 20, 24, 28, 32, 36 at ratio 1);
`ctx` counts the columns of a ratio-1 source (a ratio-2 source sees half). The indexer is replicated on every
TP rank, so the task is single-GPU at any TP. Decode-only steps (1 row a request, no DSpark) and the REINDEX
candidate-block mode of the same kernel are other scenarios and are not part of this task. Monokernels serve
decode and verify only; prefill shapes do not apply.

## Workloads

Derived from corpus statistics; not yet confirmed by a live capture (the capture plugin will confirm or
replace them). Var axes: `num_requests` 1-8 (8 x 6 = the kernel's 48 rows) and `ctx` 513-524288, the longest
request's columns; the other requests draw theirs uniformly in `[max(513, ctx / 2), ctx]` (ragged, not
multiples of a page or a chunk), as the paged-attention tasks do. Each request owns `ceil(ctx / 64)` distinct
plane pages scattered at random.

- `ctx`: the AgentX decode-step-weighted context (`semianalysisai/cc-traces-weka-062126` rev 23f152f6, weight =
  tokens decoded, context = in + out / 2): p10 52K, p25 108K, p50 220K, p75 426K, p90 632K; share by bucket
  <=4K 4.4%, 4-64K 9.2%, 64-128K 17.0%, 128-256K 25.9%, 256-512K 26.9%, >512K 16.7%. The cap is 524288 = 2^19,
  the kernel's tested bound (`BOUND_MAX` in the plugin's K2i test); AgentX steps above it are clipped onto the
  cap.
- `num_requests`: the AgentX per-session in-flight distribution (k = 1 90.0%, 2 5.4%, 3 2.1%, 4 0.85%, ...)
  summed over C = 1, 2, 4, 8 concurrent sessions, each concurrency point weighted equally, keeping steps that
  fit 8 requests: R = 1 22.6%, 2 21.8%, 3 3.0%, 4 18.5%, 5 4.8%, 6 2.4%, 7 1.2%, 8 25.8%.

**Production-range span exemption** (`workloads.json` `performance_span`, evidence
`workload_evidence.json`). The kernel is DeepSeek-V4.1-Flash-specific (every non-var dimension is fixed by the
model) and its work is linear in requests x ctx, so the context axis was tried first: the >= 10 us shapes need
about 245K columns at one request and about 40K at eight, so they span only about 17x in `num_requests * ctx`
(max 8 x 524288) and the 100x rule is not reachable. The evidenced perf range is `num_requests` 1-8 x `ctx`
65536-524288 (64x box); the observed range is 1-8 x 1024-524288. Performance cases (11): R = 1 at 448K and
512K, R = 2 at 256K and 512K, R = 3 at 220K (the median), R = 4 at 128K and 320K, R = 8 at 64K, 128K, 256K and
512K, weighted toward the frequent R = 1, 2, 4, 8 and the 128K-512K buckets. Estimated baseline time per case,
from the bytes moved (68 bytes per key per request, 4 bytes per row per column for the logits below the bound
and for the -inf tail past it, each written once) at 4.5 TB/s plus 5 us for the three launches: about 14 us
for the smallest (`r8_c65536`, `r4_c131072`, `r1_c458752`) and 77 us for the largest (`r8_c524288`). Endpoint
cases: `r1_c458752` / `r1_c524288` (R = 1, est. 14 / 16 us), `r8_c65536` (ctx = 64K, est. 14 us).
Correctness-only: the axis minima (`r1_c518`, whose first row sees 513 columns, one into its fifth 128-column
chunk), a non-power-of-two count with an unaligned context (`r3_c1000`), a short context (`r5_c4097`), the
widest step at a short context (`r8_c8191`), and three shapes estimated at 7-9 us, below the floor
(`r7_c33333`, `r2_c65536`, `r1_c131072`).

Queries and keys are standard normal bfloat16 activations quantized to MXFP4 (`scale = 2 ** ceil(log2(amax / 6))`
per 32 elements, round to nearest e2m1), as the sibling task quantizes. The head weights are bfloat16 from
N(0.5, 0.5), mostly positive: trained head weights are not public, and a positive mean gives the summed score
the nonzero mean a trained indexer has, so that a global scale error is caught by the mean-error test (the
sibling's reasoning).

## Baseline

`baseline/k2i.py` and `baseline/score_fp4.py` vendor sgl-amd-mono's K2i (Apache-2.0, with ROCm/ATOM MIT
portions: `baseline/LICENSE`, each file's header kept and extended with its changes). `baseline/kernel.py` is
`K2iRunner.walk_plan` then `K2iRunner.score` of one FULL layer as `run()`: a fresh `[256, 4]` plan buffer, the
walk plan, the -inf tail (`baseline/tail_fill.py`), then the scorer, all on the current stream and graph
capturable. Modifications (all in `baseline.provenance.modifications`):

- `baseline/tail_fill.py` is new: K2i writes each row's columns below its bound only (SGLang's top-k never
  reads the rest), and the evaluator poisons the output, so a small FlyDSL kernel writes -inf into the columns
  at or past each row's bound (a CTA a row and 4096 columns, returning at once when they are all below the
  bound). Every element is written once; an earlier revision pre-filled the whole output with `fill_`, which
  wrote the visible columns twice (about a fifth of the largest case's traffic). A candidate may fold the tail
  into its scorer, which saves one launch.

- The REINDEX branch of `run_scores4` (a producer layer's candidate blocks, scored from the CTA index), its
  `reindex` argument and the row count only it used are removed: this task scores FULL layers, and one build
  serves every row count (the plugin builds one per row count; the FULL path does not depend on it).
- `baseline/walk_plan.py` is a FlyDSL port of the plugin's Triton `_walk_plan_kernel`: a FlyDSL baseline may
  not launch Triton. One wave, a row a lane (warp scans for the run starts and piece indices, lane gathers over
  a lead's next five rows for the piece's span and chunks, warp reductions for the totals), the width search as
  a bisection over the Triton kernel's range (feasibility is monotone in the width, so the smallest feasible
  width is the same), then each thread writes four CTAs' tasks. A CPU emulation of this lane program equals
  ATOM's host `fill_walk_plan` on 2900 random steps (the plugin test's distribution: decode, verify and ragged
  owners, bounds 0..2^20, 1-48 rows, plus this task's shapes).
- `index_plan.py`, `ops.py`, `mx.py`, `plan.py` keep only what the scorer uses; the build-key source tag is a
  constant (upstream hashes source files) and the kernel symbol is formatted locally.

## FlyDSL 0.3.2 adaptation

sgl-amd-mono targets FlyDSL >= 0.3.4; the evaluation image has 0.3.2, which has no `fx.memory_fence`,
`fx.AtomicOrdering`, `fx.generic_load` or `fx.Pointer.ir_type`. As `muse_spark/gfx950/compat.py` bridges them
on 0.3.2 (and as the repository's Qwen3.8 monokernel task does), `ops.global_load` / `global_ptr` and
`ops.memory_fence` are written on llvm-dialect ops (`llvm.LoadOp` on an address-space-1 pointer,
`llvm.FenceOp`), without patching flydsl. `_lds_barrier` keeps its workgroup release fence, `s_barrier`,
acquire fence; `_stage_rows` loads its 16-byte units with `ops.global_load` (16-byte aligned) instead of
`fx.ptr_load` on an `fx.inttoptr` global pointer. The other APIs the scorer uses (`fx.struct` / `fx.Array` LDS,
`fx.ptr_load` / `ptr_store` on LDS, `range(..., init=...)` loops with `yield`, `rocdl.mfma_scale_f32_16x16x128_f8f6f4`,
`permlane32/16_swap`, `s_setprio`, `sched_barrier`, `s_barrier`, inline asm) and the walk plan's
(`fx.coop.warp_reduce` / `warp_inclusive_scan`, `fx.shuffle_idx`) appear in the FlyDSL v0.3.2 sources or in
the Qwen3.8 task's QA-passed baseline on 0.3.2; none of this was compiled here.
`buffer_ops.py` is AITER's copy (dbd8bf5bd) with the CDNA descriptor flags taken without reading the
environment, as in the Qwen3.8 task.

## Comparison

`baseline_relative` with `k` = 2 and `max_violation_fraction: 1.0e-3`, the sibling task's bound for the same
arithmetic: fp4 and E8M0 products are exact in float32, so the baseline's error is the scaled MFMA's float32
accumulation and the float32 head sum, close to a float32 evaluation's own error, and a correct candidate that
sums in another order exceeds twice the baseline's per-row error on a few elements (on MI355X a float32
evaluation of the sibling's reference failed the default bound at violation fractions up to 1.3e-4, about 8x
below 1e-3). Elements admitted by the allowance must still lie within 16x the bound; non-finite elements (the
-inf columns) must match exactly, so zeroed, half-written and stale outputs are rejected
independently of it. The fraction has not been measured for this task yet.

## Unverified

The first MI355X audit (job 279635, previous revision) compiled the scorer and the walk plan on FlyDSL 0.3.2
and passed baseline correctness on every case but `r1_c513` (SQNR about 160 dB against the reference), which
led to the domain change above; it stopped before timing. Unverified until the next audit: the new
`tail_fill.py` kernel, graph capture, the timing estimates (and so the 10 us margins) and the bound's
violation fraction. Checked on the CPU: the contract (including the span exemption) loads, the reference
equals the plugin test's own fp64 `ref_logits` bit for bit on generated inputs, the generator's shapes, page
ownership and every row's bound above 512, and the walk-plan lane program against the host plan.
