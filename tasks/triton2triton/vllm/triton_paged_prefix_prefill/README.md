# triton_paged_prefix_prefill

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton paged prefix prefill attention kernel `_fwd_kernel` for
maximum GPU throughput while maintaining numerical correctness.

The kernel implements paged attention for the prefix prefill stage of LLM
inference. It reads context tokens from a paged KV cache (5D K / 4D V layout)
and new query tokens from dense tensors.

Two-phase attention:
1. Query-vs-Context: queries attend to all cached context tokens (no causal mask)
2. Query-vs-Query: queries attend to new query tokens (with causal mask)

Key data structures:
- K cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x] (5D vectorized)
- V cache: [num_blocks, num_kv_heads, head_dim, block_size] (4D)
- Block table (B_Loc): maps logical block indices to physical block IDs

Key optimization opportunities:
- Tile size tuning (BLOCK_M, BLOCK_N, BLOCK_SIZE) for the target GPU
- Memory access coalescing for the 5D K cache layout
- Loop unrolling factors for cache vs request loops
- Warp scheduling and pipeline stages
- Reducing masking overhead at block boundaries
- Prefetching block table entries

Constraints:
- Must maintain the same function signature for `context_attention_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Must handle the wrapper's full "Supported input domain" below, not only the scored shapes


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


### Protected execution and replay checks

The task computes each oracle from pristine inputs before calling the public
wrapper, checks all returned or supplied outputs (including auxiliary statistics),
and rejects changes to read-only input bytes, tensor metadata, or output/input
aliasing. Shape-only inputs are compared bytewise without assuming finite values.
The wrapper must dispatch the declared genuine Triton JIT kernel; Torch allocation,
same-device casts/copies and views are permitted, while operator computation in
Torch is rejected. References execute outside that dispatch guard. This is a
backend execution contract, not a Python security sandbox.

Performance retains the original cases, 0.01 absolute/relative gates, ten warmups,
100 samples, and shared timing helper. The actual measured graph outputs are
checked against the pristine reference. Outside timing, source data is perturbed
and every output poisoned, then the same captured graph is replayed and fully
compared against a new reference. Inputs are restored afterward. No reference,
poisoning, comparison, or extra GPU check is added inside the timed invocation.
A timing fallback that cannot expose the actual measured outputs fails closed.

Additional correctness-only controls are declared with concrete shapes, dtypes, sequence
lengths, optional arguments and seeds in `workloads.json`; the protected evaluator
requires those parameters to match the actual generator. All original scored
cases remain unchanged. Each added case runs the real public wrapper with the
same 0.01 gates and an unscored captured-replay diagnostic during correctness.
That diagnostic uses ten warmups and 100 samples; it is never a performance row.
The control covers ragged context/query lengths, non-identity physical pages, a
nondefault attention scale, and the wrapper's window or ALiBi path. The oracle
reconstructs context from actual cache bytes and independently forms the masked
attention operation; cached generator outputs cannot substitute for this check.

A second ragged control uses physical block size 24 to exercise the original
non-power-of-two dispatch; its dimensions and device timing remain explicit.

The plain-prefix context page-table load now masks tokens beyond the context
length, matching its existing masked K/V loads. This fixes an out-of-bounds
metadata read for a partial tile with non-power-of-two physical page sizes;
all other original kernel operations remain unchanged.

The official scoring domain remains exactly `perf1` through `perf5`, with their
original inputs, tolerances, timing boundaries, ten warmups and 100 samples.
Every added control declares only `correctness`. Its real measured/replayed
branch is checked inside the correctness action and reported as an explicitly
unscored diagnostic in that case's metrics. The performance action executes
only the five original workloads; neither their weights nor aggregation change.


### Editable device implementation boundary

The editable device functions are `_fwd_kernel`.
Their bodies and additional implementation helpers may be optimized. Module
imports, initialization, host dispatch and pointer preparation remain protected,
along with the existing numerical checks and benchmark protocol. The structural
guard enforces this boundary; it does not isolate arbitrary Python execution.

The protected wrapper also fixes the launch configuration: the grid and the tile
sizes, `num_warps` and `num_stages` it passes at launch. Optimization notes above
that mention tuning those values refer to the original whole-file contract; under
this boundary, optimize the kernel bodies and implementation helpers instead.


### Supported input domain

The protected `context_attention_fwd` launch defines the inputs the kernel must
handle. The scored rows sample only part of this domain:

- Data: float16 `q`, `k`, `v`, caches and output in the harness's contiguous
  layouts; int32 page table, start locations and sequence lengths. The 5D K
  cache uses vector width x = 8 (16 bytes).
- Head dimension: a multiple of 8 from 16 through 128. The wrapper pads it to
  the next power of two (`BLOCK_DMODEL_PADDED`) and the kernel masks the padded
  lanes, so non-power-of-two sizes such as 24, 80, 96 and 112 are included.
  Head dimensions below 16 or above 128 are outside the evaluated contract; the
  fixed launch configuration has not been qualified for them.
- Heads: `num_heads` is a multiple of `num_kv_heads` (MHA, GQA with any integer
  group size, or MQA).
- Physical page size: any positive size. Power-of-two pages launch
  `BLOCK_M=128, BLOCK_N=64`; other sizes launch `BLOCK_M=BLOCK_N=32`. Context is
  read in 32-token tiles, so a page may be smaller or larger than a tile. The
  page table may map logical pages to any physical pages and needs at least
  `ceil(context_length / page_size)` entries per sequence.
- Lengths: per sequence, a context length >= 0 and a query length >= 0 (empty
  query segments and single-token queries included), with
  `b_seq_len = context + query` and `max_input_len` at least the largest query
  length. Lengths need not be multiples of the context tile, the query tiles or
  the page size.
- Scale and window: `sm_scale` omitted (`1/sqrt(head_dim)`) or any explicit
  value; `sliding_window` omitted or <= 0 (disabled), or any positive width,
  including widths longer than the sequence.

Seven further correctness-only rows sample this domain beyond the original
cases and controls. Their declarations live in the protected
`_domain_controls.py`, which `_arena_checks.py` merges into the same control
table. They therefore use the same manifest binding and public-wrapper checks
described above: the FP32 oracle computed from pristine inputs, which rebuilds
context from the actual cache bytes through the page table; read-only input and
metadata checks; full-output 0.01 gates; and the unscored captured-replay
diagnostic. Each row uses a non-identity page table, finite data in unused pages
and page tails, and an explicit nondefault scale. No row is scored or timed in
the performance action.

| Control | head_dim | Page | Tiles M/N | Heads q/kv | Context lengths | Query lengths | Scale | Window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `domain_hd80_page8_two_m_tiles` | 80 | 8 | 128/64 | 8/2 | 0, 33, 130 | 129, 1, 17 | 0.07 | none |
| `domain_hd96_page64_mqa_window` | 96 | 64 | 128/64 | 4/1 | 95, 64, 1 | 70, 64, 2 | 0.25 | 40 |
| `domain_hd112_page12_gqa3_window` | 112 | 12 | 32/32 | 6/2 | 47, 0, 25 | 33, 31, 1 | 0.06 | 3 |
| `domain_hd24_page40_wide_window` | 24 | 40 | 32/32 | 8/8 | 300, 81 | 65, 96 | 0.3 | 4096 |
| `domain_hd16_page16_empty_query` | 16 | 16 | 128/64 | 2/1 | 5, 200, 31 | 0, 3, 40 | 0.5 | none |
| `domain_hd128_page32_long_ragged` | 128 | 32 | 128/64 | 8/4 | 257, 0, 513 | 255, 7, 1 | 0.05 | none |
| `domain_hd32_page128_window` | 32 | 128 | 128/64 | 4/4 | 1, 129 | 128, 63 | 0.1 | 16 |

The default scale remains covered by the scored rows with head dimensions 64
and 128. These rows sample the domain; they do not enumerate it.
