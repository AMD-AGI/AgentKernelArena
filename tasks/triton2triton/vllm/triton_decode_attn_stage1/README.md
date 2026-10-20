# triton_decode_attn_stage1

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton decode attention stage1 kernel `_fwd_kernel_stage1` for
maximum GPU throughput while maintaining numerical correctness.

The kernel implements the first stage of split-KV decode attention. Each program
instance handles one (batch, head, kv_split) and computes partial attention
over a contiguous range of KV tokens from a paged KV cache:
- Loads Q vector for current (batch, head)
- Iterates over KV tokens in the assigned split, loading K/V via paged
  Req_to_tokens mapping
- Computes Q @ K^T scaled dot-product with online softmax
- Outputs partial attention output and logsumexp for later reduction

Key optimization opportunities:
- BLOCK_N tuning for the target GPU
- Memory access coalescing for paged KV cache reads
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Reducing masking overhead
- Loop unrolling

Constraints:
- Must maintain the same function signature for `decode_att_m_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Must handle the input domain stated under "Supported input domain" below


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

The five original scored cases remain unchanged. The three `contract_case`
controls in `workloads.json` execute the same public wrapper and the same
10-warmup/100-sample unscored graph-replay checks, with concrete input shapes,
dtypes, routing and seed. Their exact manifest parameters are checked against
the protected generator. They do not contribute performance rows or change the original scoring case set.
Stage-1 controls use ragged lengths 63 and 37, reversed page routing, and both
zero and positive logit caps. The positive-cap oracle applies tanh before
softmax. Replay changes page-table data and exchanges sequence lengths.

The additional inactive-split control uses lengths 2 and 37. Upstream stage 1
intentionally leaves empty split slots untouched; those slots are initialized
to a distinct finite sentinel and checked byte-for-byte as caller-owned state.
All active output elements, including logsumexp, are poisoned before numerical
checks/replay. No inactive slot is claimed as a kernel write. The original five
cases have only active splits and still require complete output overwrites.

The official scoring domain remains exactly `perf1` through `perf5`, with their
original inputs, tolerances, timing boundaries, ten warmups and 100 samples.
Every added control declares only `correctness`. Its real measured/replayed
branch is checked inside the correctness action and reported as an explicitly
unscored diagnostic in that case's metrics. The performance action executes
only the five original workloads; neither their weights nor aggregation change.

Additional unscored public-branch controls from PR105: Ragged split/page/head tails with explicit capped logits.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.


### Editable device implementation boundary

The editable device functions are `tanh`, `_fwd_kernel_stage1`.
Their bodies and additional implementation helpers may be optimized. Module
imports, initialization, host dispatch and pointer preparation remain protected,
along with the existing numerical checks and benchmark protocol. The structural
guard enforces this boundary; it does not isolate arbitrary Python execution.

The protected wrapper also fixes the launch configuration: the grid and the tile
sizes, `num_warps` and `num_stages` it passes at launch. Optimization notes above
that mention tuning those values refer to the original whole-file contract; under
this boundary, optimize the kernel bodies and implementation helpers instead.

### Supported input domain

The protected wrapper launches one program per (request, query head, split), uses a
fixed `BLOCK_N = 64` token tile, and pads the K and V head dimensions to the next
power of two with masking. Within the limits below, a candidate must produce the
correct partial output and log-sum-exp for every active split:

- Sequence lengths: at least 1 token per request, chosen independently per request.
  Lengths need not be multiples of the page size or of `BLOCK_N`. Request `b` uses
  row `b` of `Req_to_tokens` and `B_Seqlen[b]`.
- Page size: any positive integer. Entry `p` of a request's `Req_to_tokens` row is
  the physical page that holds that request's tokens `p * page_size` through
  `(p + 1) * page_size - 1`. The row needs at least `ceil(L / page_size)` entries,
  and each referenced entry must be a valid page of `k_buffer`/`v_buffer`. Entries
  after a request's last logical page, unreferenced pages, and slots past a
  request's length in its final page must not affect the result.
- Splits: any `num_kv_splits >= 1`. With `C = ceil(L / num_kv_splits)`, split `s`
  covers tokens `[s * C, min(L, (s + 1) * C))`. A split with no tokens is inactive:
  the kernel does not write its `att_out` slot, and the caller's bytes there must
  remain unchanged.
- Head dimensions: the Q/K head dimension `Lk` and the V head dimension `Lv` are
  independent integers from 16 through 256. They need not be powers of two or
  equal. `att_out` has shape `[batch, num_heads, num_kv_splits, Lv + 1]`. Head
  dimensions outside 16-256 are outside this contract.
- Heads: `num_heads` must be a multiple of `num_kv_heads`. Query head `h` reads KV
  head `h // (num_heads // num_kv_heads)`.
- Data: `q`, `k_buffer` and `v_buffer` share one dtype, float16 or bfloat16.
  `att_out` is float32, and `Req_to_tokens` and `B_Seqlen` are int32. A
  `logit_cap` of 0 disables capping; a positive value applies
  `logit_cap * tanh(logits / logit_cap)` before softmax.
- Layout: the last dimension of `q`, `k_buffer`, `v_buffer`, `att_out` and
  `Req_to_tokens` has stride 1, and `B_Seqlen` is contiguous. Each tensor must
  hold fewer than 2^31 elements, because the kernel computes element offsets in
  32-bit integers.

### Domain controls

Seven correctness-only `control-domain-*` rows (case indices 20000-20006) sample
this domain beyond the scored cases and earlier controls. The protected
`_domain_controls.py` holds their generator and oracle. `_arena_eval.py` checks its
`CASES` table against the manifest's `domain_controls` list and the exact rows,
and `scripts/task_runner.py` dispatches case indices of 20000 and above to it.

| Control | Lk / Lv | Page size | Splits | Lengths | Q / KV heads | Dtype, cap |
|---|---|---|---|---|---|---|
| `page1_hd16_short_inactive` | 16 / 16 | 1 | 4 | 1, 2, 5, 70 | 4 / 4 | fp16, 0 |
| `page3_hd96_group3` | 96 / 96 | 3 | 3 | 100, 7, 64 | 6 / 2 | fp16, 0 |
| `page48_hd32_mqa_one_split_capped` | 32 / 32 | 48 | 1 | 129, 47, 48 | 8 / 1 | fp16, 1.0 |
| `page128_k192_v128` | 192 / 128 | 128 | 2 | 300, 129 | 4 / 2 | fp16, 0 |
| `page64_k80_v256` | 80 / 256 | 64 | 4 | 65, 191 | 2 / 1 | fp16, 0 |
| `long_page16_splits16` | 128 / 128 | 16 | 16 | 7999, 4099 | 8 / 2 | fp16, 0 |
| `bf16_page5_hd112_splits7_capped` | 112 / 112 | 5 | 7 | 1, 33, 257 | 4 / 1 | bf16, 2.5 |

Together with the scored cases and earlier controls, the checked points are head
dimensions 16, 32, 64, 80, 96, 112, 128, 192 and 256 (with `Lk` both larger and
smaller than `Lv`), page sizes 1, 3, 5, 8, 16, 32, 48, 64 and 128, split counts 1,
2, 3, 4, 5, 7, 8 and 16, and lengths from 1 to 7999 tokens.

Each domain control works as follows:

- Pages: a seeded random permutation assigns physical pages. Table padding entries
  point to unused decoy pages. Every K/V slot that no active position references
  holds V = 32, so attending to such a slot changes the result.
- Oracle: before each call, an independent FP64 CPU oracle is computed from
  pristine clones. The call also goes through the installed correctness guard
  used by the other controls: the Triton dispatch audit and its FP32 reference
  (the original harness reference without a cap, the guard's capped reference
  otherwise).
- Outputs: every active output element is poisoned with NaN and every inactive
  split slot is filled with 23.5. After the call, active elements must be finite
  and match the oracle within atol = rtol = 0.01, and inactive slots must be
  byte-identical.
- Inputs: any change to the bytes or metadata of `q`, `k_buffer`, `v_buffer`,
  `Req_to_tokens` or `B_Seqlen` is rejected, as is any change to the shape,
  stride, dtype, device or storage of `att_out`.
- Second call: the wrapper runs again on the same tensors, with lengths rotated
  across requests and all data and the page permutation redrawn from the second
  declared seed.

On CPU, `validate-task` also checks that the FP64 oracle matches the original
harness reference for uncapped controls, and that it rejects zero output,
uncapped logits and shifted page routing. These rows have no graph replay or
timing. The performance action still runs only `perf1`-`perf5`, and no score
weight, tolerance, warmup, sample count or timing boundary changes.
