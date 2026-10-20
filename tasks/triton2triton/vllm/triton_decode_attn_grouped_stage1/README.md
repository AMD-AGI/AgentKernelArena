# triton_decode_attn_grouped_stage1

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton decode attention grouped stage1 kernel
`_fwd_grouped_kernel_stage1` for maximum GPU throughput while maintaining
numerical correctness.

The kernel implements the first stage of split-KV decode attention for
grouped-query attention (GQA/MQA). Multiple Q heads that share the same KV
head are processed together in tiles of BLOCK_H=16 heads:
- Uses a grid of (batch, num_kv_heads * ceil(kv_group_num / BLOCK_H), num_kv_splits)
- Each program loads up to BLOCK_H Q vectors of one KV group and computes
  attention against that shared KV head; a group's last tile is masked at the
  group boundary
- Reads K/V from a paged KV cache through the Req_to_tokens logical-page table
- Uses tl.dot for batched Q @ K^T computation
- The kernel body keeps the upstream optional positional-encoding split
  (BLOCK_DPE, used upstream for DeepSeek MLA), but the protected wrapper always
  launches with BLOCK_DPE=0, so that branch is unreachable through this contract
- Outputs partial attention output and logsumexp for later reduction

Key optimization opportunities:
- BLOCK_N and BLOCK_H tuning for the target GPU
- Memory access coalescing for paged KV cache reads
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Efficient use of tl.dot for the grouped head computation
- Reducing masking overhead

Constraints:
- Must maintain the same function signature for `decode_grouped_att_m_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Must be correct over the input domain listed under "Supported input domain"
  below


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

The five original scored cases remain unchanged. Additional correctness-only
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


### Editable device implementation boundary

The editable device functions are `tanh`, `_fwd_grouped_kernel_stage1`.
Their bodies and additional implementation helpers may be optimized. Module
imports, initialization, host dispatch and pointer preparation remain protected,
along with the existing numerical checks and benchmark protocol. The structural
guard enforces this boundary; it does not isolate arbitrary Python execution.

The protected wrapper also fixes the launch configuration: the grid and the tile
sizes, `num_warps` and `num_stages` it passes at launch. Optimization notes above
that mention tuning those values refer to the original whole-file contract; under
this boundary, optimize the kernel bodies and implementation helpers instead.

### Group boundaries crossing a 16-head tile

Two additional correctness-only controls use 2 KV heads with 34 or 48 query heads
(group sizes 17 and 24), ragged lengths 63/37 and nonidentity page routing. The
second also uses logit capping. They reuse the existing independent FP32 reference,
full active-output/unchanged inactive-slot checks and unscored graph replay.
The launch grid assigns `ceil(group_size/16)` blocks to each KV head independently;
query-head offsets start at that KV group's base and the final block is masked at
the group boundary. This prevents cross-group routing when a group needs a partial
second block. The original group sizes 1/4/8 have exactly the same launch grid and
head assignments as before. All original five scored cases and the three earlier
controls remain, along with seeds, inputs, tolerances, ten warmups and 100 samples.
No score weight or timing boundary changes. Kernel source changes require fresh
baseline/candidate measurements; historical GPU qualification does not transfer.

### Supported input domain

The protected wrapper fixes BLOCK_N=32, BLOCK_H=16, BLOCK_DPE=0, `num_warps=4`
and `num_stages=2`, and derives every other launch value from its arguments.
The kernel must be correct over this domain:

| Input | Contract in the protected wrapper and kernel | Checked values (scored cases and controls) |
| --- | --- | --- |
| Head widths | Lk is `k_buffer.shape[-1]` (and q's last dimension), Lv is `v_buffer.shape[-1]`, and they may differ. `att_out` has Lv + 1 entries per slot. Each width is padded to one next-power-of-two tile and masked; there is no positional-encoding split. | Lk/Lv 16/16, 40/72, 64/64, 80/80, 96/64, 128/128, 256/256 |
| Query and KV heads | `num_heads` must be a multiple of `num_kv_heads`. `kv_group_num = num_heads // num_kv_heads`, and query head h reads KV head `h // kv_group_num`. Each KV head gets `ceil(kv_group_num / 16)` head tiles. With an indivisible head count the trailing query heads are never written. | groups 1, 3, 4, 5, 7, 8, 12, 16, 17, 24, 33; KV heads 1, 2, 3, 4, 8 |
| Page size | Any positive integer. Position p of request b reads physical page `Req_to_tokens[b, p // page_size]` at offset `p % page_size`. | 1, 2, 3, 5, 12, 16, 32, 64 |
| KV splits | Any positive integer. With length L and S splits, split s covers `[s*ceil(L/S), min((s+1)*ceil(L/S), L))`. Splits starting at or after L are inactive and their slots are left untouched. | 1 through 8 |
| Sequence lengths | Per request, `1 <= b_seqlen[b] <= page_size * table columns`. Lengths need not be multiples of the page size or of BLOCK_N. | 1 to 512, including 1, 2, 3, 7, 21, 31, 33, 37, 45, 53, 63, 70, 77, 100 and 130 |
| Data types and layout | The task uses float16 q, `k_buffer` and `v_buffer`, float32 `att_out`, and int32 `Req_to_tokens` and `B_Seqlen`. The kernel addresses the last dimension of every tensor with unit stride and receives the other strides from the wrapper. | contiguous generator layouts |

Outside the checked domain: widths whose padded tile is narrower than 16 or
wider than 256 (for example MLA's 576/512 widths, which upstream splits with
BLOCK_DPE), other data types, and zero-length requests (the kernel writes none
of that request's slots).

### Domain controls

Seven further correctness-only controls cover that domain beyond the cases
above:

| Control | Batch | Q/KV heads (group) | Lk/Lv | Page | Splits | Lengths | Cap | Covers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `domain_group1_page1` | 2 | 3/3 (1) | 64/64 | 1 | 3 | 45, 7 | 0 | MHA with an odd head count, one-token pages, partial tile |
| `domain_group3_page5_dim80` | 2 | 6/2 (3) | 80/80 | 5 | 2 | 53, 21 | 1.5 | non-power-of-two width, odd page size, capping |
| `domain_group5_page3_k96_v64` | 2 | 15/3 (5) | 96/64 | 3 | 5 | 70, 3 | 0 | Lk > Lv, three KV heads, inactive splits |
| `domain_group7_page64_dim256` | 2 | 14/2 (7) | 256/256 | 64 | 1 | 100, 33 | 0 | widest tile, page larger than BLOCK_N, one split |
| `domain_group33_page12_k40_v72` | 2 | 66/2 (33) | 40/72 | 12 | 6 | 77, 1 | 2.5 | three head tiles per group, Lk < Lv, length 1 |
| `domain_mqa_group12_page2_dim16` | 3 | 12/1 (12) | 16/16 | 2 | 4 | 31, 64, 2 | 0 | MQA, narrowest tile, three requests |
| `domain_group16_page16_dim128` | 2 | 32/2 (16) | 128/128 | 16 | 7 | 130, 40 | 1.5 | group exactly filling one tile, seven splits |

Each control draws q, K and V from a seeded CPU generator, so every device sees
the same bytes. Each request routes through a seeded permutation of physical
pages; the table has one spare column, and one physical page is never
referenced. `att_out` starts at the finite sentinel 23.5. The protected
`_arena_domain_controls.py` declares the table, the generator and the oracle,
and the adapter checks the manifest rows against that table.

The oracle is an independent float64 evaluation of the pristine input clones
taken before the candidate call. Per request it gathers tokens through the page
table, maps query head h to KV head `h // kv_group_num`, and computes each
split's logsumexp and softmax-weighted values, applying `cap * tanh(x / cap)`
before softmax when the cap is positive. The controls reuse every check of the
earlier controls: the launch audit; NaN poisoning of every active output
element; read-only input, metadata and alias checks; byte-exact inactive slots;
finite outputs within atol = rtol = 0.01; and the unscored 10-warmup/100-sample
graph replay with perturbed inputs. Each control declares only `correctness`. The performance action, the five
scored cases and their inputs, tolerances, warmups, samples and timing are
unchanged.
