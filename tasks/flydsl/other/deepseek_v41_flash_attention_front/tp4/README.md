# DeepSeek-V4.1-Flash K1 (attention seam and front), TP4

K1 is the first of the two launches that sgl-amd-mono's DeepSeek-V4.1-Flash monokernel path runs for every covered
decoder layer of a decode step (`spec.py`: "Every layer is K1, then K2"; inventory record `dsv41.k1`). This task is
one K1 launch on each of 4 tensor-parallel ranks: one persistent FlyDSL kernel of 256 CTAs x 512 threads a rank.

## Operator

Per decode row (one request):

1. **Attention mHC seam**: the previous layer's owed FFN post folded into its 4 residual streams,
   `R' = bf16(post pend + comb^T R)` (-> `res_out`), the collapse `x = bf16(pre . R')` with the previous gate's pre,
   24 mix projections over the 4 x 5120 streams, the 20-iteration Sinkhorn gate (-> `pre_out`, `post_out`,
   `comb_out`, which K2 uses), input RMSNorm -> y.
2. **Attention front**: y -> MXFP8 (vLLM's rule) -> the fused wq_a | wkv_a GEMV (FP8, 32 x 32 E8M0 blocks, replicated)
   -> bf16. q: RMSNorm (q_norm) -> bf16 -> MXFP8 -> the rank's wq_b rows (16 heads x 512) -> bf16 -> GPT-J RoPE of dims
   448..511 -> bf16 -> `q`. kv: RMSNorm (kv_norm) -> GPT-J RoPE of dims 448..511 -> the V4.1 fp8_ds_mla record written
   at the row's slot of the paged SWA cache (`swa`, mutable): 448 E4M3 codes in 7 blocks of 64 with E8M0 scale
   `ceil(log2(max(amax, 1e-4) / 448))`, 64 bf16 values, and the 7 scale codes plus a zero byte in the page's scale rows.
3. **K2's key table**: `kt[t] = topk | 1 << 31` for the row's top-k compressed slots, then its window slots, then -1;
   `klen = (kv_len, ntopk)` per row.

The float64 definition is `dsv41_math.py`; `reference.py` returns its outputs and the final cache.

## Multi-GPU contract (`distributed`, world size 4)

* The logical inputs hold all 64 heads of wq_b; `shard_inputs` gives rank r rows `8192r ..` (and their scale
  blocks) and its own copy of the SWA cache; everything else is replicated. `rank_expected` gives rank r its 16 heads
  of q; the seam outputs, the key table and the cache (every rank writes the same records into its replica) are
  replicated and in `ranks_agree`.
* K1 below 12 rows exchanges nothing between ranks (token parallel, from 12 rows, gathers rows through the peer
  buffer); the peer buffer holds the guard's error flags at `2 * half`. `symmetric_bytes` = 20,553,472 is the
  runner's allocation (the two parity halves of `layer.peer_layout(48, 4)` and 256 flag bytes).
* `run()` allocates the epoch buffer (epoch 0) and the scratch mailboxes in one zeroed buffer in every call (the
  runner keeps them across launches; `guard.py` reads the zero deadline words as a 10 s deadline). Its memset is a
  second node of the timed graph (an estimated 1-3 us); a zeroed local workspace prepared outside timing would remove
  it (a framework request).

## Baseline

The same vendored package as the K2 task (`tasks/flydsl/other/deepseek_v41_flash_attention_back_moe/tp4`; see its
README and `task.yaml` provenance): sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d, aiter's
FlyDSL buffer / DPP helpers at dbd8bf5bd, the FlyDSL 0.3.2 bridge as plain llvm-dialect functions, msgspec replaced
by dataclasses, no index query. `layer.py` builds K1, K2 and the FFN launch from one scratch layout, so the MoE and
attention-back modules are imported though K1 traces none of their stages. The build is
`MonoBuild(rows, 4, kt=True, fold=True, index=False)` (a layer that is not an index source, after a layer that owes
its post), row-major wqkv and wq_b. `kernel.py` is new: `DSV41MonoRunner.k1` for one launch, with the DSpark capture,
the index-source outputs and the index query off.

## Workloads

`rows` 1..8 and the frequencies are the K2 task's (AgentX concurrency 1, 2, 4, 8, one decode row a request;
`workload_evidence.json`). K1 reads its weights (wqkv_a 9.2 MB replicated, the rank's wq_b 10.5 MB, the mHC
projections 2 MB, ~21.7 MB a rank) and writes a few rows; the context only sets the RoPE position and how much of
the key table is live, so its work does not depend on the context. The task uses the **production-range span
exemption**: performance cases rows 1, 2, 4, 8 at ctx = 262,144 (the AgentX decode-step bucket holding the weighted
median 219,840); ctx varies only in the correctness cases (one-token context, partial windows and top-k lists, the 4K
bucket, the 1M cap), which reuse the four compiled widths. Derived from corpus statistics; not yet confirmed by a live
capture.

**Estimated baseline time and the 10 us floor (risk)**: by bytes alone K1 is ~4.3 us at 5 TB/s, for every row count.
Its serial chain (seam slices -> gate / norm -> MXFP8 publish -> wqkv GEMV -> q / kv norms and the record ->
q publish -> wq_b GEMV, each hand-off a device-scope mailbox wait of ~1-2 us, the weights prefetched behind them)
suggests ~9-14 us. **If the GPU QA measures the rows-1 case (or more than one case) below 10 us, this task must be
dropped** (fewer than 4 performance cases would remain above the floor; the brief forbids batching). Evaluation
budget (estimated): 4 compiled widths a role, 9 cases x 2 seeds of light input generation, 4 x 110 timed samples:
about 2-4 minutes for baseline and candidate.

## Reference and its assumptions

Float64 on dequantized operands, rounded where the kernel stores (`dsv41_math.py` docstring), written from
`seam.py`, `layer.py store_x8`, `front.py stage_qkv / stage_wqb / stage_kt`. The upstream check (`check_k1.py`)
compares bitwise against SGLang's ops (`hc_boundary_fused_deferred`, `mxfp8_native_blockscaled_linear`,
`fused_k_norm_rope_flashmla`) and q against a torch `query_ref` (cosine 0.99999, rel 5e-3).

## Comparison bounds

`q` uses `baseline_relative` with `k = 4` and `max_violation_fraction = 0.01`: q_norm's output is requantized to MXFP8
before wq_b, so another accumulation order moves a few E4M3 codes at other elements than the baseline's, and a
correct candidate's errors are not correlated with the baseline's. The float32 mixes (`pre_out`, `post_out`,
`comb_out`, a few values a row, compared against the whole output's RMS) use `k = 4` for the same reason at float32
precision (20,480-term projections and 20 Sinkhorn iterations). `res_out` keeps the default (its error is its own bf16
rounding). bf16 outputs are compared before their final rounding (the reference returns the unrounded value).
`kt`, `klen` and the cache `swa` compare exactly (integers and packed bytes).

## Planted exactness of the KV record

The SWA cache is packed bytes (E4M3 codes, bf16 halves, E8M0 codes), so it compares exactly, and a float64 reference
reproduces a float32 kernel's bytes only away from rounding boundaries. The generator (`inputs.py` docstring) keeps
every rounding on the record's path at least 2e-5 (relative to its terms) from a boundary: the hidden columns that
feed the kv projection are the seam columns whose bf16 roundings and MXFP8 block codes are safe for every row; wkv_a
(wqkv rows 1280..1791) has **one nonzero E4M3 weight a row** at such a column, so `kv_a` is one exact product (the q
rows stay dense); `kv_norm` gains are resampled until every record code, bf16 value and block scale is safe for every
row (checked; the generator raises otherwise). The kv projection is thus a scaled gather of seam outputs: it still
exercises the GEMV's addressing and block scales over all of K, the kv RMSNorm, RoPE, the record quantization, packing
and slot addressing, but not the accumulation of dense kv rows (the q rows test that).

## What is verified and what is not

Verified on CPU (Mac, home minipc): the contract and the span exemption load (`src.tools.check_task --smoke-limit
2000000`: the smallest and largest cases, every rank's shard / expected), the generator plants the record margins
(56 seeds over rows 1, 2, 8 on the Mac, every perf width on the minipc; under 1 s a case, 0.9 GB), the reference
runs and its outputs are finite, the record encode / decode round-trips, the baseline passes the candidate-rule audit
and imports with FlyDSL stubbed.
**Not verified (no GPU in this build session): that the vendored kernel compiles and runs on FlyDSL 0.3.2, the
bit-exactness of the record under the planted margins, the `baseline_relative` bounds, the timing and the 10 us
floor.**
