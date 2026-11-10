# DeepSeek-V4.1-Flash K2 (attention back, FFN seam, MoE), TP4

K2 is the second of the two launches that sgl-amd-mono's DeepSeek-V4.1-Flash monokernel path runs for every
covered decoder layer of a decode step (`spec.py`: "Every layer is K1, then K2"). This task is one K2 launch on each
of 4 tensor-parallel ranks: one persistent FlyDSL kernel of 256 CTAs x 512 threads a rank, whose TP reductions run
inside the kernel through symmetric peer memory. It covers the inventory records `dsv41.k2` and `dsv41.kffn`: the
test-only FFN launch (`build_mono_ffn`, which "runs in no forward") is K2's FFN half, so it is not a separate task and
its torch golden model (`check_ffn.py golden_partial`) is the MoE half of this reference.

## Operator

Per decode row (one request) and each of the rank's 16 heads:

1. **Sparse MQA decode with sink**: the row's keys come from K1's key table: its top-512 compressed rows
   (`slot | 1 << 31`) and the 128 positions of its window (its own new record included). Each key is a V4.1
   fp8_ds_mla record (448 E4M3 dims with an E8M0 scale per 64, then 64 bf16 RoPE dims); K = V over all 512 dims.
   `softmax(q.k / sqrt(512), sink)` (the sink takes probability mass, no value), bf16 output, inverse GPT-J RoPE of
   dims 448..511 by the row's position, bf16 -> XO.
2. **wo_a / wo_b**: per group of 8 heads, XO x wo_a (FP8, 32 x 32 E8M0 blocks, widened to bf16) -> bf16 -> MXFP8
   (vLLM's rule) -> the rank's 2048 wo_b columns -> a bf16 partial; the partials are summed over the 4 ranks in rank
   order (in-kernel peer exchange) -> `a`.
3. **FFN mHC seam**: `R' = bf16(post a + comb^T R)` (K1's attention post / comb, -> `res_out`), the collapse with K1's
   next pre, 24 mix projections, the 20-iteration Sinkhorn gate (-> `pre_out`, `post_out`, `comb_out`), RMSNorm with
   post_attention_layernorm -> h.
4. **Router**: bf16 logits `h W_g^T`, `score = sqrt(softplus(logit))`, the 6 largest `score + bias` of 384, weights =
   scores renormalized x 1.5.
5. **MoE** on the rank's 576 of 2304 intermediate columns: 6 routed MXFP4 experts (x MXFP8 with aiter's rule,
   gate | up, `bf16(min(g, 10) sigmoid(min(g, 10)) clamp(u, -10, 10))`, MXFP8 mid, down, `bf16` at every weighted add
   in top-k order) plus the FP8 shared expert (vLLM's MXFP8 rule, bf16 roundings of SGLang's
   silu_and_mul_with_clamp), a bf16 partial a rank, summed over the 4 ranks in rank order -> `out`.

The float64 definition is `dsv41_math.py` (its docstring lists every rounding); `reference.py` returns its outputs.

## Multi-GPU contract (`distributed`, world size 4)

* The generator draws the **logical** layer once per seed: all 64 heads of q and the sinks, the 8 wo_a groups and
  wo_b's 8192 columns, all 384 experts over the whole 2304-wide intermediate, the whole shared expert. `shard_inputs`
  gives rank r heads `16r ..`, wo_a groups `2r, 2r + 1` (woa rows, wob columns `2048r ..` and their scale blocks),
  intermediate columns `576r ..` of every routed expert (gate and up rows each padded to 640 with zero weights, w2's
  columns padded alike, w2_s columns padded to 24 with 0x7F, as SGLang's loader pads a rank's share) and of the shared
  expert. Everything else (caches, key table, positions, RoPE table, seam state, mHC, router) is replicated.
  `to_kernel_layout` applies SGLang's gfx950 MXFP4 load layout (aiter `shuffle_weight` / `shuffle_scale` with the
  gate / up interleave; checked against the kernel's `mx.scale_index` and `fp4_tile_load` addressing on CPU).
* `rank_expected`: every rank writes the whole outputs (the reductions sum in rank order on every rank), so every
  output is in `ranks_agree`.
* Reserved inputs: `peers` / `symmetric` are the kernel's `peers` / `sym` arguments. `symmetric_bytes` = 20,553,472:
  the two parity halves of `layer.peer_layout(48, 4)` (the ATTN / FFN exchange and the token-parallel gather regions,
  10,276,608 bytes each, sized for 48 rows as the runner allocates them) and the guard's 256 error-flag bytes at
  `2 * half`. With epoch 0 every call uses parity half 0; the framework zeroes the buffer before every call.
* Persistent state: the runner keeps the scratch mailboxes and the epoch buffer across launches and the epoch moves
  on every K2. Here `run()` allocates the epoch buffer (epoch 0, zero error words, CTA marks and MoE queue counters;
  `guard.py` reads its zero deadline words as a 10 s deadline) and the scratch (`scratch_bytes`, 0.25-2 MB) in one
  zeroed buffer in every call, so the call's fixed tags never meet an earlier call's words. Its memset is a second
  node of the timed graph (an estimated 1-3 us); a zeroed local workspace prepared outside timing would remove it
  (a framework request). The scratch stays in ordinary device memory, as the runner's: the framework's symmetric
  buffer is uncached, which would slow the mailbox traffic production keeps in cached memory.

## Baseline

Vendored from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d (SGLang ROCm images pin
e52803f9, 11 commits older): `deepseek_v41/gfx950/{layer,token_par}.py`, `attention/{back,front,gemv,plan,device}.py`
(as `back.py`, `front.py`, `attn_gemv.py`, `attn_plan.py`, `attn_device.py`), `stages/*`, `common/{ops,sync,guard,
epoch,mx,plan,stamps}.py`, and aiter's FlyDSL `buffer_ops.py` / `dpp_utils.py` at dbd8bf5bd. `ops.py`, `guard.py`,
`buffer_ops.py` and `dpp_utils.py` are the copies of `tasks/flydsl/other/qwen38_flash_next_decoder/verify_gdn3_qsa1`
(QA PASS on MI355X), `ops.py` plus `global_atomic_add`. The FlyDSL 0.3.2 bridge of
`muse_spark/gfx950/compat.py` is applied as plain functions (no module patching): generic stores / loads, the
pointer type, the MoE's global atomics and its route-mask LDS atomic OR (`ops.lds_atomic_or`, upstream
`fx.llvm.atomic_or`) are llvm-dialect ops. The index-source index query (`index_query.py`,
which imports NumPy) is not vendored; no build here uses it. `baseline/LICENSE` carries the Apache-2.0 notice and
ROCm/ATOM's MIT license (the seam, MoE and common files are adapted from ATOM 97359d6). Every change is listed in
`task.yaml` and in each file's header. `kernel.py` is new: `DSV41MonoRunner.k2` for one launch.

The build is `MonoBuild(rows, 4, kt=True, compressed=True, index=False)`: a layer with a compressed cache that is
not an index source (DeepSeek-V4.1-Flash's covered layers are compress ratio 0, 1 or 2; an index source's K2 also
builds the key table, and a window-only layer reads no compressed cache). Row-major wo_b (SGLang's default, without
SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE); `ug_groups` 1 below 7 rows, else 2, as `runner.route_at`.

## Workloads

`rows` 1..8 is the AgentX interactive range (decode, one row a request; the runner compiles widths 1..48, and
from 12 rows a multiple of 4 runs the token-parallel variant, a different kernel). The CRS smoke job 278746 built
K1 / K2 widths 1-4 at TP4 and ran decode bs 1-4 on them. Rows frequency (`workload_evidence.json`): the sum of C
per-session in-flight draws (AgentX `agentx_inflight.json`) at C = 1, 2, 4, 8 sessions, equally weighted, kept to
<= 8 rows: 1 26.7 %, 2 25.6 %, 3 3.5 %, 4 20.9 %, 5 5.4 %, 6 2.8 %, 7 1.4 %, 8 13.7 %.

K2's keys are a fixed set (128 window positions + top-512 compressed rows) at every context of 1,024 tokens or more,
so its work depends on the rows alone and its production range spans 8x. The task therefore uses the
**production-range span exemption** (`workloads.json performance_span`, `workload_evidence.json`): the performance
cases are rows 1, 2, 4 and 8 (the four most frequent values, both endpoints of 1..8) at ctx = 262,144 (the AgentX
decode-step bucket (131K, 262K], 25.9 % of decode steps, which holds the decode-step-weighted median 219,840); ctx
is a var axis only for the correctness cases (one key, a partial window, kv_len < 640, the 4K bucket, the 1M cap),
which reuse the four compiled widths. Derived from corpus statistics; not yet confirmed by a live capture.

Estimated baseline time (bytes a rank from cold HBM at ~5 TB/s, plus the launch's serial mailbox chain of ~8-12 us):
weights read every step 33 MB (wo_a 8.4, wo_b 10.5, router 3.9, shared expert 8.8, mHC 2.0) plus about 5.2 MB a
distinct routed expert (6 picks a row, 384 experts) and 0.37 MB of KV records a row: rows 1 ~65 MB (~13 + 8 us), 2
~97 MB (~19 + 8 us), 4 ~155 MB (~31 + 10 us), 8 ~277 MB (~55 + 12 us). Every performance case is expected above
10 us. Evaluation budget (estimated, not measured): 4 distinct compiled widths a role (K2 compiles in ~5-15 s a
width), 9 cases x 2 seeds of GPU input generation (~7 GB of logical expert banks a draw) and float64 reference, 4 x
110 timed samples (each reloads ~2.3 GB of rank inputs, ~1 ms): about 4-7 minutes for baseline and candidate.

## Reference and its assumptions

The reference is float64 on dequantized operands with every narrow store rounded as the kernel stores it (bf16, E4M3
with the producing stage's E8M0 rule computed in float32 as the kernel does). It was written from the kernel source
(`back.py stage_split / stage_combine / stage_woa / stage_wob`, `seam.py`, `moe.py`); the upstream checks compare
against SGLang / aiter ops (`check_k2.py`: torch `attention_xo`, then wo_a / wo_b of K2's own XO; `check_ffn.py`:
`golden_partial`). Assumptions:

* The kernel rounds the attention probabilities and each 64- or 128-key chunk's partial to bf16 before merging; the
  reference does not (exact softmax), as `check_k2.attention_xo` does not. It rounds o, XO, wo_a's output, the
  partials and every MoE intermediate where the kernel stores them.
* The routed experts' MXFP8 uses aiter's rule (`max(amax, 1e-10) * f32(1/448)`, codes 1..254), the shared expert,
  wo_a's output and K1's activations vLLM's (`RN(max(amax, FLT_MIN) / 448)`, codes 0..254), as the kernel's
  `put_mxfp8_group` / `vllm_mx_code` / `mx_code`.
* The kernel's sigmoid / exp use hardware approximations (`v_exp_f32`, `v_rcp_f32`); the reference uses exact ones.
* XO's RoPE dims get the inverse rotation at the row's position (`e' = e cos + o sin`, `o' = o cos - e sin`), as
  `back.stage_combine` and `check_k2.attention_xo` apply it (`dsv41_math.attention_xo`, `rope_rotate(...,
  inverse=True)`). An earlier revision rotated forward: the identity at position 0, so only `r1_ctx1` passed.

## Comparison bounds

Every output uses `baseline_relative` with `k = 4` (the bf16 `out` and `res_out` also `max_violation_fraction =
0.01`). The launch requantizes intermediates to MXFP8 twice before `a` (wo_a's output) and twice in each expert (x
and the swiglu output), so an accumulation order other than the baseline's moves some E4M3 codes by one step (an
eighth of the element) at different elements than the baseline's; those errors are not correlated with the
baseline's, and a correct candidate's elementwise error exceeds twice the baseline's per-row RMS for a few percent of
the elements. `k = 4` covers the tail, and the violation fraction admits the rare element (each within 16 x the
bound) where both effects stack; a wrong expert, scale or key still moves the output by tens of percent. The bf16
outputs are compared before their final rounding (the reference returns the unrounded float64 value).

## Planted margins

The router's top-6 is the one discrete decision. The generator computes the FFN input h exactly as the reference does
and plants the router weight (`inputs.py` docstring): every row gets 6 random winners whose selection keys are spaced
over [1.55, 1.85] and losers with keys <= 0.74, so the 6th / 7th margin is >= 0.5 and the winners' order is separated
by >= 0.02 (checked after the bf16 rounding of the weight). The kernel's h differs from the reference's by bf16
roundings and MXFP8 flips (~1e-3 relative), which moves a logit by ~1e-2.

## What is verified and what is not

Verified on CPU (Mac and the home minipc, `src.tools.check_task --smoke-limit 2000000`, `py_compile`, the
candidate-rule audit of the baseline, an import of the baseline with FlyDSL stubbed): the contract and the span
exemption load; the generator (with its routing-margin check), the reference and every rank's shard / layout /
expected run for the smallest (r1_ctx1) and largest (r2_ctx1048576) cases and for r8_ctx262144 and r4_ctx1000
(minipc: 15 s generation, 7 s float64 reference, 17.3 GB peak RSS on 4 CPU threads; the logical expert banks are
~7 GB and every rank's shard clones them); the MXFP4 layouts match the kernel's `mx.scale_index` / `fp4_tile_load`
addressing; the record encode / decode round-trips; the outputs are finite.

On 4 MI355X (FlyDSL 0.3.2, torch 2.11 ROCm 10; a debug diagnostic that reads the baseline's scratch, cases r1_ctx1,
r1_ctx262144 and r2_ctx100 with key-table variants): the vendored kernel compiles and runs; its XO matches this
reference's attention at ~50 dB (NoPE and RoPE dims alike), its chunk partials at ~55 dB, wo_a's MXFP8 codes
bitwise; the float64 seam of its own wo_b partials matches its `res_out` at 55 dB and the mixes at 71-84 dB; the
float64 MoE of its own h matches `out` at 55 dB; repeated calls are bitwise equal. The QA run before the RoPE fix
failed every case with a position > 0 (see the reference assumptions). The fix changes the generator's planted
router (it plants on the reference's h). **Not verified: a QA run of this revision (the baseline's error against the
fixed reference, so the `baseline_relative` bounds), the timing, the 10 us floor of the rows-1 case and the
evaluation budget.**
