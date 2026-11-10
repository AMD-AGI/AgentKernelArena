# Muse Spark MXFP4 decoder layer, one TP8 / EP8 rank (8 GPUs, in-kernel collectives)

One persistent 256-CTA launch per rank runs a whole Muse Spark decoder layer for one DFlash2 verify step
(`requests` x 8 rows): this layer's verify attention, K2 (o-norm-gate, o_proj with the tensor-parallel
all-reduce, the attention boundary, the router, the pooler), K3 (the MXFP4 experts, the finalize, the unpooler,
the FFN boundary) and the next layer's K1 (fused q | k | v | gate projection, q / k norms, RoPE, fp8 cache
write). The ranks exchange partial rows inside the kernel through each other's symmetric buffers over XGMI;
there is no host-side collective. This is a multi-GPU task (`distributed.world_size: 8`).

**Model support is semi-experimental.** Muse Spark is registered in sgl-amd-mono's `SPECS` but absent from
the repository README's model table; it landed in one PR (#26), has no end-to-end server test, and its
default served widths are 8 and 16 rows (`gfx950/config.py`: "32 and 64 rows are under test").

## Production origin

sgl-project/sgl-amd-mono at `52486703fbf27e77f05b573e4d931d4eec87118d` (Apache-2.0; `common/` files adapted
from ROCm/ATOM, MIT): `sgl_amd_mono/muse_spark/gfx950/layer.py::build_layer` (`layer_`), with its stages
`stages/{attention,post_attn,ffn,pre_attn,mxfp4}.py`, `build_k2.py`, `build_k3.py`, `config.py`, `layout.py`,
`device.py`, the FlyDSL 0.3.2 bridge `compat.py`, and `common/{guard,epoch,ops,plan,sync,mx}.py`. SGLang's ROCm
images pin sgl-amd-mono `e52803f9` (11 commits older); the baseline is vendored from `52486703`.

The build is `MuseMonoRunner.layer_kernel(rows, 8, next_k1=True)` with the runner's default environment:
TP experts (`SGLANG_AMD_MONO_MUSE_TPX=1`), `XQ_PRE=1`, `UG_DYN=1`, the row-sharded pooler (`PSHARD=1`, which
the runner enables only for rows <= 16), no deferred K1, bf16 pooler / unpooler (`FP8_DENSE=0`), not the last
layer. With TP experts each rank holds its 1/8 slices of all 256 experts (the runner builds them from EP's 32
local experts a rank with `weights.tp_expert_copies`), so the task's per-rank expert tensors are those slices,
not 32 whole experts. `baseline/kernel.py::run` passes `MuseMonoRunner.launch`'s merged-path arguments.

Vendoring changes (one stage-code change, the mask fix below; details in `task.yaml` provenance): one flat package with
relative imports; the 0.3.2 bridge as plain functions called by `ops.global_ptr`/`global_load`,
`epoch.gstore`, `guard.stop_launch` instead of patching modules at import; frozen dataclasses / a slotted
class instead of `msgspec`; no `plan.kernel_sources` source digest (FlyDSL hashes a kernel directory's own
helpers and the package is flat); a local `kernel_signature`; environment reads replaced by their defaults;
`buffer_ops.py` from ROCm/aiter 652f2d4b (sgl-amd-mono imports `aiter.ops.flydsl.kernels.buffer_ops`; the
image's aiter is 4ad998328) and `dpp_utils.py` from ROCm/FlyDSL dd837334.

**One kernel change: a fix of the attention's causal mask.** `run_kattn`'s chunk loop applies the mask only
on a task's last chunk and the request's first ("a chunk with no masked key" otherwise). A verify step's
masked keys are the step's later rows (up to 7 keys before the end), so when a request's key count is 1-6
past a multiple of 256 they reach into the chunk before the last; when that chunk and the last share a
partition (every multi-chunk partition: contexts above ~16K), the step's earlier rows attended up to 6
future draft keys. A simulation of the kernel's partition sizing shows it, e.g. at 25,859 keys (two-chunk
partitions) rows 0-3 see 4, 3, 2, 1 future keys. The repository's fixed test lengths avoid it (2049, its
one straddling length, ends a 4-chunk partition) and its random-length cases use unplanted data, where a
few leaked keys among thousands rarely move a row past the test's tolerance. The fix adds one scalar
compare a chunk: a chunk skips the mask only if it also ends at or before the step's first row's last key.
The sliding-window path has the same pattern at the window's start; this task's attention layer has no
window. `r1_c25859` is the regression case: with this task's unplanted draws the leak moves the affected
rows by ~0.2% (about the bf16 output rounding), and in a CPU simulation of the attention alone (this
task's draws for all 8 ranks, exact attention rounded to bf16 as the baseline) the unfixed mask fails the
default bound on some rank in 10 of 10 draws. The same fix is in
`tasks/flydsl/attention/paged_attention_verify/fp8kv_q16_kv2_d64_p16_ql8`.

**Scratch placement.** The runner keeps four local mailbox scratch buffers (ordinary device memory) and moves
the epoch word on after every step, so mailbox tags never repeat. Here the epoch is a fixed input (`guard`,
epoch 1) and the four scratch regions sit in the framework's `symmetric` buffer after the peer regions
(`symmetric_bytes` = 47,190,016 peer bytes + 90,320,896 scratch bytes), which the framework zeroes before
every call. That buffer is uncached memory (the same allocation as the runner's `PeerBuffer`), so the
baseline's local scratch traffic is uncached where production's is not; the kernel's hand-offs already use
device-scope (L2-bypassing) accesses, so the cost is expected to be small but is unmeasured.

## Semantics (reference.py, float64)

Composed from the repository's per-launch test references (`test/kernels/test_muse_spark_{kattn,k2,k3,k1}.py`,
`test_muse_spark_k3_tp8.py`), keeping the production ops' bfloat16 tensors and the MXFP4 / fp8 quantization
exactly, everything else in float64:

- **Attention** (full-attention layer, query head h reads KV head h // 8): s = 0.125 q.k over the request's
  `seq_lens + 8` keys, causal over the step's 8 rows; softmax; o = P V.
- **K2**: og = bf16(o rstd sigmoid(g)) per head; attn = bf16(og w_o^T); nb = bf16(attn rstd) (eps 1e-8);
  new = alpha res + beta nb; x = bf16(new rstd pre_w); sigmoid(0.1767767 x (hi + lo)^T) + bias, top-8 (ties
  to the lower expert); scores = picked sigmoids / (sum + 1e-15); xin = bf16(x w_pool^T) rstd xn_w.
- **K3**: per pick, aiter's `_mxfp4_quant_op` rule on bf16 xin against the E2M1 x E8M0 gate | up rows,
  mid = MXFP4(bf16(silu(gate) up)), y = bf16(mid w2^T), v = bf16(y pen_w), out += score v / rms(v);
  part = bf16(out); branch = bf16(part w_unpool^T); post = bf16(branch pfw); nb2 = bf16(post rstd);
  res = alpha2 new + beta2 nb2; attn_out = res rstd npw.
- **Next K1** (a sliding layer with RoPE): y = bf16(attn_out) w_qkvg1^T; q = rope(bf16(bf16(norm(y_q))
  5.473)), k = rope(bf16(norm(y_k))), v = y_v, gate = y_g; interleaved RoPE on bf16 cos / sin with rounded
  products; cache rows `swa_map[slots]` take fp8(bf16(k)) and fp8(v).

Returned bf16 / fp8 outputs are the values before their final rounding. The attention layer is a
**full-attention** layer, whose work grows with context (a sliding layer reads at most 2047 + 8 keys); the next
layer is a sliding layer with RoPE, the repository layer test's configuration (the checkpoint's layer pattern
is not in the source).

## Inputs, shards and outputs

`inputs.py` documents every tensor. The generator returns logical (all-head, all-expert) tensors; `shard_inputs`
gives rank r its 16 query heads / 2 KV heads (q, g, caches, w_o columns, w_qkvg1 rows), the TP slices of every
expert (gate rows 512r.., up rows 4096+512r.., down rows 512r..) and the replicated rest; `rank_expected`
slices the sharded outputs. Outputs: `o` (rank's heads), `ids`, `scores`, `xin`, `attn_out`, `k_out`, `v_out`;
mutable: `q` and `g` (overwritten with the next layer's), `res`, `k_cache1`, `v_cache1`. `ranks_agree`:
ids, scores, xin, attn_out, res.

Data: q per-head normed x 5.473, g, K / V unit normal (fp8), weights N(0, 0.02) bf16, norm weights 1 + N(0,0.1),
gates U(0.5, 1) / U(0, 0.5), MXFP4 codes uniform with E8M0 scales 117..121, contexts ragged (others in
[ctx/2, ctx]), pages scattered, distinct sliding slots for the step's rows.

**Routing margins.** The top-8 must not flip between reference and kernel. Each token's 8 experts are planted:
expert e owns hidden columns 32e..32e+31; the residual stream carries levels 85, 75, .., 15 along random signs
of its experts' blocks; the router column reads its block divided by alpha * pre_w; noise N(0, 5e-4) and a bias
U(-1e-3, 1e-3). Measured on CPU (seed 12345, cases r1_c8, r2_c3000, r8_c4099): the smallest gap between
consecutive biased scores among a token's top 8 is 0.0135 and between the 8th and 9th 0.020 (the repository's
own near-tie tolerance is 1e-3); 61 / 101 / 226 distinct experts are picked at 8 / 16 / 64 rows.

## Workloads

Derived from corpus statistics (AgentX `cc-traces-weka-062126`, decode-step-weighted context: p10 52K, p25 108K,
p50 220K, p75 426K, p90 632K, no request above 1M); **not yet confirmed by a live capture**. Phase: decode /
verify only (the monokernels serve no prefill). Rows: 8, 16, 64 (R = 1, 2, 8 requests of 8 draft rows; three
compiled widths for the 10-minute budget). Context cap assumed 1,048,576 (the model's maximum context is not in
the source; AgentX's maximum is 990K). Seven performance cases span 16K..2M in R x ctx (128x):
r1 at 16K / 128K / 256K / 512K, r2 at 256K, r8 at 64K / 256K. Correctness-only: r1_c8 (minimum), r2_c3000
(non-power-of-two), r8_c4099 (not a multiple of 16), r1_c25859 (the step's rows straddle a chunk boundary
inside a two-chunk partition: the mask fix's regression case), r1_c1048576 (cap).

Estimated baseline time per rank (bytes at ~5 TB/s plus ~40 us of hand-offs; TP-expert slices of ~57 / 101 /
221 distinct experts at 3.2 MB each, KV 256 B a key a rank): r1_c16k ~90 us, r1_c128k ~100 us, r1_c256k
~105 us, r1_c512k ~120 us, r2_c256k ~145 us, r8_c64k ~230 us, r8_c256k ~290 us. All are above the 10 us
floor. The repository's own measurement (16 rows, ~95K context: 11.88 ms for a 62-layer step) is consistent.

## Comparison

- `ids` exact; `o`, `k_cache1`, `v_cache1` the default bound.
- `scores`: tolerance rtol 1e-2, atol 1e-4 (the repository test's 1e-2 relative + 1e-3 absolute). Each row has
  8 elements, so `baseline_relative`'s row RMS is an 8-sample estimate: a float32 implementation with another
  GEMM order failed it on one of 512 elements.
- `xin`: `baseline_relative` with `max_violation_fraction: 1e-3` (measured 0.004-0.011% violations for that
  float32 stand-in, from bf16 rounding flips of the pooled row).
- `res`, `attn_out`, `q`, `g`, `k_out`, `v_out`: `sqnr` >= 27 dB. They lie downstream of the experts' MXFP4
  activation quantization, which turns tiny rounding differences into E2M1 code flips: the float32 evaluation
  of the reference reaches only 37-44 dB against float64, and two float32 evaluations with different GEMM orders
  fail `baseline_relative` on 0.004-0.14% of `res` elements with peaks of 15-30x the bound even at k = 4.
  The baseline kernel itself (bf16 intermediates, so more E2M1 flips than a float32 evaluation) measured
  29.5-31.3 dB on MI355X over every correctness case, both seeds and all 8 ranks (192 samples per output;
  medians 30.3-30.5 dB), so a 30 dB bound rejected it. 27 dB keeps a 2.5 dB margin below the baseline's
  lowest sample and still rejects a wrong branch (a zeroed FFN branch is ~14 dB on `res`). It cannot see a
  single dropped pick in one token (~2.5% of one row).

## Unverified

Nothing has run on a GPU: FlyDSL 0.3.2 compile of the flattened package, the dataclass / slotted-class
replacements and the attention mask fix (the bug itself was found by reading the kernel and simulating its
partition schedule, not on hardware), the vendored aiter `buffer_ops` revision against the image, numerics and the bounds against the
real baseline, timings, the 64-row TP8 merged launch with TP experts (the repository runs only 8 / 16 rows
with TP experts at TP8; its 64-row runs use the replicated pooler, as here), the uncached-scratch cost, compile
time per width (the main risk for the 10-minute budget), and the CPU-side generation cost on the framework's
eight ranks (each rank draws ~7 GiB of logical inputs per draw, mostly the 256 experts).
