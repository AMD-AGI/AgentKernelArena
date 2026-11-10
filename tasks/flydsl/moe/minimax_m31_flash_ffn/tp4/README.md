# MiniMax-M3.1-Flash FFN monokernel, TP4

One sparse decoder layer of MiniMax-M3.1-Flash after SGLang's attention, as one persistent FlyDSL launch per
rank of a tensor-parallel-4 group: o_proj, the attention all-reduce and residual add, the Gemma
post-attention RMSNorm, the FP32 router, the routed and shared experts and the MoE all-reduce. It serves
the model's decode and DSpark verify steps of 1 to 16 rows on the 57 sparse layers.

## Operator

Per step of S rows, on rank r (`task.yaml` describes it in full; `ffn_numerics.py` is the float64 model):

1. **o_proj**: per-token FP8 of the rank's attention output `attn` [S, 2048] (AITER `per_token_quant_hip`:
   scale = amax / 448, RNE E4M3) times the rank's o_proj columns in SGLang's rowwise-FP8 decode copy
   (`wo` E4M3 [6144, 2048], `wos` per-row scale), a bf16 partial.
2. **Attention all-reduce and residual**: the 4 partials summed in rank order to bf16 (a);
   `res_out` = bf16(a + res).
3. **Norm**: Gemma RMSNorm of a + res (eps 1e-6, gain 1 + `norm_w`) to bf16 (xn).
4. **Router**: FP32 logits xn . `router_w`; top 4 of sigmoid(logits) + `bias` (the bias only selects);
   weights = the picks' sigmoids / their sum x 2.0.
5. **Experts** on the rank's intermediate slice (768 of 3072): the routed experts in MXFP4 (E2M1 codes, low
   nibble first, E8M0 scale per 32; AITER `shuffle_weight (16, 16)` codes and `e8m0_shuffle` scales) and the
   shared expert in the checkpoint's MXFP8 (E4M3, E8M0 per 32; the launch's a16w8 lane layout). gate | up =
   xn . w13^T, SwiGLU-OAI min(g, 7) sigmoid(1.702 min(g, 7)) (clamp(u, -7, 7) + 1) to bf16, then . w2^T;
   the rank's partial = bf16(shared + route-weighted routed outputs).
6. **MoE all-reduce**: `out` = the 4 partials summed in rank order, bf16.

Both all-reduces happen inside the launch: every rank stores its partial rows into every peer's symmetric
buffer with system-scope 64-bit stores (tagged pairs) and polls its own. `out` and `res_out` are the
all-reduced rows, so every rank writes the same values, bitwise (`ranks_agree`).

## Multi-GPU contract

- `distributed.world_size` 4 (the production TP). `symmetric_bytes` = `ffn.peer_bytes(4)` = 5,243,136:
  both parity halves of the attention (three bf16 a 64-bit pair) and MoE (a bf16 pair a 64-bit pair)
  partials, then the guard's peer error flags. The framework zeroes it before every call, so the launch's
  epoch-0 tags are always fresh.
- Logical inputs stack the rank-specific tensors on a leading TP axis (`inputs.py`); `shard_inputs` takes
  rank r's slice, `to_kernel_layout` prepares the load-time layouts, `rank_expected` gives every rank the
  replicated `out` / `res_out`.
- The kernel's other persistent state (the epoch / guard block, the slot mapping and the mailbox scratch)
  is rebuilt inside every call by `baseline/kernel.py`: one workspace allocation, one zeroing of its
  tagged regions and one fill of the 10 s wait deadline. These two small device writes are inside the
  timed graph, as in the other sgl-amd-mono tasks; a candidate may organize its state differently.
- The persistent grid is 256 CTAs of 512 threads that must all be co-resident (one a CU): a 256-CU gfx950
  GPU used exclusively, as in the framework's timing.

## Provenance

- Baseline: sgl-project/sgl-amd-mono @ `52486703fbf27e77f05b573e4d931d4eec87118d` (`minimax_m31/ffn.py`,
  `sync.py`, `a16w4.py`, `a16w8.py`, host behaviour of `runner.py` / `spec.py`; `common/ops.py`, `sync.py`,
  `guard.py`, `epoch.py`, `plan.py`), Apache-2.0. sgl-amd-mono adapted the kernel from vLLM
  (vllm-project/vllm@a50259bb, `vllm/models/minimax_m3/amd/specialized_decode/mono`, Apache-2.0); its
  `common/` primitives are partly adapted from ROCm/ATOM 97359d6 (MIT). AITER's FlyDSL `buffer_ops.py` /
  `dpp_utils.py` (Apache-2.0) at dbd8bf5bd. See `baseline/LICENSE`. SGLang ROCm images pin sgl-amd-mono
  e52803f9 (11 commits older); the baseline is 52486703.
- **FlyDSL 0.3.2 port.** The kernel was written for the SGLang image's FlyDSL (0.4.0 in the smoke image) and
  its newer LLVM. The AgentKernelArena runtime ships FlyDSL 0.3.2, so `baseline/ffn_sync.py` emits
  `minimax_m31/sync.py`'s monotonic loads / stores as llvm-dialect atomics, its
  `llvm.amdgcn.av.load.b128` as the sc1 buffer load it lowers to, and its `amdgcn-av:none` fences as the
  waits they lower to beside workgroup-scope fences (compiler order only). These replacements emit the same
  cache bits and waits that the upstream doc says its operations lower to, but **they are unverified on
  GPU** (see below). The device code of `ffn.py`, `a16w4.py` and `a16w8.py` is otherwise unchanged;
  `task.yaml` (`baseline.provenance.modifications`) lists every change.
- **Hardware.** The sgl-amd-mono README lists MI350X (gfx950) for MiniMax-M3.1-Flash. Nothing in the kernel
  is MI350X-specific: it is gfx950 code (the spec's device key is `gfx950`), its CU-residency check reads the
  device's CU count at run time, and MI350X and MI355X both have 256 CUs and the same 100 MHz
  `s_memrealtime` clock. This task is validated on MI355X.
- **Checkpoint.** MiniMax-M3.1-Flash ships NVFP4 routed experts and MXFP8 dense / shared-expert weights;
  SGLang re-quantizes the routed experts to MXFP4 at load (the layout the launch reads in place) and the
  spec keeps the shared expert's checkpoint MXFP8. The task draws MXFP4 / MXFP8 weights directly.
- Serving the model on this path also needs SGLang PRs #45925 and #45936; the task does not depend on them.

## Reference and inputs

`reference.py` (via `ffn_numerics.py`) is independent float64 PyTorch. It applies the operator's own
roundings (per-token E4M3 of the attention output, bf16 o_proj partials, the bf16 attention all-reduce, the
bf16 normalized input, the bf16 SwiGLU-OAI intermediate, each rank's bf16 MoE partial) and computes the
routed experts from their exact MXFP4 values (E2M1 x 2^(e8m0 - 127)) and the shared expert from its MXFP8
values; it does not call AITER's fused MoE, which sgl-amd-mono's own test uses as the routed-expert
reference. Both outputs use the default `baseline_relative` bound: the baseline's own error (its
AITER-style combine, bf16(weight x output) added in bf16 with the shared expert first, its FP32
accumulation orders and its v_rcp-based FP8 quantization) sets the scale a candidate is held to.

Value ranges follow sgl-amd-mono's kernel test `test/kernels/test_minimax_m31_ffn.py`: attention output
3 N(0, 1), residual 4 N(0, 1), o_proj 0.02 N(0, 1) before its rowwise FP8, random MXFP4 codes with scales
2^-8 .. 2^-5 (gate / up pre-activations of a few units, so the SwiGLU limit of 7 is exercised), the shared
expert 2 N(0, 1) in FP8 with the same scales, norm gain 0.1 N(0, 1), bias U(-0.03, 0.03).

**Routing margin.** Top-4 routing is the only discrete decision. The generator plants the router weight:
a random 0.02 N(0, 1) matrix plus the minimum-norm correction in the span of the rows' normalized inputs
(computed exactly as the reference does) that gives every row 4 random winners with logits in [1.5, 3] and
the others in [-4, -0.25]. sigmoid + bias then separates each row's 4th and 5th choice by >= 0.32
(the generator refuses less than 0.1), far beyond the kernel's FP32 logit and bf16 input differences.

## Workloads

Derived from corpus statistics, **not confirmed by a live capture**: the MiniMax-M3.1-Flash checkpoint is
not on the CRS cluster, so no AgentX replay through this model ran. The source is the AgentX corpus
(semianalysisai/cc-traces-weka-062126 rev 23f152f6, 393 traces, 98,827 requests) and the model's serving
configuration in sgl-amd-mono 52486703 (`--tp 4 --speculative-algorithm DSPARK`; the e2e servers run DSpark
verify steps of 8 rows a request and plain decode steps; the spec compiles every width 1-16).

The FFN reads no KV cache: its work does not depend on the context length, so there is no context axis and
the 100x span rule is unreachable. The only var axis is `rows` (requests in a decode step, or requests x 8
in a DSpark verify step), and the kernel serves 1-16 rows (larger steps run SGLang's own path). The task
therefore uses the **production-range span exemption** (`workloads.json` `performance_span`,
`workload_evidence.json`): observed range 1-16, perf range 1-16 (span 16x).

Row weights (`workload_evidence.json`): for each serving mode (decode, DSpark verify; equal weight) and
AgentX concurrency point C in {1, 2, 4, 8} (equal weight), R = the sum of C per-session in-flight counts
(k = 1 90.0%, 2 5.4%, 3 2.1%, ...), rows = R or 8 R, renormalized over the steps the kernel serves:

| rows | 1 | 2 | 4 | 5 | 8 | 9 | 16 | others |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weight | 15.7% | 15.1% | 12.3% | 3.2% | 23.8% | 4.0% | 15.3% | 10.8% |

Cases (every case is checked for correctness and timed): rows 1 (decode C1; the minimum), 5 (decode C4;
the non-power-of-two boundary), 8 (DSpark verify of one request; decode C8) and 16 (DSpark verify of two
requests; the largest width). They carry 58% of the served steps' weight. FlyDSL compiles the persistent
kernel per row count and the task needs at least four distinct row counts (four timed cases, one var
axis), so the cases are the minimum four builds; rows 2 (15.1%) and 4 (12.3%) are left out to keep the
10-minute evaluation budget (add rows 2 first if the measured wall time allows), and no case between 9
and 15 rows is checked (those builds take the code paths of the 16-row one: no XN in LDS and the 4-wave
o_proj split). Prefill shapes do not apply: the kernel serves decode and verify steps only.

Estimated baseline time (bytes read once from HBM at ~5 TB/s; the dependent stages and two cross-GPU
hand-offs add latency on top): fixed 31.7 MB a rank (o_proj 12.6 MB, router 3.1 MB, shared expert 15.9 MB)
plus 7.5 MB a distinct routed expert (about 4, 19, 28.7 and 50 distinct experts for 1, 5, 8 and 16
rows of random routing): ~12 (rows 1), 35 (5), 50 (8) and 82 us (16). Rows 1 is the case closest to the
10 us floor.

## Unverified (no GPU was available when the task was written)

- That the FlyDSL 0.3.2 port compiles and runs (the llvm-dialect atomics and fences of `ffn_sync.py`, the
  0.3.2 forms of the MFMA / conversion builders the kernel calls), that its numerics match the reference
  within the default bound, and that the rows-1 case takes at least 10 us.
- The evaluation wall time against the 10-minute budget: four compiled widths, 4 ranks, about 3.6 GB of
  logical weights drawn per input draw.
- The workload weights, until a live capture of MiniMax-M3.1-Flash serving AgentX replaces them.
