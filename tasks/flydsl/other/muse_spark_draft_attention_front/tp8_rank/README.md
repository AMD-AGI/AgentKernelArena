# Muse Spark DFlash2 draft attention front (D-pre), one TP8 rank

The first of the four launches of each DFlash2 draft layer's attention half in sgl-amd-mono's Muse Spark
draft path (D-pre, D-attn, D-post, then D-MLP), for one rank of the draft's TP8 group. One persistent
256-CTA FlyDSL launch for `rows` rows in 8-row draft blocks (one block a request):

1. the input norm's fused add, r = x (the draft's first layer) or bf16(res + x), written out as `res_out`;
2. h = RMSNorm(r) w_in (eps 1e-5);
3. the attention conv's kernel projection, coef = h W_kp^T: 2048 coefficients a row (2 sides x 2 taps x
   512 groups of 16 channels), float32, also read by D-post (side 1);
4. conv0 (side 0) within each 8-row block: hc[t] = (base[0, 0] + coef[t, 0, 0, g]) h[t] + [t % 8 >= 1]
   (base[0, 1] + coef[t, 0, 1, g]) h[t - 1];
5. the rank's q | k | v projection (8 query heads, 1 KV head, head size 128: 1280 rows);
6. per-head RMSNorm of q and k with their weights and neox RoPE from the float32 cos | sin table;
7. q out (bf16); k and v as float8_e4m3fn into the layer's KV cache at the rows' slots.

D-pre exchanges nothing with the other ranks, so the task is single-GPU. `reference.py` (float64)
documents the exact semantics.

**Model support is semi-experimental.** Muse Spark and its DFlash2 draft (RadixArk/Muse-Spark-Open-DFlash2:
5 layers, hidden 8192, two-tap grouped convolutions) are registered in sgl-amd-mono's `SPECS` but absent
from the repository README's model table; the model landed in one PR (#26), has no end-to-end server test,
and its default served widths are 8 and 16 rows (`gfx950/config.py`: "32 and 64 rows are under test").

## Baseline and provenance

`baseline/` is vendored from sgl-project/sgl-amd-mono at `52486703fbf27e77f05b573e4d931d4eec87118d`
(Apache-2.0; `common/ops.py`, `plan.py`, `sync.py`, `mx.py` adapted from ROCm/ATOM, MIT):
`muse_spark/gfx950/draft_attn_half.py` (D-pre only), the pieces of `draft_mlp.py` D-pre imports (the shapes,
the GEMV helpers and the coefficient / conv0 stages), `config.py`, `layout.py`, `device.py`, the FlyDSL 0.3.2
bridge `compat.py`, and `common/{guard,epoch,mx,ops,plan,sync}.py` in the flattened form of
`tasks/flydsl/other/muse_spark_layer/tp8` (relative imports, frozen dataclasses / a slotted class instead
of `msgspec`, the 0.3.2 bridge called directly instead of `compat.install()` patching modules,
environment reads replaced by their defaults, no source-file digest); `buffer_ops.py` and `dpp_utils.py`
are ROCm/aiter `dbd8bf5bd` (the copies of `tasks/flydsl/other/qwen38_flash_next_decoder`). SGLang's ROCm
images pin sgl-amd-mono `e52803f9` (11 commits older); the baseline is vendored from `52486703`. No kernel
or stage code is changed.

`baseline/kernel.py` is new. It makes the draft runner's `DPreBuild(rows)` and passes
`draft.py::launch_attn`'s D-pre arguments (int64 positions and cache slots, as SGLang passes them). The
runner keeps one scratch and one epoch for the draft's launches and moves the epoch on once a step
(`epoch[0] += 1`), so no mailbox pair of an earlier step carries the current tag. Here the scratch (30 MB,
sized for 64 rows) and the epoch / guard words share one device buffer, created zeroed at the first
(eager) call, and every call moves the epoch on after its launch with a new one-thread FlyDSL kernel (the
runner's `add_` is a PyTorch op, which the candidate rules leave to kernels). Clearing the mailbox regions
instead would put a 28 MB memset into every timed call.

## Inputs

Weights are random at the draft's real shapes (W_kp 2048 x 8192 and W_qkv 1280 x 8192 bf16, 54.5 MB a rank),
drawn as the repository's D-pre test draws them (`test/kernels/test_muse_spark_dhalf.py`): x N(0, 0.5),
res N(0, 1), norm weights 1 + N(0, 0.1), W_kp N(0, 0.01), base 1 + N(0, 0.05), W_qkv N(0, 0.02). `inputs.py`
makes the neox RoPE table (base 10000, the test's; the checkpoint's is not in the source; 65,536 positions,
which does not change the work), consecutive positions within each 8-row block, distinct int64 cache
slots and a stale fp8 cache whose other slots must stay unchanged. No discrete decision depends on the
data, so no margins are needed.

## Workloads and the production-range span exemption

Rows are requests x 8, launched at the compiled widths 8, 16, 32 and 64 (a step pads to the next width;
8 and 16 are the default served widths). D-pre streams the same 54.5 MB of weights whatever the rows, so
its time hardly depends on them, and no context axis applies (it reads one RoPE row and writes one cache
slot a row). The production range therefore spans 8x in var-axis size, below the 100x rule: the task
declares the **production-range span exemption** (`workloads.json` `performance_span`,
`workload_evidence.json`). Evidence (corpus statistics, **not yet confirmed by a live capture**): the
AgentX per-session in-flight distribution (`semianalysisai/cc-traces-weka-062126` rev 23f152f6) at
concurrency 1, 2, 4 and 8, rows padded to the compiled widths, steps above 64 rows excluded (they leave
the monokernel path): 8 rows 26.7%, 16 rows 25.6%, 32 rows 24.4%, 64 rows 23.3%. Observed range 8..64,
perf range 8..64 (every width is estimated above 10 us), perf span 8x.

Performance cases (4): 8 rows (layers 1-4), 8 rows with `first` (layer 0, one of the draft's five D-pre
launches a step), 16 rows, 64 rows. Correctness only: 24 rows (non-power-of-two; three requests) and
`first` at 16 and 64 rows. Compiled widths: 8, 16, 24 and 64; 32 is left out to keep the evaluation within
its 10-minute budget. A var axis whose domain holds at most 16 units of its `multiple_of` (8..64 in steps
of 8) has no unaligned value, so the unaligned boundary case does not apply.

Estimated baseline time: the weights alone take ~11 us at ~5 TB/s cold HBM, and the W_qkv stream (21 MB)
starts only after three mailbox hand-offs (the norm statistics, the coefficient partials and their
reduce); with the launch and the final head stage, ~20-25 us at 8 and 16 rows and ~25-32 us at 64 rows
(the mailbox partials grow from 3.4 MB to 27 MB). All four performance cases are expected well above the
10 us floor; QA-SPAN-EVIDENCE checks the 8- and 64-row endpoints on MI355X.

## Comparison

`res_out`, `coef`, `k_cache` and `v_cache` use the default `baseline_relative` bound. `q` uses `k: 4` and
`max_violation_fraction: 1e-3`: it lies at the end of a bf16 chain (h, hc, qkv, the normed heads, RoPE)
whose rounding points differ between correct implementations. On the CPU, with a float32 emulation of
the kernel's roundings as the baseline, an emulation of SGLang's module chain (bf16 coefficient GEMM
output and bf16 conv ops, float32 coef) violates the default bound on 1.3-1.6% of the `q` elements and the
k = 4 bound on at most 0.012% (none beyond 16x), while a float32 implementation that rounds only the
outputs passes every default bound. A missing conv, a missing RoPE, a flipped `first` and positions off by
one all fail (`q`: 36-99% of its elements over its bound). The emulated kernel reaches 48 dB SQNR on `q`, 56 dB
on `coef` and 31 dB on the fp8 cache rows against the reference (rows 8, 16, 24, 64; `first` both ways).

## Unverified

Nothing here has run on a GPU: FlyDSL 0.3.2 compilation of the flattened package and of the new epoch
kernel, the persistent-buffer scheme, numerics and the bounds against the real baseline, compile time for
four widths, and timing (the 10 us floor at 8 rows is estimated). The CPU checks covered the contract and
the exemption (`check_task`), the baseline's candidate-rule audit, the inputs and reference for every
case, and the emulation above.
