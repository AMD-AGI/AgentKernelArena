# a8w8 GEMM with a preshuffled weight, N = 5120, K = 1280

`out = (xq @ w^T) * x_scale * w_scale`: fp8 (e4m3fn) activations `xq` [m, 1280] with one float32 scale per token
(`x_scale` [m, 1]) and fp8 weights `w` [5120, 1280] with one float32 scale per output channel (`w_scale` [5120, 1]).
Products are accumulated in float32, multiplied by the outer product of the scales and rounded to bfloat16.
The shape is a 5120-wide projection over a 1280-wide hidden state (an AITER tuned shape).

**Layout.** Production stores the weight in AITER's `(16, 16)` preshuffle layout (`shuffle_weight`), a permutation
of each 16 x 32 byte tile that makes the MFMA loads contiguous. `to_kernel_layout` in `inputs.py` applies it outside
timing, so the baseline and every candidate receive `w` shuffled (same shape, dtype and values, permuted order) while
the reference receives the logical weight. A candidate must therefore read `w` in the shuffled layout: element
`[n, k]` of the logical weight lives at the position given by reshaping to `(n/16, 16, k/32, 2, 16)` and permuting to
`(n/16, k/32, 2, 16, 16)`.

**Inputs.** `inputs.py` quantizes standard-normal activations per token and weights per channel (scale = row
amax / 448), so codes and scales are mutually consistent. The reference multiplies the exact fp8 values in float64.

**Workloads.** The two old tasks (a PyTorch-oracle FlyDSL task and an upstream FlyDSL `preshuffle_gemm_v2` task on
square 256 to 2048 problems) are both preshuffle GEMMs; the shared scenario is the first one's N = 5120, K = 1280,
which is a row group of AITER's tuned table (`aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`) for M of 1 to 32768.
The square `preshuffle_gemm_v2` problems are not kept: that kernel is the same family with a different (layout
API) code base and no production shape. Decode-sized M run below 10 us in the table (M = 256 measures 8.4 us) and are
correctness-only; performance cases run from 512 (10.5 us in the table) to 65536 tokens (128x). M = 65536 is
beyond the last table row and extends the span; it uses that row's configuration.

**Baseline.** The FlyDSL `compile_preshuffle_gemm` kernel with AITER's tuned per-M configuration (tile shape,
async copy, occupancy hint, LDS stages), vendored from the old task's self-contained copy (see `provenance` in
`task.yaml`; `baseline/LICENSE` is the AITER MIT license for the dispatch table and `baseline/LICENSE-FlyDSL` the
Apache-2.0 license of the kernel). The rows tuned to a CK kernel upstream (M = 256 and 1024) use the neighbouring
FlyDSL row. The kernel itself does not take upstream's `enable_scheduler` switch (it uses its default scheduling).

**GPU audit risks.** The table's tile shapes with 256x128x128 and 128x256x128 tiles and single LDS stage have only
been exercised by upstream, not by the old task; if one is rejected by the vendored kernel the table row needs the
nearest working configuration. The 10 us floor at M = 512 is close (10.5 us upstream, on a slightly different kernel
build). The baseline needs the FlyDSL runtime image (no `flydsl.expr.buffer_ops`; buffer operations are inlined).
