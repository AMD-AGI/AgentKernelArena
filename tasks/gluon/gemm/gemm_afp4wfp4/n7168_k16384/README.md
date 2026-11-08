# afp4wfp4 GEMM (Gluon baseline), N = 7168, K = 16384

`out = dequant(a) @ dequant(w)^T`: MXFP4 activations `a` and weights `w`, both stored as two e2m1 codes per byte (the
even K index is the low nibble) with uint8 e8m0 scales (`2^(s - 127)`), one per 32 values of K. `a` is [m, 8192] bytes
with `a_scales` [m, 512]; `w` is [7168, 8192] bytes with `w_scales` [7168, 512]. The block products are accumulated in
float32 and the result is rounded to bfloat16. Operator, interface, input generator and reference are those of the
Triton scenario `tasks/triton/gemm/gemm_afp4wfp4/n7168_k4608`; the shape and the baseline (a Gluon kernel) differ.

**Reference.** `reference.py` dequantizes both operands from codes and scales and multiplies in float64, row-chunked
to bound memory. It is independent of the baseline and of the quantizer in `inputs.py`.

**Inputs.** `inputs.py` draws normal activations (std 1) and weights (std 0.05) and quantizes them like AITER's
dynamic MXFP4 quantization: the scale of a 1x32 group is the group maximum rounded to a power of two minus the
exponent of the largest e2m1 value, and codes round to nearest with ties to the even code, saturating at 6. In
production the activation quantization is a separate upstream kernel, so it is part of the input, not timed.

## Scenario and workload origin

N = 7168 and K = 16384 are the attention output projection (`o_proj`) of DeepSeek-V3. Config citation, DeepSeek-V3
`config.json` (Hugging Face): `hidden_size` 7168 (output channels N), `num_attention_heads` 128 and `v_head_dim` 128
(contracted dimension K = 128 x 128 = 16384). The published DeepSeek-V3 checkpoint is FP8; the MXFP4 format (OCP
microscaling: e2m1 values, e8m0 scale per 32 values) is the operator's own, applied here to a real 7168-wide
layer shape of the model (AITER's MXFP4 GEMM tests use other 7168-wide shapes, for example N = 7168, K = 4608, which
the Triton scenario of this operator uses; this scenario picks the layer whose dimensions are public config fields).

`m` is the token count of the call. The fp4 weight is 58.7 MB, so reading it alone takes about 13 us on the
roofline model; decode-sized batches are therefore real performance cases (the first has `m = 16`), and the span runs
to very large batched prefills: 10 performance cases from `m = 16` to `m = 65536` (4096x). The largest case writes a
940 MB output; its float64 reference output is 3.8 GB. Correctness-only cases: `m = 1` (axis minimum),
`m = 3` (not a power of two), `m = 17, 100, 129, 257, 1000` (not multiples of 16 and above 16; 129 and 257 straddle
the 128- and 256-row tiles).

## Baseline

AITER's Gluon `_gemm_afp4wfp4_kernel` (`aiter/ops/triton/gluon/gemm_afp4wfp4.py`, gfx950 only: `mfma_scaled` with e2m1
operands and 32x32x64 MFMA, `buffer_load`, LDS double buffering) with its launcher. The Gluon table
`configs/gfx950/gluon/gemm/GEMM-AFP4WFP4.json` has a single entry used for every shape: 256x256 tiles over K steps of
256 values, 8 warps, group size 2, `NUM_KSPLIT = 1`. With K = 16384 and no split-K the Gluon split-K reduce kernel
`_gemm_afp4wfp4_reduce_kernel` is never launched; it is a Gluon kernel, kept unchanged with the rest of the upstream
wrapper (so a candidate copy of the baseline keeps the whole production path). One Gluon kernel is launched per call.
`remap_xcd` and `pid_grid` are `@triton.jit` helpers that the Gluon kernel calls inline, as upstream does; they are
copied unchanged into `baseline/pid_preprocessing.py` and are never launched. The kernel code is byte-identical to
upstream; the launcher is renamed `run`, writes into the preallocated output and drops `skip_reduce`, the logger, the
config-file lookup and the fp4-availability assertion (see `provenance` in `task.yaml`).

Provenance: `ROCm/aiter` at `652f2d4b768239aefd515c929e84817681ff598d`, MIT. The upstream `gemm_afp4wfp4.py` has no SPDX
or copyright header, so the repository LICENSE (MIT, Copyright (c) Advanced Micro Devices, Inc.) applies and is included
as `baseline/LICENSE`; `pid_preprocessing.py` keeps its upstream header (`SPDX-License-Identifier: MIT`,
`Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`). The file's history starts with the AITER
commit "gluon gemm_afp4wfp4"; neither the file nor its directory records a third-party origin.

## Estimated timing

Roofline estimate (fp4 4e15 flop/s, 4.5e12 B/s, 4 us fixed): `m = 16` is bound by the weight read, 13 us + 4 us, about
17 us; `m = 65536` is 1.5e13 flop, about 3.8 ms. The 256x256 tile makes the real kernel slower than the roofline at
small `m` (28 tiles for 256 compute units, each padded to 256 rows).

## Notes

- **Upstream Gluon tests for this kernel are disabled** (`op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py` at
  the pinned commit skips `impl == "gluon"` with the note "Fix gluon instr shape then enable gluon tests conditionally
  on 950"), so the task's own correctness cases, not upstream's tests, cover the kernel on gfx950 with the image's
  Triton 3.8.
- Gluon API: `mfma_scaled`, `DistributedLinearLayout` and `gl.amd.cdna4.buffer_load` on gfx950 in Triton 3.8.
- Deterministic: one K loop per output tile, no split-K and no atomics. The float32 accumulation order differs between
  candidates; the `baseline_relative` bound absorbs it.
- The 65536-token case needs the 3.8 GB float64 reference output next to 0.94 GB of baseline and candidate outputs.
