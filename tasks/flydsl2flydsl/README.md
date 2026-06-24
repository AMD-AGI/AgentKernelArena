# FlyDSL (`flydsl2flydsl`) Tasks

## Prerequisites

These tasks require [FlyDSL](https://github.com/ROCm/FlyDSL) and a ROCm-enabled AMD GPU.

Recommended setup from the repository root:

```bash
make setup-flydsl
```

This installs FlyDSL into the project virtual environment and verifies both FlyDSL import and ROCm PyTorch GPU availability.

Manual install, if needed:

```bash
pip install flydsl
```

Verify: `python3 -c "import flydsl; print(flydsl.__version__)"`

## Hardware support

Benchmark hardware is **AMD MI300X = gfx942 (CDNA3)**. Not every catalogued
kernel runs there. Per-task hardware support is machine-readable in each
`config.yaml` (`platform_support.required_arch` / `runnable_on_gfx942` /
`status`); see **[`ARCH_SUPPORT.md`](ARCH_SUPPORT.md)** for the full matrix.

`pa_decode_fp8_kernel` additionally requires **`aiter`** to be available in the
environment (used for fp8 KV quantization and the paged-attention metadata/reduce
reference); the other gfx942 tasks need only FlyDSL.

Notably, `fp8_gemm_4wave_kernel` and `fp8_gemm_8wave_kernel` are **gfx950/CDNA4-only**:
they emit the CDNA4-only 16B `buffer_load_lds` intrinsic, which the gfx942 LLVM
backend cannot legalize (`LLVM ERROR: Do not know how to expand this operator's
operand!`, exit 134). They are marked `status: skip` and excluded from the
gfx942 benchmark/validation set.

## Compute pattern (L1 / L2 / L3)

Examples are grouped by **compute pattern** (not by any other “difficulty” scale):

- **L1** — Elementwise or single per-row reduction; threads work independently.
- **L2** — No matrix multiply, but requires cross-thread cooperation via shared
  memory (LDS) or a fused multi-step pass.
- **L3** — Contains a matrix multiply (MFMA): GEMM or attention, with software
  pipelining, double-buffered LDS, split-K, or paged / FP8 KV-cache.

### L1

| Task | Reason |
|------|--------|
| `softmax_kernel` | Numerically stable softmax, register-buffered per-row reduction, exp2 fast path. No matmul, no cross-thread cooperation. |
| `rmsnorm_kernel` | RMSNorm with float32 accumulation. Per-row reduction; the multiple kernels are just dtype variants. |

### L2

| Task | Reason |
|------|--------|
| `layernorm_kernel` | LayerNorm with shared-memory (LDS) reduction and fused `x*scale+bias` epilogue. No matmul. |
| `fused_rope_cache_kernel` | Fused rotary embedding + KV-cache write; cross-lane `ds_bpermute` shuffles, vectorized buffer_load/store. No matmul. |
| `topk_gating_softmax_kernel` | Fused MoE gating softmax + top-K + optional renormalize + token_expert_indices. |
| `moe_sorting_kernel` | MoE token/expert sorting (oneshot + multiphase); CDNA-focused. |
| `silu_and_mul_fq_kernel` | Fused activation (SiLU/SwiGLU) + optional quant + sorted scales for split-K MoE stage-1. |

### L3

| Task | Reason |
|------|--------|
| `flash_attn_func_kernel` | Fused multi-head attention: online softmax, MFMA32 GEMM, DMA-to-LDS, software-pipelined QK/PV. |
| `hgemm_splitk_kernel` | Half-precision GEMM with split-K, double-buffered LDS, pre-shuffled B. |
| `pa_decode_fp8_kernel` | Paged-attention decode with FP8 KV-cache and multi-partition reduce; most complex kernel. **Requires `aiter` available in the environment** (fp8 quant + pa metadata/reduce reference). |
| `blockscale_preshuffle_gemm_kernel` | FP8 blockscale GEMM with preshuffled B and MFMA epilogue. |
| `fp8_gemm_4wave_kernel` | FP8 GEMM (4-wave) with row scales. **gfx950/CDNA4-only** — emits the CDNA4-only 16B `buffer_load_lds`; aborts at codegen on gfx942 (`status: skip`, see `ARCH_SUPPORT.md`). |
| `fp8_gemm_8wave_kernel` | FP8 GEMM (8-wave) with row scales, ported from HipKittens CDNA4. **gfx950/CDNA4-only** — emits the CDNA4-only 16B `buffer_load_lds`; aborts at codegen on gfx942 (`status: skip`, see `ARCH_SUPPORT.md`). |
| `preshuffle_gemm_v2_kernel` | Preshuffle GEMM v2 (layout API; fp8/fp16/bf16). |
| `pa_decode_swa_kernel` | Paged-attention decode for sliding-window (partitioned) paths. |

## Shared vendored modules (`tasks/flydsl2flydsl/kernels/`)

FlyDSL helper modules used by several examples (same `kernels.` import path as `kernels_common.py` / `tensor_shim.py`):  
`mfma_epilogues.py`, `mfma_preshuffle_pipeline.py`, `fp8_gemm_utils.py`, `layout_utils.py`, `moe_common.py`, `dpp_utils.py`, `pa_decode_swa.py`, `preshuffle_gemm.py`.

## Benchmark config

The benchmark config (`config_geak_flydsl.yaml`) that lists these tasks (grouped by
the same L1/L2/L3 pattern) lives on the **`geak-flydsl-common-benchmark`** branch.
