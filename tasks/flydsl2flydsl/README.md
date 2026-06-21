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
| `pa_decode_fp8_kernel` | Paged-attention decode with FP8 KV-cache and multi-partition reduce; most complex kernel. |
| `blockscale_preshuffle_gemm_kernel` | FP8 blockscale GEMM with preshuffled B and MFMA epilogue. |
| `fp8_gemm_4wave_kernel` | FP8 GEMM (4-wave) with row scales. |
| `fp8_gemm_8wave_kernel` | FP8 GEMM (8-wave) with row scales. |
| `preshuffle_gemm_v2_kernel` | Preshuffle GEMM v2 (layout API; fp8/fp16/bf16). |
| `pa_decode_swa_kernel` | Paged-attention decode for sliding-window (partitioned) paths. |

## Shared vendored modules (`tasks/flydsl2flydsl/kernels/`)

FlyDSL helper modules used by several examples (same `kernels.` import path as `kernels_common.py` / `tensor_shim.py`):  
`mfma_epilogues.py`, `mfma_preshuffle_pipeline.py`, `fp8_gemm_utils.py`, `layout_utils.py`, `moe_common.py`, `dpp_utils.py`, `pa_decode_swa.py`, `preshuffle_gemm.py`.

## Benchmark config

For the shared FlyDSL benchmark recipe, see the **`geak-flydsl-common-benchmark`**
branch, where these task paths are wired into `config_geak_flydsl.yaml` (grouped by
the same L1/L2/L3 pattern).
