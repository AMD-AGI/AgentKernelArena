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

## Task Difficulty (L1 / L2 / L3)

Tasks are classified by **compute pattern**:

- **L1** — Elementwise or single per-row reduction; threads work independently.
- **L2** — No matrix multiply, but requires cross-thread cooperation via shared
  memory (LDS) or a fused multi-step pass.
- **L3** — Contains a matrix multiply (MFMA): GEMM or attention, with software
  pipelining, double-buffered LDS, split-K, or paged / FP8 KV-cache.

| Task | Level | Reason |
|------|-------|--------|
| `softmax_kernel` | L1 | Numerically stable softmax, register-buffered per-row reduction, exp2 fast path. No matmul, no cross-thread cooperation. |
| `rmsnorm_kernel` | L1 | RMSNorm with float32 accumulation. Per-row reduction; the multiple kernels are just dtype variants. |
| `layernorm_kernel` | L2 | LayerNorm with shared-memory (LDS) reduction and fused `x*scale+bias` epilogue. No matmul. |
| `fused_rope_cache_kernel` | L2 | Fused rotary embedding + KV-cache write; cross-lane `ds_bpermute` shuffles, vectorized buffer_load/store. No matmul. |
| `flash_attn_func_kernel` | L3 | Fused multi-head attention: online softmax, MFMA32 GEMM, DMA-to-LDS, software-pipelined QK/PV. |
| `hgemm_splitk_kernel` | L3 | Half-precision GEMM with split-K, double-buffered LDS, pre-shuffled B. |
| `pa_decode_fp8_kernel` | L3 | Paged-attention decode with FP8 KV-cache and multi-partition reduce; most complex kernel. |
