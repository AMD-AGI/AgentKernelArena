# FlyDSL (`flydsl2flydsl`) Tasks

## Prerequisites

These tasks require [FlyDSL](https://github.com/ROCm/FlyDSL) and a ROCm-enabled AMD GPU.

Recommended setup from the repository root:

```bash
make docker-setup-flydsl
```

This installs FlyDSL into the Docker runtime's persistent pip user-base when the
selected image does not already provide it, then verifies the import and ROCm
PyTorch GPU availability.

For manual inspection, enter the supported runtime first:

```bash
make docker-shell
python3 -c "import flydsl; print(flydsl.__version__)"
```

## Task Difficulty

| Task | Difficulty | Description |
|------|-----------|-------------|
| `softmax_kernel` | Easy | Numerically stable softmax with exp2 fast path |
| `rmsnorm_kernel` | Easy | RMSNorm with float32 accumulation |
| `layernorm_kernel` | Medium | LayerNorm with shared-memory reduction |
| `fused_rope_cache_kernel` | Medium | Fused rotary position embedding with KV-cache |
| `flash_attn_func_kernel` | Hard | Fused multi-head attention with online softmax, MFMA32 GEMM, DMA-to-LDS |
| `hgemm_splitk_kernel` | Hard | Half-precision GEMM with split-K, double-buffered LDS, pre-shuffled B |
| `pa_decode_fp8_kernel` | Hard | Paged attention decode with FP8 KV-cache, multi-partition reduce |
