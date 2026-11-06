# Routed experts with int4 weight-only (WNA16) group-quantized weights, 384 experts, top-8, hidden 7168, intermediate 256

The routed-expert block of a Kimi K2-family model (Kimi-K2.7-Code) served by vLLM on MI355X with int4 weight-only
quantized experts. Activations are bfloat16; each expert's weights are int4 codes with one bfloat16 scale per group of
32 consecutive K elements and no zero point. For each token `t` and selected expert `e = topk_ids[t, k]`:

```text
W      = (nibble - 8) * scale                # dequantized weights, group size 32 along K
h      = hidden_states[t] @ W13[e].T          # [512]: gate = h[:256], up = h[256:]
a      = silu(gate) * up
y[t,k] = (a @ W2[e].T) * topk_weights[t, k]
out[t] = sum over k of y[t, k]                # rounded to bfloat16
```

The router decisions (`topk_ids`, `topk_weights`) are given.

**Production origin.** vLLM's compressed-tensors WNA16 MoE calls `fused_experts_impl(..., use_int4_w4a16=True,
block_shape=[0, 32])`. On ROCm `should_moe_wna16_use_cuda()` is false, so both expert GEMMs always run the Triton
kernel `fused_moe_kernel_gptq_awq`; in a Kimi-K2.7-Code serving session on MI355X this path was the largest compute
leaf of the model.

**Scenario.** 384 routed experts, top-8, hidden size 7168, expert intermediate size 2048 split over tensor parallelism
8 (256 per rank), int4 group size 32, bfloat16 scales and activations, float32 routing weights, int32 expert ids. The
weight layout is vLLM's N-first uint8 layout for this kernel: `w13_qweight [384, 512, 3584]` (gate rows 0..255, up rows
256..511) and `w2_qweight [384, 7168, 128]`, two codes per byte along K with the even K element in the low nibble, and
scales `[384, 512, 224]` and `[384, 7168, 8]`. The packed weight shapes and the down-projection scale shape are those recorded for
this layer in the serving session. Only the token count varies.

**Inputs.** Activations are standard normal. `inputs.py` draws each expert matrix normal with standard deviation
1/sqrt(K) and quantizes it the way vLLM's `quantize_weights` test helper does for this kernel (type uint4b8, as in
symmetric GPTQ/compressed-tensors checkpoints): per group the scale is `max(max / 7, -min / 8)`, the code is
`round(w / scale)` clamped to [-8, 7], stored with offset 8. Every code from -8 to 7 occurs. Each token selects 8
distinct experts uniformly; its routing weights are sigmoid scores renormalized to sum to one, as the noaux_tc router
with `norm_topk_prob` does. The weights take 1.2 GB per draw.

**Workloads.** Token counts of a serving step: decode batches 4 to 256, the chunked prefill steps recorded in the
session (64, 1080 and 7211 tokens), a 2048-token prefill chunk and a 16384-token batched prefill; the performance cases
span 4096x. A 4-token step already reads about 32 experts' weights (3.1 MB each). A single-token decode, odd and
unaligned counts (3, 33, 100, 1000) are correctness-only.

**Baseline.** vLLM's `fused_experts_impl` int4 path, vendored (see `provenance` in `task.yaml`):

1. `moe_align_block_size` groups the token slots by expert, padded to `BLOCK_SIZE_M`;
2. `invoke_fused_moe_wna16_triton_kernel` / `fused_moe_kernel_gptq_awq` (unchanged): the gate/up GEMM into a bfloat16
   cache. Each int4 code is dequantized in float32 and rounded to bfloat16, then multiplied with float32 accumulation;
3. `silu_and_mul`: silu in float32 rounded to bfloat16, times up, rounded again;
4. the down GEMM with the routing weight applied to the float32 accumulator, rounded to bfloat16 per slot;
5. `moe_sum`: the 8 slots of a token added in float32 and rounded once.

The tile configuration is vLLM's tuned table for this shape on MI355X
(`E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, entry closest to the token count), as
`try_get_optimal_moe_config` selects it, used for both GEMMs. `moe_align_block_size`, `silu_and_mul` and `moe_sum` are
CUDA/HIP ops in vLLM. Here they are Triton kernels written for this task with the same contracts (`moe_align_block_size.py`,
which keeps vLLM's buffer sizing, and `moe_ops.py`). The order of the slots inside an expert's segment depends on
atomics, but every slot is computed independently, so the output does not.

**Comparison bound.** `out` uses `baseline_relative` with `k: 4` and `max_violation_fraction: 1.0e-5` instead of the
defaults (2, 0). The baseline rounds to bfloat16 five times between the inputs and the output: the dequantized weights,
the gate/up cache, the activation, its product and each slot's down-projection output. A correct implementation that
rounds at other points has errors of the same size at different elements. A CPU emulation of the baseline's rounding
against this reference (48 experts, real dimensions, 2048 tokens: 14.7M output elements) gave these results:

| Implementation | default bound (k 2) | this bound |
| --- | --- | --- |
| dequantized weights kept in float32 | fails, peak 1.96x | passes, 1.4e-7 of elements above the bound |
| silu not rounded before the multiply | fails, peak 2.03x | passes, 1.4e-7 of elements above the bound |
| split-K reordering of both GEMMs | fails, peak 1.11x | passes, peak 0.75x |
| slot outputs kept in float32 until the top-k sum | fails, peak 1.18x | passes, peak 0.79x |
| routing weight applied after rounding the slot output | fails, peak 1.88x | passes, 6.8e-8 of elements above the bound |
| all intermediates in float32 | passes | passes, peak 0.20x |
| top-k sum accumulated in bfloat16, one rounding per add (less accurate) | fails | fails, 5.9e-5 of elements |
| routing weights ignored | fails | fails, 99.7 % of elements |
| the two nibbles of a byte swapped | fails | fails, 98.6 % of elements |
| zero point 7 instead of 8 | fails | fails, 97.8 % of elements |
| output scaled by 1 + 2^-6 (1.6 %) / the control suite's 16-ulp scaling | fails | fails, 8 % / 70 % of elements |
| float8 rounding of the output | fails | fails, 25 % of elements |
| one token row with the routing weights ignored | fails | fails |

The baseline's own error is about 0.5 % RMS (45.7 dB SQNR), so an output scaling by 1 + 2^-8 (0.4 %) is within the
baseline's accuracy and passes with 8e-7 of the elements above the bound. The violation fraction admits about 1170
elements of the largest case, less than one token row (7168 elements), so a single wrongly computed token is still
rejected.

**Relation to the old tasks.** One source task timed the bare `fused_moe_kernel_gptq_awq` GEMM on small synthetic
shapes (float16, K-first layout, int4 and int8 weights, explicit zero points, and expert-parallel blocks with an
invalid expert written as zeros). The other ran the full `fused_experts_impl` at these production dimensions with
fixed tolerances. This task keeps the full layer at the production dimensions with the uint4b8 quantization of vLLM's
tests and a baseline-relative bound. The int8, zero-point and expert-parallel variants have no production shape in
either source (the recorded Kimi-K2.7-Code layer is int4 without zero points, all experts on the rank), so they are
not reproduced here; a scenario for one of them needs its own production origin.
