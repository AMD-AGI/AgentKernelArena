# GPT-OSS SwiGLU and mul on the block-interleaved layout, bfloat16, d = 2880 (FlyDSL baseline)

`out = g * sigmoid(1.702 * g) * (u + 1)` with `g = min(gate, 7)` and `u = clamp(up, -7, 7)`, computed in float32 and
rounded to bfloat16. The input row of `2 * d` columns is **block-interleaved**: for every block of 16 output
columns it holds 16 gate values followed by the 16 up values (`x.view(n_rows, d / 16, 2, 16)`, gate at index 0 of
the pair axis). This is the layout that AITER's CK-Tile split-K MoE stage 1 produces for MXFP4 weights; the
operator is the same as the Triton task `tasks/triton/activation/swigluoai_and_mul/bf16_d2880`, whose input is
`[gate | up]`. The float64 reference de-interleaves and applies the formula above.

## Name

The batch listing asked for `silu_and_mul` from `swiglu_and_mul.py`. The kernel in that file is not SiLU-and-mul:
it computes the clamped GPT-OSS SwiGLU (alpha 1.702, limit 7.0, the `+ 1` on the up half) and AITER's separate
SiLU path for the interleaved layout is `silu_and_mul_fq.py` (a quantizing kernel that is not part of this task).
The task is therefore named after what it computes, `swigluoai_and_mul`, as its Triton sibling is.

## Scenario and config citations

- gpt-oss-120b `config.json` (Hugging Face): `hidden_size` 2880, `intermediate_size` 2880 (so `d = 2880`),
  `num_local_experts` 128, `num_experts_per_tok` 4, `swiglu_limit` 7.0. Corroborated in the pinned AITER
  checkout, `op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json`, entry "GPT-OSS 120B":
  `moe_op_gemm_a8w4` has `E` 128, `Dim1` 2880, `Dim2` 5760 (= 2 x 2880), `TopK` 4.
- The constant `alpha = 1.702` is the default of `gemm1_alpha` in SGLang's `python/sglang/srt/models/gpt_oss.py`
  (`getattr(config, "hidden_act_alpha", 1.702)`, pinned SGLang commit); that file reads the clamp limit from
  `config.swiglu_limit`, which is 7.0 in the config above and the constant in AITER's `swiglu_and_mul.py`.
- AITER calls the kernel from `aiter/fused_moe.py` (CK-Tile stage 1, `ActivationType.Swiglu`, interleaved layout) on
  `tmp_out[:token_num * topk]` viewed as `(-1, 2 * inter_dim)`, so the row count is tokens x 4.

## Baseline

`baseline/swiglu_and_mul.py` is AITER's FlyDSL kernel `build_swiglu_and_mul_module` (MIT, `baseline/LICENSE`),
unchanged except for the relative import of `buffer_ops.py`. `baseline/buffer_ops.py` is AITER's vendored copy of
FlyDSL's `kernels/common/buffer_ops.py` (Apache-2.0, `baseline/LICENSE-FlyDSL`, header kept verbatim). `kernel.py`
holds the cached builder, the compile-once helper and the launch of `flydsl_swiglu_and_mul_interleaved` from
`aiter/ops/flydsl/moe_kernels.py`. One 256-thread program per row; each thread handles two adjacent output columns
(one 32-bit word) per iteration with scalar 32-bit buffer loads of the gate and up words.

Origin notes (the `git log --follow` of `swiglu_and_mul.py` starts with the "kimi a16wi4 moe support" change of
AITER; the file has no "adapted from"/"based on" text, only AITER's SPDX/copyright header, and `buffer_ops.py`
states that it is a copy of FlyDSL's `kernels/common/buffer_ops.py`). No other third-party origin was found.

## FlyDSL 0.3.2 adaptation

FlyDSL 0.3.2 adaptation: in buffer_ops.buffer_load and buffer_store the cache-policy operand is passed as the keyword-only aux integer attribute of rocdl.RawPtrBufferLoadOp / RawPtrBufferStoreOp (the runtime's op constructors take no positional aux value); no change to the kernel algorithm. The rest of the FlyDSL API used by the kernel (`flyc.kernel/jit`, `fx.Tensor`, `fx.block_idx`, `arith.minimumf/maximumf/trunc_f/bitcast/extui`, `rocdl.exp2/rcp`, `CompilationContext`) was checked by name against `python/flydsl` of FlyDSL v0.3.2.

## Workloads

Rows are tokens x top-4: 2048 to 262144 rows (512 to 65536 tokens, 128x span) are performance cases. Each row moves
11520 B in and 5760 B out (17280 B), so 2048 rows are 35 MB: 7.9 us at 4.5 TB/s plus the 4 us fixed cost, about
12 us; 262144 rows are 4.5 GB, about 1.0 ms. Smaller row counts are correctness-only: 1 (axis minimum), 3, 17 (not a
multiple of 16), 48 and 1000 (not a power of two). The kernel's scalar word loads make it slower than the
bandwidth estimate, so the 10 us floor should hold with margin.

Inputs are standard normal with standard deviation 3, so about 1% of the gate and 2% of the up values exceed the
limit and both clamps matter, and the gate's sigmoid saturates over part of the range.

## Comparison

Default `baseline_relative`. The largest case holds 3.0 GB of input; the reference works in 16384-row chunks and
holds a 6 GB float64 output.

## Byte offsets

The kernel forms 32-bit byte offsets (`offset * 4`) that exceed 2^31 for the input above about 186000 rows (the
262144-row case reaches 3.0 GB, below the 4 GB buffer descriptor limit). The hardware offset is unsigned, so these
offsets are valid.
