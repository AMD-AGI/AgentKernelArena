# Lightning attention prefill, bf16, 64 heads, head size 128, block 256

Prefill of one sequence through MiniMax-Text-01's lightning (linear) attention, the block-parallel
formulation of vLLM's `lightning_attention`. Tokens are split into blocks of 256 (the last may be
partial). Per head with decay slope `r`:

```text
inside a block   out[t] += sum_{j <= t} exp(-r (t - j)) <q[t], k[j]> v[j]
across blocks    out[t] += exp(-r p) q[t] . S_b          (p = position of t in its block b)
state            S_0 = kv_history,  S_{b+1} = exp(-r m_b) S_b + sum_c exp(-r (m_b - 1 - c)) k_c v_c^T
```

`m_b` is the block's token count; the final state replaces `kv_history`. `q`, `k`, `v` and `out` are
bfloat16 `[64, n_tokens, 128]`; `kv_history` is float32 `[64, 128, 128]`, the mutable state, checked
after the call. The reference reproduces the production kernels' decay convention, in which the
history and earlier blocks decay one step less than a same-block key at the same distance (see the
docstring of `reference.py`); a literal element-by-element emulation of the four Triton kernels (CPU,
float64) agrees with it to 1e-13 on 1, 65, 300, 513 and 700 tokens.

- **Origin.** The four kernels `_fwd_diag_kernel`, `_fwd_kv_parallel`, `_fwd_kv_reduce` and
  `_fwd_none_diag_kernel` of `vllm/model_executor/layers/lightning_attn.py` and the forward of
  `_attention`, which launches them. They replace four earlier tasks, one per kernel; a single task
  now covers the whole operator. MiniMax-Text-01 has 64 heads of size 128 and uses block 256;
  `lightning_attention` handles head size 128 in a single feature split, which is the only path
  modeled. Production computes q, k, v in float32 after a SiLU in current vLLM; this task keeps the
  bfloat16 activations of the earlier tasks.
- **Inputs.** `q`, `k`, `v ~ N(0, 1)`, the state history `~ N(0, 0.1)`; `inputs.py` builds the slopes of
  `MiniMaxText01LinearAttention._build_slope_tensor` for 64 heads (0.0029 to 0.68) scaled by the
  layer factor of layer 20 of 80.
- **Baseline.** The four vLLM kernels with their launch code (diag block 256 / sub-block 32,
  kv sub-block 64), batch of one. Two changes, neither of them to the algorithm:
  - vLLM builds the per-head table of key decays with `torch.exp` on the host before
    `_fwd_kv_parallel`; because PyTorch arithmetic is not allowed in `run()`, the kernel now
    computes `exp(-r (255 - position))` itself.
  - `_fwd_kv_parallel` widens its key and value loads to float32 (`.to(tl.float32)`, exact), as
    `_fwd_diag_kernel` already does. vLLM runs the kernel on float32 activations, so both operands of
    its `tl.dot(k_trans * k_decay, v)` are float32 there; with this task's bfloat16 keys and values the
    decayed keys are float32 and the values bfloat16, and Triton 3.8.0 refuses to compile the dot
    ("Both operands must be same dtype. Got fp32 and bf16"; every case failed the MI355X audit this
    way). The GEMM is now the float32 GEMM of the production path.

  Intermediate block states are allocated per call (`torch.empty`, 4 MiB per 256 tokens). The
  diagonal kernel stores the output rounded to bfloat16 and the last kernel adds to that rounded
  value, as in vLLM. Run on CPU through the Triton 3.8.0 interpreter (float32 dots, bfloat16 stores
  rounded to nearest even as on the device), the baseline reaches 52.6-55.4 dB SQNR on `out` (a single
  bfloat16 rounding of the reference gives 55.4-56.0 dB) and 136-152 dB on `kv_history`, on 1, 65,
  257, 1000 and 1024 tokens (64 heads) and 4096 tokens (8 heads). All four kernels also compile for
  gfx950 with upstream Triton 3.8.0 in the specializations of every case, including the 131072-token
  one, whose 2 GiB tensors take 64-bit global addressing instead of buffer loads.
- **Comparison.** `out` uses the default `baseline_relative` bound: its error is bfloat16 rounding,
  and a float32 evaluation of the reference stays within 0.33x of it (CPU emulation, 1000 tokens).
  `kv_history` uses `sqnr` with `min_db: 127` instead. It is float32 and its error is float32
  accumulation (256-token block GEMMs, then the scan over blocks), so the default bound, twice the
  baseline's own error or its per-row RMS, compares two independent rounding errors element by
  element. In a CPU emulation (1000 tokens, 64 heads; baseline through the interpreter) the float32
  evaluation of the reference, which is what the QA calibration probe runs, fails it on 1.3% of
  the state (peak 4.1x) and still on 3.8e-6 at `k = 8`, the largest multiple allowed; a per-token
  sum with exact decays inside each block fails it on 2.3% (peak 4.8x) and on 3.1e-5 at `k = 8`.
  No per-element multiple accepts them without a violation fraction. SQNR of the final state against
  the float64 reference (CPU, 64 heads, 1 to 16384 tokens; it is flat from about 1000 tokens on,
  since the slowest head's decay has an e-folding length of about 340 tokens):

  | Implementation | SQNR of `kv_history` |
  | --- | --- |
  | baseline kernels (Triton interpreter; up to 4096 tokens) | 136.1-156.3 dB |
  | baseline algorithm in float32, exact `exp` / fused scan update | 136.9-161.7 dB |
  | baseline algorithm, `exp` off by up to 1 / 2 / 4 ulp | 135.6-140.6 / 133.9-138.2 / 130.0-132.4 dB |
  | float32 evaluation of the reference (the probe) | 135.1-157.7 dB |
  | one 256-token GEMM per block; chunks of 64 or 1024 tokens | 135.0-157.7 dB |
  | per-token sum with exact decays inside each block (up to 4096 tokens) | 133.5-157.7 dB |
  | control: output scaled by 1 + 16 u (`shifted_16_ulp`) | 120.4 dB |
  | control: state stored through bfloat16 / fp8 e4m3 | 55.6 / 31.5-31.6 dB |

  The compiled kernels evaluate `tl.exp` as `exp2(x log2 e)` on `v_exp_f32` and accumulate with
  float32 MFMA, so the 1-2 ulp rows are the realistic model of the device baseline. 127 dB, a
  relative RMS error of 4.5e-7 (about 7.5 float32 unit roundoffs), sits 6.5 dB below the least
  accurate correct design above with exact decays, 3 dB below a 4-ulp `exp`, and 6.6 dB above the
  16-ulp shift control. A single state element wrong by the RMS magnitude gives 60 dB (1 M
  elements), so isolated wrong elements stay rejected. A per-token recurrence that multiplies the
  state by a rounded `exp(-r)` at every token compounds that rounding over the head's memory and
  reaches only 111.9-125.0 dB (65 to 4096 tokens), below the 16-ulp shift, and is rejected; decays must
  be computed from the distance, as the reference and the baseline do.
- **Workloads.** 1024 to 131072 tokens (128x span); `out` plus `q`, `k`, `v` are 64 KB per token,
  read several times by the four kernels, so even 1024 tokens (about 70 MB) should take the baseline
  over 10 us. Production prefill chunks are typically 1-16K tokens; 65536 and 131072 stand for long
  prompts and are needed for the span. Axis minimum 1, 65, 257 (one block plus one token) and 1000
  are correctness-only. The largest case holds about 6.4 GB of inputs and an 8.6 GB float64
  reference output; the CPU pre-check does not smoke it, but the reference was verified on small
  cases as described above.
- **Timing risk.** `kv_history` is restored by the evaluator before every call; the reduce kernel is
  a serial scan over blocks per head (64 programs), which dominates short sequences.
