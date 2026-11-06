# GDN packed recurrent decode, bfloat16, 16 query/key heads, 32 value heads, K = V = 128

One decode step of the gated delta rule (Gated DeltaNet) per sequence, reading the packed `q | k | v`
projection row and updating the float32 recurrent state of the sequence's pool slot in place
(`fused_recurrent_gated_delta_rule_packed_decode`). For each of the 32 value heads (query/key head
`hv // 2`):

```text
q, k    = l2norm(q), l2norm(k);  q *= scale
g       = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
S       = exp(g) S                        # S: [V, K] float32 state of the slot
S      += ((v - S k) * beta) k^T
out     = S q                             # bfloat16
```

`ssm_state_indices` gives each sequence a distinct slot of the pool (`batch + 2` slots, so some stay
untouched); every 16th sequence is padding with slot -1, whose output is zero and whose state is not
touched, as in a CUDA-graph-padded decode batch. The pool is a mutable input, restored before every
call and compared afterwards.

- **Origin.** The packed decode kernel of SGLang's FLA package
  (`python/sglang/kernels/ops/attention/fla/fused_recurrent.py`, adapted from flash-linear-attention),
  Qwen3-Next dimensions and float32 state. It merges the earlier SGLang task (which drew the state in
  bfloat16 and covered 8/16-head splits) and the vLLM fused-recurrent variant; the other head counts
  and the vLLM kernel (different interface) are not part of this scenario.
- **Baseline.** The earlier task's Triton kernel (one program per 32-wide value tile and value head,
  1 warp, 3 stages), wrapped in `run()` with in-kernel L2 normalization on, as the earlier harness ran
  it. The kernel rounds `sigmoid(b)` to the dtype of `b` (bfloat16), which the float64 reference does
  not; this belongs to the baseline's own error.
- **Inputs.** `a`, `b`, `dt_bias` and the packed projection are standard normal, `A_log` uniform in
  [0, 2.7] (the model initializes it to the log of a value in [1, 16]), the state normal with standard
  deviation 0.1. The key normalization keeps the recurrence contractive.
- **Reference.** Float64, vectorized over groups of 32 sequences; checked on CPU against a per-sequence
  emulation of the kernel (about 53 dB SQNR on `out`, 59 dB on the state).
- **Comparison (overrides of the default bound).** `ssm_state` uses `baseline_relative` with `k: 8` and
  `max_violation_fraction: 0.001`. `out` keeps `k = 2` and allows `max_violation_fraction: 0.0001`.
  The numbers below come from CPU float32 emulations measured against the float64 reference. The
  baseline was emulated with its tree reductions, fused multiply-adds, bfloat16 `beta` and 1-ulp
  libm errors.
  - *State.* Most of the baseline's state error comes from its bfloat16 rounding of `beta`
    (relative error up to 2^-9), so in most rows the bound is wide. Two kinds of rows have no beta
    error:
    - rows with `v ≈ S k`;
    - heads whose `b` has a sigmoid within a relative 1e-6 of a bfloat16 number. `b` is bfloat16, so
      its draws repeat, and about one sequence in twelve has such a head.

    In those rows the baseline's error is plain float32 rounding, about one unit roundoff of `|y|`.
    The default bound allows twice that plus one rounding. It rejects independent float32
    implementations whose per-head decay `exp(g)`, sigmoid or summation order rounds differently.
    The MI355X audit's float32 evaluation of the reference failed the default bound in 10 of 12
    cases, on up to 6.9e-5 of the elements and at up to 3.5x the bound.

    The emulations used 126 case draws from b1 to b100, plus b128 and b256. Correct variants needed
    a k of up to 19 on single elements with 1-ulp math, and over 100 with 4- to 8-ulp fast math in
    the per-head scalars. At `k = 8`, the largest share of elements out of bound in any case was:
    - 1.1e-5 for PyTorch float32, other libm roundings, float32 `beta` with `log1p`, sequential
      unfused sums and the rank-1 form `out = exp(g) (S q) + delta (k . q)`;
    - 5.6e-4 for 4-ulp fast math, with or without 4-ulp errors in every element of q and k;
    - 2.0e-3 for an 8-ulp stress model, in one b1 draw of 60.

    A fraction of `0.001` accepts all of these except that single stress draw. At b800 it allows
    420k elements, still less than one sequence's 524k-element state.

    The same bound still rejects all of the following in every case from 16 sequences up; the
    figure is the smallest share of elements out of bound:
    - bfloat16-rounded state: 6.6e-2 (2e-2 at b1);
    - float16-rounded state: 1.0e-2;
    - bfloat16 decay: 4.4e-2;
    - bfloat16 normalized keys: 3.1e-2;
    - bfloat16 or TF32 operands in `S k`: 2.2e-3 and 1.0e-3;
    - a prediction from the undecayed state: 0.64;
    - padded sequences updating a slot: 8.9e-3;
    - every QA control: 7.8e-2 and up. Untouched pool slots must stay unchanged for any `k`.

    An SQNR bound cannot replace this one. The baseline's state is at about 59 dB and a
    bfloat16-rounded state at 55 to 62 dB.
  - *Output.* `out` is a near-total cancellation in some heads: those where `exp(g)` is tiny
    (`g < -9`, about 15% of the value heads with these inputs) and `q` is almost orthogonal to `k`
    (`|cos| < 2.5e-5`, about 2e-4 of the key heads). There it is about 1e-8 while its terms are of
    order 1e-4. Float32 rounding then leaves every implementation a relative error of about
    `u / |cos|` (u the float32 unit roundoff), and the candidate's error can exceed twice the
    baseline's in a few elements.

    A Monte Carlo sampled 8000 such value heads from the task's distributions. With the default
    bound, 15% (PyTorch float32) to 53% (sequential sums) of the evaluations of a correct float32
    candidate would fail somewhere among both draws of all cases, and 71% with 4-ulp errors in every
    element of q and k. Each affected head has a median of 1 to 6 and at most 102 elements out of
    bound. Allowing 1e-4 of the elements lowers the failure rate below 1% (5% for the
    every-element 4-ulp model). That is 0 elements at b1, 15 at b37 and 327 at b800; every other
    element must still meet the default bound. Every QA control is still rejected at b1, where no
    element may exceed it.
- **Workloads.** 8 to 800 sequences (100x span). Each sequence moves 2 MB of state in and out, so 8
  sequences are about 32 MB, or 11 us with the 4 us floor: the smallest case is tight, which the GPU
  audit should confirm. Batches of 1, 3, 37 and 100 are correctness-only. The largest case holds 802
  float32 slots (1.7 GB); its float64 reference and comparison temporaries need tens of GB.
