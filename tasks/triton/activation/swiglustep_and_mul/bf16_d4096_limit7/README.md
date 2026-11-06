# SwiGLU-step and mul, bfloat16, d = 4096

`out = min(silu(gate), limit) * clamp(up, -limit, limit)` with the gate half first and the up half
second in each `[n_tokens, 2*d]` row, `limit = 7`. Differs from a clamped-SiLU (`clamp_silu_and_mul`)
in that the upper clamp comes after the SiLU and the gate has no lower clamp. The reference is float64.

**Origin.** vLLM `SwigluStepAndMul`, used by Step-3.x models (`models/step3p5.py`) for the shared
expert and MoE activation when a layer's `swiglu_limits` entry is set. The limit is read per layer
from the checkpoint config; 7.0 is vLLM's default. The width d = 4096 is a representative size, not
confirmed against a specific checkpoint (this checkout does not state the model's intermediate size).

**Baseline.** vLLM's Triton kernel at the recorded commit, unchanged (one row per program on axis 0,
1024-wide column blocks on axis 1).

**Workloads.** Prefill token counts; the old task's five fp16 shapes (up to d = 2048) were all below
the timing floor. Inputs use std 3 so roughly 2% of gate and up values exceed the limit and the clamps
matter. One correctness case uses `limit = 0.5` so a hard-coded limit fails. At 24 KiB per row,
about 1100 rows reach 10 us; smaller batches are correctness-only.

**Comparison.** Default `baseline_relative`.

**Memory.** The largest case is 4.3 GB of input; the reference works in 16K-row float64 chunks and
holds an 8.6 GB float64 output.
