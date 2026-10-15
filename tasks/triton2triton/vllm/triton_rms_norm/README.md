# triton_rms_norm

The starting candidate is implemented Triton. Improve the declared kernel symbol in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton RMS normalization kernel `_rms_norm_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel computes RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight along the last
dimension, using float32 accumulation for numerical stability.

Key optimization opportunities:
- Reduction strategy and work organization inside the kernel
- Fused computation (reduce passes over data)
- Memory access pattern optimization
- Vectorized loads/stores

Constraints:
- Must maintain the same function signature for `rms_norm`
- Output must match reference within atol=1e-2, rtol=1e-2 for the declared
  float16 and bfloat16 cases, preserving input shape, dtype and device
- Input and weight buffers are read-only; write results only to the output buffer


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares eight scored cases and four additional correctness cases;
the adapter verifies both tables against the protected harness. All original five
cases and their input seeds, comparisons, tolerances, warmups, sample counts and
graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

The finite scored workload uses contiguous float16 inputs, one-dimensional
float16 weights and `eps=1e-6`. Its `(rows, hidden_size)` cases are:

| IDs | Shapes | Purpose |
| --- | --- | --- |
| `perf1`–`perf5` | `(32,128)`, `(64,512)`, `(128,1024)`, `(256,2048)`, `(512,4096)` | Original workload, unchanged |
| `perf6` | `(1,4096)` | One-token decode |
| `perf7` | `(32,576)` | Masked tail within the first 1024-element block |
| `perf8` | `(32,1536)` | A full block followed by a masked block tail |

Correctness draws use seed `42 + case_index`; each performance draw uses seed 0.
All eight cases retain 10 warmups and 100 samples. Baseline and candidate execute
the same cases and timing policy. The added cases extend the score, so a score
over all eight cases is a new workload and is not directly comparable to the old
five-case aggregate. The original five per-case timings remain comparable.

The canonical adapter also runs these correctness-only cases, with the same
float32 reference accumulation and `atol=rtol=1e-2`. They do not affect timing:

| ID | Input shape | Dtype | Epsilon | Input scale | Seed |
| --- | --- | --- | --- | --- | --- |
| `control_bf16` | `(8,1536)` | bfloat16 | `1e-6` | 1 | 1001 |
| `control_rank3` | `(2,3,576)` | float16 | `1e-6` | 1 | 1002 |
| `control_rank3_bf16_eps` | `(2,3,1536)` | bfloat16 | `1e-5` | `1e-3` | 1003 |
| `control_eps` | `(4,1536)` | float16 | `1e-3` | `1e-3` | 1004 |

Each input and weight is a standard normal draw; the input alone is multiplied
by its declared scale. The small-input variants make epsilon materially affect
normalization. The rank-three cases exercise the protected wrapper's flattening
and output reshape, including BF16 dispatch. These are declared examples, not a
claim of validation for arbitrary shapes, layouts, ranks or dtypes.

The protected `_arena_checks.py` snapshots input and weight buffers and computes
the reference from those snapshots before candidate execution. It rejects changes
to either buffer, checks output shape, dtype, device and finite values, and restores
the input buffers on success or failure. During performance it checks the actual
captured output and read-only inputs, changes the captured inputs, computes a new
reference before replay, poisons the output, and checks the same graph's replay.
Snapshot, reference and validation work runs outside device timing and uses the
same policy for baseline and candidate. The measured callback and all timing
options remain unchanged. Observable graph replay is required; an
unobservable event fallback fails instead of claiming replay validation.

Only `_rms_norm_kernel` and permitted new implementation helpers are editable.
The existing `rms_norm` wrapper, imports and module-level execution remain
protected. The wrapper fixes one program per row, its `BLOCK_SIZE=1024` launch
argument, input contiguity/reshape operations and output allocation. Changing
those host or launch choices is outside this kernel task's optimization scope;
the kernel may reorganize its internal reduction and memory operations while
preserving the declared interface and full numerical contract.
