# triton_bmm

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton batched matrix multiplication kernel `bmm_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel implements batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N) with support
for large tensors and contiguous memory hints.

Key optimization opportunities:
- Tile size tuning (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
- Memory access pattern optimization
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `bmm_triton`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

Correctness and performance checks retain pristine input snapshots and require
that the candidate preserves caller-owned inputs. The runner checks the actual
timed BMM output, then changes the captured graph's input tensors, poisons the
output with NaNs and checks the same graph replay against a reference computed
before replay. Output shape, dtype, device and finite values are required.
The original FP32 BMM reference, FP16 output and `atol=1e-2`, `rtol=1e-2` stay
unchanged. All added reference work and replay checks run outside device timing;
the original five cases, seeds, warmups, samples and timed operator are preserved.


### Editable device implementation boundary

The editable device functions are `bmm_kernel`.
Their bodies and additional implementation helpers may be optimized. Module
imports, initialization, host dispatch and pointer preparation remain protected,
along with the existing numerical checks and benchmark protocol. The structural
guard enforces this boundary; it does not isolate arbitrary Python execution.

The protected wrapper also fixes the launch configuration: the grid and the tile
sizes, `num_warps` and `num_stages` it passes at launch. Optimization notes above
that mention tuning those values refer to the original whole-file contract; under
this boundary, optimize the kernel bodies and implementation helpers instead.


### Correctness-only domain controls

The protected `bmm_triton` wrapper defines the public domain. It accepts two
3-D tensors with matching batch and inner dimensions and one shared dtype:
float16, bfloat16 or float32, the only keys of its launch-configuration table.
It forwards all six operand strides to the kernel without copying, so
transposed, offset, padded, element-strided and batch-broadcast (stride-0)
views are valid inputs. It always allocates a new contiguous `(B, M, N)` output
of the input dtype on `a.device`; both operands must be on that GPU. Zero-sized
dimensions and the int64 path for tensors over 2**31 elements are not exercised.

Eleven unscored `control-domain-*` rows (`case_index` 20000-20010, checks
`[correctness]`) come from the protected `_domain_controls.py` table; the
adapter verifies that table and these rows against `workloads.json`. They cover
bfloat16 and float32 as well as float16, and B, M, N and K values that are not
multiples of the wrapper tiles (128/128/32 for 16-bit dtypes, 64/64/32 for
float32), including `M = K = 1` and other small odd shapes. Operand layouts
include transposed A, B or both, a batch-inner `(M, B, K).transpose(0, 1)` view,
offset views of padded storage, element stride 2, batch stride 2 and a
batch-broadcast B. Padding and stride gaps hold a sentinel value, so a read
outside a view changes the result.

Each control draws seeded N(0, 1) inputs on the CPU. It computes an FP64 CPU
`torch.bmm` reference from pristine clones before calling the candidate. The
input views and their complete backing storage must be unchanged. The output
must have the exact shape, dtype and device, contiguous `(M*N, N, 1)` strides
and finite values, and match within `atol=rtol=1e-2` for every dtype. For
float16 this is the original gate. One bfloat16 ulp is at most 0.78% of a
value. Float32 keeps the bound that protected `_arena_checks.py` already applies
to every call, including these controls; layout, tail and dtype defects give
much larger errors at these magnitudes. The five scored float16 cases, their
seeds, tolerances, warmups, samples and timing are unchanged, and no control
is timed.
