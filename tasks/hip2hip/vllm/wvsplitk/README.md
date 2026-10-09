# Small-batch split-K GEMM

Optimize `wvSplitK` in `src/rocm/skinny_gemms.cu`. Its fixed interface computes
`activation @ weight.T + optional_bias`, with activation `[M,K]`, weight `[N,K]`,
and output `[M,N]`. All tensors are contiguous; M is 1 through 4. The protected
binding passes the live device CU count. Other operators present in the bundled
translation unit are dependencies, not separately scored objectives.

The 18-case manifest retains all 14 captured cases from PR #75, including optional
bias and large vocabulary projections. It also includes M=2,3,4 at N=256,K=7168
and an FP16 biased M=4 case. The old default c32 and c64 cases were the same M=4
problem; that distinct shape appears once. Original c2's M=2 shape is retained.
The revised workload has a new score identity; old aggregate timings are not
comparable. Matrices are dense random inputs, not zero or simplified fixtures.

The independent reference uses FP32 matmul and bias before casting to the declared
FP16/BF16 output. The original `atol=0.05, rtol=0.05` elementwise gate is retained.
Each call's activation changes during measured replay; weights and bias are fixed
within a case, as in decode serving. Their contents, storage, and layout must stay
unchanged. Output allocation performed by the operator is inside the measured call.

The self-contained HIP source and headers originate from vLLM's extracted operator
in PR #75. Existing source license notices are retained. Build caches are bound to
all source bytes and GPU architecture under `build/`; hipify runs on a copied source
tree there and must not modify bundled sources. All native launches must use the
current PyTorch stream so they remain captured and measured.

## Evaluation contract

The schema-v2 runner implements all seven arena-eval-v1 actions. Baseline actions
use the framework's frozen initial candidate in a separate workspace; candidate
actions use the same relative interface in the working workspace. Every manifest
case participates in both correctness and performance. Compilation builds and
launches every specialization, rather than accepting source text alone.

The protected task API constructs inputs, computes an independent reference before
calling the implementation, and rejects wrong shape, dtype, device, non-finite
values, and modified read-only inputs. `validate-task` also tests deliberately
wrong, NaN, and wrong-dtype outputs against the comparator.

Performance uses the materialized canonical GPU graph/Event helper: 10 warmups,
100 device samples, one logical invocation per sample. Preparation copies fresh
call inputs outside device timing. Three randomly seeded draws rotate at the same
addresses; eight privately selected measured outputs and the final actual replay
are checked against references computed before any candidate execution. Four
previously unused draws follow the same device timing path; correctness must pass
and their minimum time must be at most 1.5 times the reported mean. A separate
poisoned-output replay checks complete writes, after actual samples have already
been verified. Seeds, checked sample indices, replay checks, timing method, and
unseen timings are retained in result metadata. This is a practical integrity
check, not a sandbox for adversarial native code or a proof against all caching.

Only declared candidate implementation code is editable. The workload, references,
launch ABI, build policy and runner are protected. Do not read or import protected
reference code, cache answers, change task files, download dependencies, or access
other workspaces. Use only the bundled implementation dependencies and the runtime
compiler. Keep allocations and computation within the measured invocation except
for the input/output/scratch buffers allocated by the protected API.

Only gfx950 is declared. Other architectures require separate qualification.
Create an untracked `config_validator.yaml` at the repository root using the
[validator guide](../../../../docs/how-to/task-validator.md#run-the-validator).
Select `tasks: [hip2hip/vllm/wvsplitk]` and `target_gpu_model: MI355X`, then run:

```bash
make docker-run CONFIG=config_validator.yaml
```

A successful direct runner command is diagnostic evidence. The acceptance gate is
a fresh framework-finalized `validation_report.yaml` with `overall_status: PASS`.
