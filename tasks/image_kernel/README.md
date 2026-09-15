# Image kernel tasks: v2 contracts and runtime compatibility

All 21 packages use the same [task schema](../../docs/how-to/add-task.md).
There is no runtime dispatch on the directory name or legacy `task_type`.
These are optimization tasks with existing implementations, not empty ports.
Each has `candidate.initial_state: implemented` and a required-correctness
`baseline.kind: initial_candidate`. The framework must materialize sources, run
setup, and freeze an independent baseline **before** an agent edits the candidate.
Baseline actions use that snapshot; candidate actions use the submitted files.
Missing sources or unsupported runtimes fail explicitly without baseline fallback.

## Inventory

Counts are independently enumerated in each protected `workloads.json`.
Correctness-only cases are additional checks, never additional score points.

| Task directory | Final language | Correctness cases | Performance cases |
| --- | --- | ---: | ---: |
| [mi300x_sglang_hip_mha_batch_prefill](mi300x_sglang_hip_mha_batch_prefill/README.md) | hip | 4 | 2 |
| [mi300x_sglang_hip_pa_decode](mi300x_sglang_hip_pa_decode/README.md) | hip | 4 | 2 |
| [mi300x_sglang_hip_pa_ragged](mi300x_sglang_hip_pa_ragged/README.md) | hip | 4 | 2 |
| [mi300x_sglang_triton_fp8_gemm](mi300x_sglang_triton_fp8_gemm/README.md) | triton | 4 | 2 |
| [mi300x_sglang_triton_gemm](mi300x_sglang_triton_gemm/README.md) | triton | 4 | 2 |
| [mi355x_sglang_triton_mxfp8_grouped_gemm](mi355x_sglang_triton_mxfp8_grouped_gemm/README.md) | triton | 3 | 3 |
| [mi355x_sglang_triton_mxfp8_linear](mi355x_sglang_triton_mxfp8_linear/README.md) | triton | 6 | 6 |
| [mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3](mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3/README.md) | flydsl | 14 | 2 |
| [mi355x_vllm_ck_a8w8_blockscale_gemm](mi355x_vllm_ck_a8w8_blockscale_gemm/README.md) | hip | 5 | 3 |
| [mi355x_vllm_ck_cktile_moe_2stage](mi355x_vllm_ck_cktile_moe_2stage/README.md) | hip | 1 | 1 |
| [mi355x_vllm_ck_moe_2stage](mi355x_vllm_ck_moe_2stage/README.md) | hip | 4 | 3 |
| [mi355x_vllm_hip_dynamic_per_tensor_quant](mi355x_vllm_hip_dynamic_per_tensor_quant/README.md) | hip | 3 | 3 |
| [mi355x_vllm_hip_paged_attention_decode](mi355x_vllm_hip_paged_attention_decode/README.md) | hip | 7 | 7 |
| [mi355x_vllm_tilelang_mhc_fused_post_pre](mi355x_vllm_tilelang_mhc_fused_post_pre/README.md) | tilelang | 4 | 4 |
| [mi355x_vllm_triton_fused_moe_gemma4](mi355x_vllm_triton_fused_moe_gemma4/README.md) | triton | 3 | 3 |
| [mi355x_vllm_triton_fused_moe_gptq_awq](mi355x_vllm_triton_fused_moe_gptq_awq/README.md) | triton | 3 | 3 |
| [mi355x_vllm_triton_kda_linear_attn_kimi_k3](mi355x_vllm_triton_kda_linear_attn_kimi_k3/README.md) | triton | 5 | 5 |
| [mi355x_vllm_triton_paged_attention_2d](mi355x_vllm_triton_paged_attention_2d/README.md) | triton | 3 | 3 |
| [mi355x_vllm_triton_sparse_attn_prefill_ragged](mi355x_vllm_triton_sparse_attn_prefill_ragged/README.md) | triton | 3 | 3 |
| [mi355x_vllm_triton_unified_attention](mi355x_vllm_triton_unified_attention/README.md) | triton | 10 | 5 |
| [mi355x_vllm_triton_unified_attention_gemma4](mi355x_vllm_triton_unified_attention_gemma4/README.md) | triton | 4 | 4 |
| **Total** | | **98** | **68** |

## Task-local execution

Each task supplies `scripts/evaluate.py`, `scripts/setup_task.py`,
`scripts/task_adapter.py`, `scripts/reference_controls.py`, the original
protected `scripts/task_runner.py`, and a complete `workloads.json` manifest.
No task imports repository `src` or `agents` modules. README instructions carry
forward the former task prompt's operator, layout, dispatch and optimization
constraints. Exact editable paths are explicit, including nested Python packages,
HIP headers, and Kimi's five-file FlyDSL/config/dispatch implementation.

The entrypoint accepts `validate-task` and `{baseline,candidate}
`{compile,correctness,performance}` and emits one `ARENA_EVAL_RESULT=` JSON envelope
using `arena-eval-v1`, including on dependency, import, compile or comparison errors.
Validation reports the initial state, complete case manifest, source hashes, and
independent reference-control results. Compile executes the original JIT/operator
smoke. Correctness covers every declared case. Performance converts **fresh**
returned device measurements; it never reads a previous performance report.
Conflicting case IDs, shape/parameter changes, duplicate or missing cases,
nonfinite/nonpositive timings, and host timing methods are rejected.

The original harness comparison assertions are retained. Assertion failures can
also indicate shape, dtype, state or dispatch problems, so they are reported as
`evaluation_error`, not falsely classified as an acceptable numerical diagnostic.
Both baseline and candidate require successful numerical checks.

Python imports and direct loaders are checked against the materialized candidate
paths. HIP actions use separate fresh build directories and record successful
compiler inputs matching a declared target. A changed AITER dispatcher selecting
another implementation does not establish that the requested CK source compiled.
Compiler hooks are outside device timing; build dependencies must be materialized
before evaluation. Build-input evidence is limited coverage, not an execution trace
of every kernel or proof against arbitrary hostile Python.

Setup freezes Kimi's numerical stage implementation separately from its editable
`fused_moe.py`. The SGLang MXFP8 references use a protected, independent UE8M0
per-32-element dequantizer instead of importing the editable kernel's helper.
The remaining Torch references and all input generation remain protected.

## Preservation evidence and deliberate corrections

`tests/test_image_task_migration_v2.py` pins evidence from commit `5c9f8ef2`:
original CASES/PERF_CASES, original session JSON bytes, numerical constants, and
ASTs of all original compile/correctness/performance methods. The only change to
the original performance methods is returning their freshly produced rows.
Generated `AKA-GENERATED` performance regions are byte-for-byte unchanged.
Thus seeds, workload cases, tolerance constants, warmup/repetition settings,
graph/event method, synchronization, state reset and timed replay logic remain.

The manifest distinguishes original small checks from scored dimensions:

- Five older AITER tasks retain two small correctness cases plus two full-size
  performance cases, all four already checked by the original correctness action.
- CK GEMM and CK MoE originally clamped correctness M/token to 64. Those checks
  remain, and the adapter additionally checks every larger scored shape against
  the same reference/tolerance. CK-Tile's single M=64 case needs no extra shape.
  This adds three full-size comparisons; their GPU cost and numerical outcome
  still require formal qualification, rather than assuming the original small
  check proves the scored shape correct.
- AITER unified attention retains all five context-128 2D checks and five original
  full-context 3D scored cases. Kimi retains all 14 reachable M buckets: two scored
  cases and twelve additional correctness cases, with its worst-of-three rule.
- MHC now explicitly rejects missing output tuple members; its four individual
  `assert_close` gates are unchanged. MXFP8 linear additionally rejects wrong
  shape, dtype/device and nonfinite output before its unchanged relative-error gate.
- Three Torch references allocate on their input's device rather than a literal
  CUDA device, allowing small independent CPU known-answer tests with identical
  GPU behavior.

Representative unchanged gates include old A16W16 GEMM `atol=0.01` (BF16) / `0.005` (FP16), `rtol=0.02`,
FP8 GEMM `0.03/0.01`, old HIP attention `0.02/0.02`, CK GEMM `0.15/0.12`,
CK MoE cosine error `<0.03`, and MHC `0.08/0.08`. Kimi's per-case cosine/norm
thresholds and KDA's per-case output/state thresholds remain in the original
session specifications. No universal tolerance replaces these task-owned rules.

Independent nonzero known answers and deliberately wrong outputs are provided for
all tasks. The CPU suite directly executes the real reference/comparison controls
for 15 tasks. Six controls require the image's imported dependencies: the two older
HIP PA tasks, two CK MoE tasks, Kimi MoE, and TileLang MHC. Those six controls have
not been executed in a matching runtime in this migration. Tiny controls validate
reference semantics; they do not replace full quantized workloads or GPU validation.

The CPU session regression uses the actual migrated envelope runner and real
TaskSession to execute all seven actions, then corrupts the candidate and proves
that correctness fails while the frozen baseline still passes. Its synthetic
latency is explicitly a test fixture and provides no GPU performance evidence.

## Actual immutable-image probe: job 138977

On 2026-09-15, a single bounded job requested `amd-aicos-qos`, node
`crsuse2-m2m-213`, **one MI355X**, 8 CPU, 32 GiB, with a 15-minute cap. It initially
waited for `Resources`; without a retry or request change it ran at 05:14:52 UTC
and completed at 05:15:38 with exit `0:0` (scheduler elapsed 45 seconds).
`ROCR_VISIBLE_DEVICES=0` exposed exactly one device. Both containers executed a
nonzero Torch GPU calculation and the original image task's AITER Triton A16W16
wrapper against an independent Torch matmul reference. The allocation was released.
This was an import/JIT/API numerical smoke and source inventory, **not** a v2
validator, full workload sweep, or timed performance qualification.

The precise image references were:

- Old: `lmsysorg/sglang-rocm@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78`
- New: `lmsysorg/sglang-rocm@sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93`

| Observed component | Old image | New image |
| --- | --- | --- |
| Python package tree | `/opt/venv/lib/python3.10/site-packages` | `/opt/venv/lib/python3.12/site-packages` |
| Torch | `2.9.1+rocm7.2.0.lw.git7e1940d4` | `2.11.0+rocm10.0.0` |
| Triton | `3.6.0+git42270451` | `3.8.0+git4cff872c.rocm10.0.0` |
| SGLang | `0.5.14.dev20260705+g3ea875fef4` | `0.5.19.dev20260913+g14b647cf27` |
| FlyDSL | `0.2.2` | `0.3.2` |
| TileLang | `0.1.7.post3+cuda.gita55a8230` | same |

Image source inventory gives these **availability** results; none means the full
task passed:

| Task group | Old image | New image | Required next step |
| --- | --- | --- | --- |
| Five `mi300x_sglang_*` AITER tasks | All declared target files found | All found | Full role/case validation; only Triton A16W16 API probed here |
| Two `mi355x_sglang_triton_mxfp8_*` tasks | Both original target modules found | Both target modules absent | Retain the compatible pinned source/runtime; qualify an explicit pinned overlay or task migration before upgrading |
| Kimi AITER FlyDSL MoE | Four of five corrected nested source paths found; tuned Kimi CSV missing | All five found under `aiter/aiter/` | Validate required tuned dispatch, SiTU layout, all M buckets and numerics; source presence alone is insufficient |
| Other 13 `mi355x_vllm_*` tasks | Declared installed source directory absent | Declared installed source directory absent | Use and pin their actual session-compatible vLLM image or build a qualified overlay with the exact dependencies/layout |

The missing vLLM paths are real filesystem observations, not just missing package
metadata. They include the declared `/usr/local/lib/python3.12/dist-packages/`
`vllm`, `aiter`, and `aiter_meta` roots. The complete AITER repository under
`/sgl-workspace/aiter` supplies some similarly named files, but does not establish
that the vLLM dispatch/API/package layout exists. In particular, an AITER
`fused_moe.py` is not a replacement for vLLM's identically named file. The probe's
heuristic alternative-file inventory must not be treated as a source fallback.

Original session JSON specifies the intended vLLM/SGLang base images and, where
needed, custom builds. Those historical tags are provenance, not newly qualified
immutable defaults. No image default or declared source path is silently changed
for the 13 unavailable tasks. `aiter_meta` is explicitly declared as a second source
for the package-layout unified-attention task so headers resolve within the copied
workspace, rather than via an external image symlink.

The new SGLang image is therefore not a suite-wide replacement for the old image.
Keep the existing default until suitable per-task runtimes and complete v2 checks
are qualified. Missing source packages require source/runtime qualification;
loosening numerical gates or accepting installed fallback would not solve them.

## Integration handoff and evidence locations

The parent migration owns shared runtime/loader/evaluator integration. Its perf
helper discovery must materialize the original `scripts/task_runner.py` even
though `evaluation.runner` now points to `scripts/evaluate.py`, which reaches the
harness by a task-local import. The committed generated stubs intentionally fail
if that materialization is missing. This migration does not hand-edit them.

Run the focused CPU suite with `python -m pytest -q
tests/test_image_task_migration_v2.py` in the parent-provided test environment.
Run formal initial baseline validation and candidate compile/correctness/performance
on compatible hardware only after the parent wires the shared v2 pipeline. No
full validator or optimization campaign was launched by this worker.

Uncommitted reproducibility artifacts are retained under
`logs/image-migration-v2/`: the Slurm request, final job state, masked GPU output,
Docker image inspections, old/new JSON inventories, copied inspected source files,
original harness evidence, and CPU test outputs. Job-specific files are under
`logs/image-migration-v2/138977/`. Source SHA256 hashes in action envelopes identify
the actual materialized implementation, since image tags alone are insufficient.
