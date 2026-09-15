# mi355x-ci-ck-moe-2stage-20260720

Self-contained image_kernel harness generated from Hyperloom 2026-07-20 MI355X sessions. See session_cases.json for exact provenance, shapes and dtypes.

## Effective task instructions

Optimize ck_moe_2stage on MI355X/gfx950. The harness cases are parsed from Hyperloom 2026-07-20 sessions and stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance.

## Arena v2 contract

The candidate is the existing implementation in the declared image sources.
Its required final language and exact task-relative editable files are in
`config.yaml`; directory names do not select execution behavior. The framework
freezes this initial implementation into a separate baseline workspace. Both
roles run the same protected harness in their own workspace; an absent candidate
or missing image source is an error, never permission to use the installed copy.

Setup runs `python3 scripts/setup_task.py` after declared image materialization
and before baseline capture. It validates source paths and required build assets.
Do not edit `scripts/`, workload files or references. Additional source files
outside `candidate.editable` are dependencies, not editable implementation.
Preserve the original numerical gates, seeds, layouts, dispatch, state handling
and CUDA graph/event timing. `workloads.json` enumerates the complete manifest
independently of reported timings; `session_cases.json`, when present, retains
its original session provenance. Cases marked correctness-only are not scored.

Use the agent-neutral commands:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Baseline commands run in the framework's frozen workspace. Each command emits
one `ARENA_EVAL_RESULT=` envelope. A failed dependency, dispatch or output contract
is a failure, not an accepted baseline numerical diagnostic. The original
`task_runner.py` remains the protected operator implementation of these checks;
its generated performance region must be materialized by Arena. Optional
profiling does not supply final evaluation evidence. Agent CLI adaptation belongs
to the agent integration; use the declared v2 runner for task evaluation, with
the task's full numerical and workload checks.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.

The original reduced-size correctness checks are retained; scored-size checks use the same tolerances in addition, so the manifest never declares untested performance shapes as covered.

HIP evaluation uses a fresh task-local JIT directory per action. The runner
requires a successful compilation whose inputs include a declared candidate
translation unit or template header. It records the covered files and rejects
unrelated/precompiled dispatch. This is build-source evidence, not exhaustive
proof that every launched GPU instruction belongs to every editable file.
No compiler-triggered repository cloning or checkout resets are permitted.

The qualified SGLang runtime stores AITER as a complete source repository at
`/sgl-workspace/aiter`, not as an installed `aiter_meta` wheel directory.
`workspace.sources` explicitly copies that repository to the task's metadata
root; the unified-attention task separately copies its `aiter/` Python package.
Editable task-relative paths and operator semantics remain unchanged. This fixes
source availability only; dispatch, compilation and numerical compatibility
still require full GPU validation on the selected immutable runtime.

## Materialized AITER package and build helpers

### FP8 block-quantization input contract

For the MiniMax `per_1x128` case, the CK interface uses 1×128 activation groups
and independent 128×128 weight blocks within each expert. Weight scales have
shape `(experts, rows/128, columns/128)`. The pinned runtime's
`get_torch_quant(per_1x128)` accepts a two-dimensional activation matrix; it is
not a quantizer for the three-dimensional expert weight tensor. Job 139599
compiled the declared CK implementation but failed when the old preparation
passed that tensor directly to the activation quantizer.

Preparation now quantizes each weight block using the provided per-row Torch
quantizer, restores the logical weight axes, and retains one scale per block.
Reference activation quantization flattens only token/top-k axes before calling
the runtime's two-dimensional API, then restores those axes and their scales.
The MXFP4 and per-tensor paths, operator cases, seeds, numerical gates and timing
calls are unchanged. CPU controls verify block isolation, axis restoration and
input preservation. This correction still requires fresh full GPU validation.

The declared runtime provides both `aiter/` (Python dispatch and JIT utilities)
and `aiter_meta/` (C++ sources and bundled compiler dependencies). They are
siblings inside each role's workspace. Code generation resolves helpers such as
`aiter/jit/utils/chip_info.py` relative to that layout. Both copies come from
the same selected image; the candidate remains limited to its declared HIP
sources, while Python dispatch, test inputs and references stay protected.

The task adapter verifies that `import aiter` resolves to this materialized
package. An installed image package is not a fallback. Each action still uses a
fresh build directory and must record compilation of a declared candidate source.


## Timed output verification

After collecting the original graph/event measurements, the harness changes
inputs in their existing storage, poisons the captured output and replays the
actual timed graph. It checks BF16 output shape/device, finiteness and the same
reference/tolerance used for ordinary correctness. GEMM uses a positive-scale
stress input; MoE negates hidden states. The stress input is checked after timing
and adds no score point. A stale answer, unwritten buffer or detached correctness
invocation is not accepted. Warmups, repetitions and graph-repeat limits remain
unchanged; a fallback that cannot expose its timed outputs fails explicitly.

The original and perturbed reference outputs are computed from private input
copies before the first candidate invocation, including warmup. The actual
original timed output is checked first. Every read-only input tensor is compared
byte-for-byte before and after replay; input contamination fails. Input buffers
are restored in `finally`, including when replay or a comparison fails. All
snapshot, reference, comparison and restoration work remains outside timing.

## CK dispatch on the current runtime

New AITER tuning defaults can choose FlyDSL or one-stage ASM for these cases,
which bypasses the declared CK candidate. The protected `ck_dispatch.py` binds
both roles to CK stage 1 and CK stage 2 through AITER's metadata transform hook.
It uses the upstream untuned CK shape heuristics and preserves AITER sorting,
quantization, activation, reduction and output allocation. Activation dtype is
explicitly taken from the existing case contract. Every action still rebuilds
and attests the declared CK source; a missing hook or unsupported specialization
fails instead of falling back to another backend.

This restores the named CK task's execution path on the new runtime. It is not a
claim that today's untuned launch configuration reproduces historical production
dispatch timings. Baseline and candidate use the same protected selection;
original cases, reference gates, warmups and sampling remain unchanged.
