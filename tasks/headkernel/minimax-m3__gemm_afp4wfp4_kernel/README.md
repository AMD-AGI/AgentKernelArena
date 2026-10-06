# minimax-m3__gemm_afp4wfp4_kernel — CPU draft, capture integration pending

The current pinned SG520 runtime still executes this original MiniMax head at
the Quark dense-linear seam. Its sampled native profile contains 7,296 prefill
and 7,296 decode kernel events (912 per rank per stage). These are profile-window
counts, not full-workload frequencies. Prefill uses BM256/BN256/BK256 and decode
BM32/BN32/BK512; those tile sizes do not establish matrix shapes. Exact profile
receipts and the recipe-parity result are in `provenance/OBSERVED-PRESENCE.json`.

This directory now contains a CPU-tested draft with a protected importer, runner
and native smoke entrypoint. It has no `config.yaml`, actual case manifest,
operand fixtures, timing report, or task-validator result. The original
`NOT_BUILT` marker below remains as historical evidence of the unfinished head.
Do not count the three existing MiniMax attention tasks as coverage of this
separate dense GEMM.

## Prepared contract

- `source/kernel.py` and `ut/reference/kernel.py` are identical copies of the
  current image's full kernel module. Only `_gemm_afp4wfp4_kernel` is editable.
  Imports, decorators, signatures, other kernels and configuration lookup are
  frozen by `ut/source_guard.py`.
- `ut/native/wrapper.py` and `ut/native/quark_linear.py` preserve exact current
  native sources. `SOURCE-PROVENANCE.json` pins them and the runtime dependencies.
- Activations have logical shape `(M, 2*K_bytes)` and packed shape `(M,K_bytes)`;
  weights arrive as `(N,K_bytes)` and the native wrapper transposes their view.
  The low nibble precedes the high nibble. E8M0 scales cover 32 logical values.
  Scale byte zero means `2**-127`; byte 255 is NaN and is never silently erased.
  Capture preserves real strides, offsets, aliases and full storage, including
  any padded scale allocation.
- `ut/reference.py` implements independent CPU decoding and mathematical dot
  products, including strided byte views and split-K partials. Its small CPU
  examples are format tests, not claimed production cases. Native FP32
  accumulation, optional BF16 partials, final casting and numerical tolerance
  still require calibration against captured outputs.
- `ut/binding.py` loads each leg under a private module name and binds the
  wrapper's actual kernel global to it. The loader removes exactly the
  `torch_compile_guard` decorator from the frozen wrapper AST, preserving its
  body. This avoids dispatch through an already registered process-global
  `torch.ops.aiter.gemm_afp4wfp4_`. This adaptation is explicit provenance and has
  not yet been checked on a GPU. Supply the captured reduction mode and config;
  do not infer them from tile names.
- `scripts/task_runner.py` refuses missing or unadmitted capture before GPU
  initialization. Each admitted case first checks captured outputs against the
  independent CPU packed-value oracle. Replays use fresh row permutations and
  FP4 sign changes with the corresponding scales, preserve CPU-only truth and
  complete storage, poison outputs, and compare after execution. Graph timing
  retains 10 warmups and 100 checked samples. Padding and input mutation checks
  remain outside the timed interval.
- `scripts/make_source_controls.py NEW_DIRECTORY` prepares guarded source-only
  no-op and zero-output candidates. Their numerical rejection still requires
  a valid native reference and actual eager/graph execution.

## Capture integration

`capture/INTEGRATION.json` is the handoff for the shared-capture owner.
`capture/adapter.py` imports without starting capture. Install it with the
pinned V4 recorder, loaded Quark/basic/kernel modules, and an owner callback
providing the recorder, current served context, and graph identity/slot/bucket.
The owner must initialize the recorder and admit memory before graph capture,
notify actual served graph replays, then seal and verify every rank.

The adapter replaces Quark's `_gemm_afp4wfp4_orig` global, which the already
registered Quark custom-op implementation reads at execution. It also observes
the exact pinned Triton kernel object's `run` method. A native smoke exposed
that AITER's legacy-module redirect executes the wrapper module twice, while
Torch registration retains the first function's globals. Replacing only the
current module's kernel global therefore missed the actual launch. The shared
kernel-object probe covers both module instances, still requires exactly one
launch, and reads split-K dtype mode from the executing native wrapper frame.
The failed run and CPU reproduction are described in
`provenance/NATIVE-CAPTURE-BINDING-REPAIR.json`; this revision needs a fresh native
smoke. The adapter captures `x`, `w`, both
scales, optional `y`, returned output, requested dtype/config, resolved launch
controls, all scalar strides, `skip_reduce`, and `_USE_GEMM_SPLITK_BF16`. Input
quantization, bias addition and the fused pre-quant/split-cat variants are outside
this head. The adapter must run in a fresh capture bundle alongside, without
changing, the existing three attention-family adapters.

Before task activation, obtain complete prefill/decode matrix/layout/control
coverage and exact full-workload all-rank counts, verify output/alias/storage
receipts, and admit actual fixture payloads. Then activate the protected runner
and scoreable manifest, prove submitted-source eager/graph controls, and run native
and framework qualification. Missing capture is never filled with guessed
shapes, weights, call distributions, or successful reports. The protected runner
and importer are prepared; activating the task configuration and its numerical
policy still requires those actual cases and calibration.

## Native smoke and capture admission

The capture owner can prepare a fresh, reviewable smoke snapshot and Docker
command without launching anything:

```bash
python3 scripts/prepare_native_smoke.py --output NEW_SMOKE_DIRECTORY \
  --common SHARED_CAPTURE_V4 --binding-helper TRUSTED_GPU_BINDING_HELPER \
  --expectation CURRENT_GPU_EXPECTATION_JSON
```

The parent owns GPU admission and execution of the resulting `PLAN.json`
command, its timeout and cleanup. The command uses the exact pinned image,
read-only task/common snapshots, fresh Triton cache, and the trusted same-process
GPU preflight. `native_smoke.py` calls the already registered original Quark
custom op, checks an exact eager result, captures/replays a real graph with
changed packed inputs, then tests private reference/candidate callbacks and
both submitted-source negatives in eager and graph modes. The M64/N64/K512
operands are explicitly synthetic diagnostic data. Smoke reports are
non-scoreable and contain zero performance samples; the workload importer
rejects synthetic fixtures. No native smoke PASS has yet been obtained.

For actual capture, the owner supplies the full-workload receipt schema defined
in `capture/INTEGRATION.json`, including all eight sealed rank manifest hashes,
successful request completion and draining. Import into a new directory:

```bash
python3 scripts/import_capture.py --common SHARED_CAPTURE_V4 \
  --receipt FULL_WORKLOAD_OWNER_RECEIPT --oracle-policy REVIEWED_ORACLE_POLICY \
  --output NEW_DATASET
python3 scripts/task_runner.py compile --dataset NEW_DATASET
python3 scripts/task_runner.py correctness --dataset NEW_DATASET
python3 scripts/task_runner.py performance --dataset NEW_DATASET
```

The oracle policy is explicit JSON with `metric: mixed_rms`, a positive
`tolerance` no larger than 0.02, and a nonempty review `basis`. It has no default
or automatic relaxation. Every captured output must calibrate against the
independent packed-value oracle before a candidate can pass. The importer
verifies raw segment hashes and complete storage, native launch controls,
cross-rank case identities and exact observed frequencies, and retains portable
provenance receipts. It does not activate `config.yaml` or mark qualification.

`scripts/check_source_binding.py` accepts an admitted dataset, exact case ID,
seed, eager/graph mode, and source-only candidate workspace. It first calibrates
the frozen reference in the requested mode. A numerical bad-source rejection
exits 1; invalid references and setup/compile failures exit 2 and never count as
successful rejection. Valid stock source exits 0. These diagnostic reports also
contain zero performance samples and cannot establish model-level completion.

CPU checks:

```bash
python3 scripts/check_draft.py
python3 -m pytest -q tests
```

## Historical unbuilt inventory

**MiniMax-M3-MXFP4** - `_gemm_afp4wfp4_kernel` (Triton, 5.14% GPU).

This is a placeholder, **not a task**. It carries no `config.yaml`, so the arena
task scanner will not pick it up and it cannot be run or scored.

## Why

IMAGE-VERIFIED 2026-09-15: the editable source EXISTS in the row's own runtime image. sglang:v0.5.17-rocm720-mi35x-profilerfix (sglang 2948168546, aiter d9e5ef7ce08e) carries the Triton kernel at /sgl-workspace/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py:36 (_gemm_afp4wfp4_kernel, 25526 bytes) and its wrapper at aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py:259 (gemm_afp4wfp4, 21731 bytes); a gluon variant also exists. The UT package is the problem, not the source: kernel_src/ holds only .gitkeep, candidate_bind is the empty stub {'kind':'module','module':'','file':'kernel_src/'}, and there is no reference_io.pt - unittest.py prints the UT_HARNESS_INCOMPLETE sentinel and exits 3. Remaining work: vendor the two files, fill in candidate_bind, and CAPTURE AN ORACLE - that last one needs a live server run and is the real cost.

| field | value |
|---|---|
| GPU time share | 5.14% |
| empirical roofline | 24% |
| optimized roofline | - |
| e2e uplift measured | - |
| device symbol | `_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_512_...` |
| production seam | `aiter.ops.triton.gemm_afp4wfp4:gemm_afp4wfp4` |
| info rows | MM-4 |
| upstream UT package | `Z/MiniMax-M3-MXFP4_gemm_afp4wfp4_kernel` |

## To promote it into the suite

1. Get an editable implementation of the seam into `source/`. Read the Why above
   first - it says whether the source has to be written (the profiled symbol is a
   prebuilt vendor artifact with no Python behind it) or merely vendored (the
   package shipped an empty `kernel_src/` but the Triton source exists upstream,
   and for the `aiter.tuned_gemm` rows a stock copy already ships in this suite at
   `tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster/source/`).
2. Add `candidate_bind` to the package `meta.json` so the candidate leg actually
   shadows the production callable - without it both legs resolve to the same code
   and any measured speedup is noise. For the `aiter.tuned_gemm` rows note that a
   bare `setattr` on the module is a DEAD rebind: `solMap` is built at import time
   holding direct function objects, so the dispatcher keeps calling the original.
3. Re-capture parity against the live server and confirm
   `selection_validation.ok == true`.
4. Re-run `tools/build_suite.py`; flip `built` to true in `tools/manifest.json`.
