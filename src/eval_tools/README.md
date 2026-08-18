# Evaluation tools

This package implements the evaluator-side control plane for optional kernel
analysis tools. Operator configuration, the capability matrix, and the full
isolation model are documented in
[`docs/how-to/use-evaluation-tools.md`](../../docs/how-to/use-evaluation-tools.md).

The built-in registry currently contains Triton FpSan, ROCm GPU
AddressSanitizer, the rocJITsu race detector, and HIP-FpSan. A tool name being
listed in a roadmap is not evidence that an optimized kernel was analyzed. A
supported tool also needs a pinned sidecar, a candidate-specific adapter,
instrumentation or dispatch attestation, a passing startup control, and a
fail-closed parser before it can produce a clean result or gate scoring.

## Planned sanitizer coverage

### ThreadSanitizer (TSAN)

ThreadSanitizer detects data races and synchronization defects caused by
concurrent accesses that are not ordered correctly. For GPU evaluation, this
would complement the current rocJITsu checks with compiler-instrumented dynamic
race detection on applicable device code.

TSAN is not currently registered as an AgentKernelArena evaluation tool. The
pinned ROCm 7.2 HIP compiler used by the `gfx950` scoring baseline warns that
`-fsanitize=thread` is unsupported for the `amdgcn-amd-amdhsa` device target
and ignores the option. Host `libclang_rt.tsan` files do not establish device
instrumentation or GPU runtime coverage. Public ROCm development documentation
describes device-side TSAN builds for `gfx942` and `gfx950`, but that project
development path has not been qualified as a pinned, general-purpose evaluator
runtime for this benchmark.

Future TSAN support should be added only after a compatible device compiler and
runtime can be pinned in an isolated sidecar. Promotion requires exact candidate
recompilation and build attestation, safe/racy startup controls, end-to-end
`gfx950` fixtures, structured findings, and fail-closed handling of missing or
partial instrumentation. Until those requirements are met, TSAN remains
unsupported and must not be reported as kernel coverage.

Upstream status:

- [ROCm Compute Profiler sanitizer build documentation](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/docs/install/source-install.rst#sanitizer-builds)

### UndefinedBehaviorSanitizer (UBSAN)

UndefinedBehaviorSanitizer detects runtime instances of C and C++ undefined
behavior, including invalid shifts, signed integer overflow, division by zero,
misaligned accesses, and other enabled checks. These findings can be useful in
host launchers and native support code, but host instrumentation does not prove
that an optimized GPU kernel was checked.

UBSAN is not currently registered as an AgentKernelArena evaluation tool. ROCm
documents its current development build as host-only, and the pinned ROCm 7.2
HIP compiler warns that `-fsanitize=undefined` is unsupported for the
`amdgcn-amd-amdhsa` device target and ignores the option. AgentKernelArena must
therefore not treat a host-only UBSAN run as device-kernel sanitizer coverage.

Future GPU UBSAN support depends on upstream device instrumentation and a
compatible device runtime becoming available. Once available, it will require
the same isolated sidecar, exact-artifact attestation, positive/negative
controls, structured parser, and `gfx950` qualification used for the current
tools. A host-only UBSAN lane may be considered separately, but it must remain
explicitly scoped to evaluator adapters or launchers and must not satisfy a GPU
kernel gate.

Upstream status:

- [ROCm Compute Profiler sanitizer build documentation](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/docs/install/source-install.rst#sanitizer-builds)
