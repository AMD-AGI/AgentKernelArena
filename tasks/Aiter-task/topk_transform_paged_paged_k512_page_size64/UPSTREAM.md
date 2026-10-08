# Production Top-k baseline

The production kernel is derived from SGLang commit
[`248c202b46d4a44ad46a3f09d48661b0d9ce6257`](https://github.com/sgl-project/sglang/blob/248c202b46d4a44ad46a3f09d48661b0d9ce6257/python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu).
The unmodified source SHA-256 is
`cf44e2fe37d64cd1f999764b70a38d1d8d956b5450e662d6a6a80a764c8990a2`.
The source retains the SGLang copyright notice and Apache-2.0 terms; the full
license is in [LICENSE.topk](LICENSE.topk).

## Local changes

The upstream FP16 coarse histogram can place more than 6144 candidates in the
threshold bin, while its shared scratch stores only 6144 indices. Silently
truncating that bin loses valid selections for the task's original long-row
input distribution. The task-owned copy checks the threshold-bin population
before filling scratch. If it exceeds capacity, an exact four-pass FP32 radix
refinement rescans the valid row and emits the selected indices without a
bounded candidate list. The original short-row and bounded-bin paths remain.
A standalone binding exposes the same destination-passing operator.

Baseline and initial candidate carry separate identical source copies at
[scripts/baseline/topk_kernel.cu](scripts/baseline/topk_kernel.cu) and
[source/implementation/topk_kernel.cu](source/implementation/topk_kernel.cu).
The candidate never imports the protected baseline. Both wrappers compile their
own adjacent source, and the independent numerical reference is unchanged.

## Runtime and timing

Use the pinned ROCm image from the run configuration on `gfx950`. The initial
production wrapper requires that image's PyTorch HIP runtime, HIP compiler,
C++ headers and Ninja through `torch.utils.cpp_extension`. No runtime source
download or package upgrade is performed. Missing build dependencies fail the
action. Compilation occurs during the first untimed launch and is cached by
source contents, PyTorch/ROCm version and architecture. A writable copy lives in
the standard Torch extension cache, honoring `TORCH_EXTENSIONS_DIR`; hipify and
the compiler never write generated files into the protected task directories.
Only the preallocated output is written by the timed kernel. Metadata remains
a read-only task-ABI input and is not used by this HIP implementation.

The baseline is this corrected production kernel, not the unmodified installed
SGLang operator, and its timings must be measured afresh. The initial Python
wrapper calling HIP is not a completed Triton solution: the final candidate
must provide its own Triton computation under the declared editable boundary.
