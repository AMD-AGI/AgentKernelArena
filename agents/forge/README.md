# Forge agent

## Kernel identity metadata

Arena runs Forge and scores the resulting kernel. Optional external knowledge
services are owned by KernelForge; their availability or publication status does
not decide whether Arena writes a score.

Tasks may provide kernel identity metadata:

```yaml
kernel_identity:
  logical_operator: unified_attention_with_output
  kernel_kind: triton
  source_owner: aiter
```

Arena forwards `logical_operator` as `--operator-name` and `source_owner` as
`--framework`. `kernel_kind` selects the fellow used for the run; it is not
passed as a `--kernel-kind` CLI argument. The kernel anchor and complete editable
source list are forwarded through `--kernel` and `--source-files`.

Arena does not derive Forge selectors from task workloads. The launcher command
never includes `--shapes-json` or `--workload-key`; correctness and benchmark
drivers run the task's complete case suite.

`source_file_path[0]` is the anchor implementation. Additional entries in
`source_file_path` and the optional `editable_sources` list form one complete
edit allowlist. Arena passes these paths through `--source-files` for Forge's
orientation and KB identity, then rejects the final result if its Git diff
escapes that allowlist. Non-ignored untracked scratch files are discarded before
scoring. Agents may inspect other dependencies but must not edit files absent
from that allowlist.
`target_kernel_functions` remains the concrete symbol list; it is not a
substitute for `logical_operator`. Keep it focused on useful edit/profile hints
defined in the editable sources. Reuse identity is based on KernelForge's
source-derived pristine implementation signature, so task hints do not
need to reproduce a consumer caller's target list exactly.

## MI355X metadata

Every `tasks/image_kernel/mi355x_*` task declares an explicit
`kernel_identity.logical_operator`, canonical `kernel_kind`, source owner, and
task workload. CK implementations use `kernel_kind: ck`; AITER ownership is
represented independently by `source_owner: aiter`.

Multi-stage MoE and KDA tasks intentionally use one task-level logical operator
covering their complete measured pipeline. Their Solution patch and workload
therefore represent the combined operation rather than an individual stage.

`mi355x_vllm_triton_unified_attention` uses the logical operation
`unified_attention_with_output`, while
`mi355x_vllm_triton_paged_attention_2d` uses `paged_attention_2d`. Keeping the
logical operations distinct prevents paged-attention recipes from being reused
as unified-attention implementations.

TileLang metadata uses `kernel_kind: tilelang`, so Arena forwards
`tilelang-fellow` without substituting another backend. KernelForge decides
whether the installed version supports that fellow.
