# Forge agent

## Kernel identity metadata

Arena runs Forge and scores the resulting kernel. Optional external knowledge
services are owned by KernelForge; their availability or publication status does
not decide whether Arena writes a score.

Tasks may provide metadata that Arena forwards to `forge-loop`:

```yaml
kernel_identity:
  logical_operator: unified_attention_with_output
  kernel_kind: triton
  source_owner: aiter
  workload:
    source: session_cases.json
    primary_case: representative-case-id
    selector_schema:
      name: hyperloom-v1
      fields:
        q_tokens: QTOKENS
        head_size: HEADSIZE
```

When `kernel_identity` is absent, Arena omits its operator and workload flags and
lets forge-loop use its normal inference/defaults. When present, each supplied
field is forwarded. `kernel_kind` selects the fellow used for the run; it is not
passed as a `--kernel-kind` CLI argument and is not part of KernelForge's
implementation signature.

The source JSON must contain a non-empty `cases` list. Each case must have a
unique `id` and a non-empty scalar `params` mapping. Tasks using session cases
may declare a `hyperloom-v1` selector schema that maps dimensions
Hyperloom can deterministically emit to uppercase selector keys without
underscores. Every original parameter remains preserved in
`session_cases.json`, while only mapped flat selectors are passed through
`primary`/`minimal`/`validation`. Arena derives an unambiguous `shape-v2`
workload key from canonical JSON rather than splitting dimension names on
underscores.

Inline workloads are also supported:

```yaml
kernel_identity:
  logical_operator: rms_norm
  kernel_kind: triton
  source_owner: vllm
  workload:
    shapes:
      primary: {M: 4096, N: 8192, dtype: bf16}
      validation:
        - {M: 1, N: 8192, dtype: bf16}
```

`source_file_path[0]` is the anchor implementation. Additional entries in
`source_file_path` and the optional `editable_sources` list form one complete
edit allowlist passed through `--source-files`. Agents may inspect other
dependencies but must not edit files absent from that allowlist.
`target_kernel_functions` remains the concrete symbol list; it is not a
substitute for `logical_operator`. Keep it focused on useful edit/profile hints
defined in the editable sources. Reuse identity is based on KernelForge's
source-derived pristine implementation signature, so task hints do not
need to reproduce a consumer caller's target list exactly.

## MI355X metadata

Every `tasks/image_kernel/mi355x_*` task declares an explicit
`kernel_identity.logical_operator`, canonical `kernel_kind`, source owner, and
structured workload. CK implementations use `kernel_kind: ck`; AITER ownership
is represented independently by `source_owner: aiter`.

Multi-stage MoE and KDA tasks intentionally use one task-level logical operator
covering their complete measured pipeline. Their Solution patch and workload
therefore represent the combined operation rather than an individual stage.

`mi355x_vllm_triton_unified_attention` and
`mi355x_vllm_triton_paged_attention_2d` share the logical operation
`unified_attention_with_output`, while their source-owner component keeps the
AITER and vLLM implementations on separate Kernel pages.

TileLang metadata uses `kernel_kind: tilelang`; running it requires a matching
KernelForge fellow/backend implementation.
