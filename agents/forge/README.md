# Forge agent

## Knowledge-base modes

The run config selects producer intent without environment variables:

```yaml
agent:
  template: forge
  knowledge_base:
    mode: producer
    finalization_margin_seconds: 900
```

These values overlay the defaults in `agent_config.yaml`. The available modes
are:

- `compatibility`: normal non-KB Arena optimization and scoring. Forge KB
  producer metadata and gbrain credentials are optional.
- `producer`: one publication-required Forge run. Preflight requires inherited
  `GBRAIN_BASE_URL` and `GBRAIN_TOKEN`, a `kernel-agents` executable on `PATH`
  that exposes the Forge KB producer contract, and complete task metadata. A
  nonzero Forge exit or a result that does not confirm publication of the latest
  local best fails the task before Arena writes a score.

Producer success uses KernelForge's commit-level publication proof:
`remote_publication.published_commit` must equal `best_commit` and
`pending_commit` must be empty. The mirrored
`kb_experience.publication` object is accepted when the top-level object is
absent. A later summary refresh returning `not_better_than_kb` does not invalidate
an earlier durable publication of that same best commit.

Producer task metadata uses this schema:

```yaml
knowledge_base:
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

`knowledge_base.kernel_kind` is Arena-internal producer metadata. Arena uses it
to select and validate the KernelForge fellow/backend before launch; it is not
passed as a `--kernel-kind` argument and is not part of KernelForge's
implementation signature.

The source JSON must contain a non-empty `cases` list. Each case must have a
unique `id` and a non-empty scalar `params` mapping. Producer tasks using session
cases declare a `hyperloom-v1` selector schema that maps only dimensions
Hyperloom can deterministically emit to uppercase selector keys without
underscores. Every original parameter remains preserved in
`session_cases.json`, while only mapped flat selectors are passed through
`primary`/`minimal`/`validation`. Arena derives an unambiguous `shape-v2`
workload key from canonical JSON rather than splitting dimension names on
underscores.

Inline workloads are also supported:

```yaml
knowledge_base:
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
source-derived pristine implementation signature, so producer task hints do not
need to reproduce a consumer caller's target list exactly.

## Unresolved task metadata

`mi355x_vllm_triton_unified_attention` is producer-ready. Its exact Hyperloom
logical identity is `unified_attention_with_output`; the source owner is aiter,
the implementation kind is Triton, and its editable source set is the same
single `_triton_kernels/attention/unified_attention.py` implementation selected
by Hyperloom. Its selector schema contains only trace-visible token/head
dimensions; context, block size, and dtypes remain preserved session metadata.

The three `mi355x_vllm_ck_*` image tasks declare the known CK implementation
kind, aiter source ownership, and checked-in workloads. They are not yet
producer-ready because their `session_cases.json` operator values are Arena
harness labels, not preserved Hyperloom logical identities. In particular, the
MoE tasks combine stage 1 and stage 2 even though Hyperloom routes those stages
as separate logical operators. Split those producer tasks by logical operation
and record the exact invocation identity before adding `logical_operator`.
The a8w8 task likewise requires confirmation of the original invocation
identity; `a8w8_blockscale_gemm` must not be assumed to be that identity.

`mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3` uses the FlyDSL fellow, but is not
producer-ready: one Arena task combines the confirmed
`moe_flydsl_stage1`/`moe_flydsl_stage2` operations with decode graph nodes whose
backend was not resolved. Split it into one logical operator per producer task
or add metadata only after the intended Hyperloom logical identity is known.

Tasks without `knowledge_base.logical_operator`, `kernel_kind`, and a workload
remain valid in compatibility mode. Producer preflight lists missing metadata
instead of deriving an operator from a task directory or guessing a backend.
