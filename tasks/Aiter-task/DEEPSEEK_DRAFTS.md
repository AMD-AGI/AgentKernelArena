# DeepSeek task packages

These 18 schema-v2 packages cover 13 block-scaled FP8 GEMMs, three MLA variants,
one mHC operator, and one paged Top-k operator. They retain all 234 original
workload rows and extend the MLA and Top-k runtime cases to cover lengths,
padding, and graph replay. The complete selection contains **435 cases**.

The task-local input generators, independent references, comparators, workloads,
and runners do not depend on the task generator at runtime. Each task provides
an executable initial implementation and a production baseline. The final
submitted candidate must implement its own GPU computation in Triton; passing
initial task qualification does not certify a later optimized implementation.

## Qualification

Fresh MI355X (`gfx950`) validation on **2026-10-08 UTC** produced
**18 PASS, 0 WARN, and 0 FAIL**. Each framework-finalized report passed its
initial validation gate, evidence verification, and all semantic checks.
All seven runtime actions passed for every task: task validation and baseline
and candidate compilation, correctness, and performance.

| Workload group | Tasks | Cases | Formal result |
| --- | ---: | ---: | --- |
| Block-scaled FP8 GEMM | 13 | 169 | 13 PASS |
| MLA without extra KV | 1 | 29 | PASS |
| MLA with page-size-2 extra KV | 1 | 63 | PASS |
| MLA with page-size-64 extra KV | 1 | 54 | PASS |
| mHC | 1 | 13 | PASS |
| Paged Top-k | 1 | 107 | PASS |
| **Total** | **18** | **435** | **18 PASS** |

The selected task paths are maintained in the
[validator configuration](../../example_configs/task_validator_deepseek_drafts_mi355x.yaml).
Validation covered the implemented initial candidates. It establishes usable
correctness and measurement contracts, not an optimization gain. Any later
material task or harness change requires fresh qualification.

## Resolved findings

The earlier run reported 2 PASS, 12 WARN, and 4 FAIL. The following repairs
preserve the original shapes, workload identities, numerical tolerances,
warmups, sample counts, and device-timing policy:

- GEMM and mHC now have task-specific instructions with working local links,
  explicit runtime dependencies, and separate initial and final implementation
  requirements.
- MLA now exercises independent main/extra lengths, empty and boundary lengths,
  mixed rows, negative padding within selected prefixes, and valid indices
  outside those prefixes. Correctness and measured replay checks use the same
  declared cases; replay refills data and lengths in the existing buffers.
- Top-k covers empty, short, intermediate, full, mixed, and random lengths.
  The comparator checks logical output order for every row with length at most
  512, including the exact 512 boundary. Measured replay is checked after both
  data and length changes.
- Top-k uses a task-local production HIP kernel with an exact overflow repair.
  The installed upstream kernel could truncate a threshold bin larger than its
  6144-index scratch capacity and lose valid selections on long rows. The
  repaired kernel rescans overflowing rows with FP32 radix refinement while
  preserving the bounded-bin fast path. Baseline and initial candidate have
  separate source copies; neither uses the numerical reference as computation.
  See [the source record and license](topk_transform_paged_paged_k512_page_size64/UPSTREAM.md).
- All 18 runners use the canonical benchmark helper's `TimedRun` implementation,
  fixing the obsolete local binding interface. Output poisoning and exact
  measured replay validation remain enabled.

The Top-k repair also passed GPU regression checks at the 6144/6145 boundary,
with negative and concentrated scores and changed-length graph replay.

## Runtime and reproduction

The qualified runtime is the immutable image pinned by the run configuration:

```text
lmsysorg/sglang@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
```

The tested image supplied ROCm PyTorch `2.11.0+rocm10.0.0`, Triton `3.8.0`,
AITER, SGLang, and their GPU runtime dependencies. No package upgrade was used.
The runner supplies writable worker-specific AITER and FlyDSL caches and a
writable AITER configuration copy. The Top-k source compiles in a separate
Torch extension cache; generated build files do not modify the task sources.

From the repository root on compatible MI355X hardware, run the complete
selection through Docker:

```bash
make docker-smoke
make docker-check-agents CONFIG=example_configs/task_validator_deepseek_drafts_mi355x.yaml
make docker-run CONFIG=example_configs/task_validator_deepseek_drafts_mi355x.yaml
```

For an eight-GPU node, the same selection can run in parallel:

```bash
make docker-parallel-run CONFIG=example_configs/task_validator_deepseek_drafts_mi355x.yaml GPU_IDS=0,1,2,3,4,5,6,7
```

Explicit image environment overrides take precedence over the configuration.
See the [Docker workflow](../../docs/install/install.md) and
[task-validator guide](../../docs/how-to/task-validator.md).

CPU regression checks cover packaging, isolated materialization, source and
runtime declarations, input integrity, coverage, comparator behavior, and timing
helper integration. These checks complement the retained GPU reports; they do
not replace GPU qualification. Inspect each fresh `validation_report.yaml` and
its evidence binding after reproduction. WARN, FAIL, partial, or stale reports
are not clean passes.
