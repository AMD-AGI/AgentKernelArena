# End-to-end kernel optimization

The upstream workload recipe retained for the 80B task is copied unchanged from
[Magpie examples/benchmarks](https://github.com/AMD-AGI/Magpie/tree/f4bed327e434c19ef8daba5868027c590abf032c/examples/benchmarks).
`upstream_manifest.json` records the source commit, original paths and SHA256
digests. `MAGPIE_LICENSE` preserves the upstream license.

The legacy manifest links that retained recipe to its maintained executable task.
The reusable [Magpie catalog](magpie/README.md) separately pins the complete newer
recipe collection and binds/materializes selected recipes through shared backend
and scenario adapters. An imported recipe alone is not a qualified executable
Arena task. Only a
directory with an Arena schema-v2 `config.yaml` is discovered as a task.
Qualification requires a framework-finalized GPU task-validator PASS.

The task owns the serving configuration and correctness policy. Baseline and
candidate use identical model weights, precision, parallelism, feature flags,
request corpus, concurrency and output lengths. Only declared kernel source
may change. Enabling an existing feature or changing TP/EP/PP cannot earn credit.

Model-specific task variants declare their differences explicitly; imported
upstream YAML files remain unchanged. Inference dependencies and runtime images
must be pinned before an executable variant is added.

The first executable task is `qwen3_0_6b_sglang`. See the
[e2e guide](../../docs/how-to/e2e-kernel-tasks.md) for runtime preparation, fixed
settings, budgets and qualification requirements.

`qwen3_next_80b_sglang` provides a TP4 Qwen3-Next-80B-FP8 variant. Both executable
tasks accept multi-file kernel implementations and fusion at compute boundaries
throughout the model, with whole-model throughput measurement. Their supplied
RMSNorm and Gated DeltaNet implementations are starting points, not the allowed
optimization scope. Imported workloads remain unchanged. See each task README
for submission, fixed runtime settings and finite correctness coverage; adding
the package does not establish GPU qualification. Profiling the actual workload
is recommended, but agents choose their tools and no profiling report is required.

New catalog workloads reuse high-level model numerical checks, fixed-work
measurement and the independent `candidate_verifier`. They do not require an
enumerated operator suite or a new oracle for every fusion. Import, materialization,
GPU qualification and accepted optimization results remain separate outcomes.
