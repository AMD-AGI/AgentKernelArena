# End-to-end kernel optimization

These workload YAML files are copied without modification from
[Magpie examples/benchmarks](https://github.com/AMD-AGI/Magpie/tree/f4bed327e434c19ef8daba5868027c590abf032c/examples/benchmarks).
`upstream_manifest.json` records the source commit, original paths and SHA256
digests. `MAGPIE_LICENSE` preserves the upstream license.

The directory ID is the original relative filename without its extension.
An imported recipe is catalog data, not an executable Arena task. Only a
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
