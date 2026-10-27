# Magpie workload catalog

Each workload has its own directory and schema-v2 `config.yaml`, including text,
multimodal and AgentX workloads. Select a directory with the normal task selector.
Its `upstream.yaml` preserves the original workload and `recipe.json` pins its
source identity. `catalog.json` is the upstream synchronization index.

Use the shared binding/materialization workflow in the
[e2e guide](../../../docs/how-to/e2e-kernel-tasks.md#magpie-workload-catalog).
The run config supplies an immutable runtime binding in `serving_runtime.bindings`.
Before starting an agent, the Docker runner prepares the shared evaluator and
protected inputs outside the checkout, then mounts that package read-only at the
selected task directory. There is no manual task-directory generation step and
no model-specific list of allowed kernels. Missing or inconsistent bindings fail
before agent execution. A task directory is not a claim that every runtime binding
has passed GPU qualification.

Maintain generated task declarations with:

```bash
python3 -m src.magpie_catalog export-tasks --check
python3 -m src.magpie_catalog export-tasks
```

The second command adds missing declarations and preserves existing edits.
After reviewing a catalog or shared-instruction update, use `--update` explicitly
to refresh generated files, inspect the diff and repeat the applicable checks.
Unexpected task directories are retained and reported rather than deleted.
