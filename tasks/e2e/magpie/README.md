# Magpie workload catalog

`catalog.json` pins every benchmark recipe from one Magpie commit, including
text, multimodal and AgentX recipes. It is an importable catalog, not a claim
that every model/image/hardware combination has passed GPU qualification.

Use the shared binding/materialization workflow in the
[e2e guide](../../../docs/how-to/e2e-kernel-tasks.md#magpie-workload-catalog).
Generated packages reuse the same whole-model evaluator and independent
candidate verifier. They do not contain a model-specific list of allowed kernels.
New upstream recipes normally need a catalog refresh and runtime binding rather
than another copy of the evaluator.
