# Earlier verification record (2026-09-15)

**Historical note.** This page used to record the September 2026 verification of
the task families that preceded the schema-v3 kernel tasks (`triton2triton`,
`hip2hip`, `torch2hip`, the FlyDSL families, `Aiter-task` and `image_kernel`),
which have since been migrated and removed, together with the `task_validator`
agent that qualified them. The detailed per-task tables no longer describe any
task in the repository and were removed.

Current qualification is by the [QA auditor](../how-to/qa-auditor.md): a task is
qualified by a framework-finalized `qa_report.yaml` with `status: PASS` bound to
its exact package hash. Task qualification, agent optimization results and
runtime qualification remain separate claims.

## Final agent matrix

The last matrix of that campaign covered Codex, Claude Code, Forge and GEAK on
five tasks from each of nine former task families (180 completed pairs, 173
accepted). It established that those agents could complete the former task
lifecycle; it does not describe the current tasks or agent versions. Re-run the
agent of interest on current tasks with the [run guide](../how-to/run-evaluation.md).
