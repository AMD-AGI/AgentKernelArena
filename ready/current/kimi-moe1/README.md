# Kimi MoE stage 1 completed qualification gates

The three captured Kimi MoE stage 1 cases have complete trusted and framework
gates on job **210191**, with independent verification and immutable OCI evidence.
**Root approval of scoped readiness is pending.** These gates are not yet added
to the branch's completed-task count.

Both gates retain the task base
`18bb4d67bea577f6951f51a95cf524c9c425f952`, task tree
`d9af79a96a8c3b8abe8fe00246502255981c20f0`. The framework additionally used the
finite-session prompt overlay
`cf563f6152e1a667754d7808c478a7a7c012e101f706636473d68579b12b5df6`.
That overlay requires awaiting configured commands and finishing the draft
report; it changes no task, case, sample count or numerical criterion.
[READY.json](READY.json) pins the completed executions and publication.
[INPUT-PINS.json](INPUT-PINS.json) retains all 307 task input identities and the
explicit framework overlay.

| Gate | Actual execution | Complete evidence |
| --- | --- | --- |
| Trusted comparison | GPU 0; FINAL at 2026-10-13 01:22:18 UTC | Six fresh phases, three cases, 1,200 paired inner samples, 27 retained routing-population correctness checks per outer leg. |
| Framework validator | GPU 1; FINAL at 2026-10-13 00:50:32 UTC | All 12 checks PASS, three cases, 600 paired inner samples, complete submitted-report and post-validator integrity chain. |

The trusted count comprises two outer runs, each containing a candidate-port and
protected-reference series of 100 samples for each case. The framework count
comprises the same two inner series in one run. Every series retains 10 policy
warmups. Both gates preserve all 307 input observations and owned-container
retirement before FINAL; trusted inputs were observed before and after each of
its six phases. No phases or samples from earlier partial attempts are reused.

| Case | Activation shape | Captured output payload shape |
| --- | --- | --- |
| Decode | `[64, 3584]` | `[64, 16, 384]` |
| Prefill | `[8192, 3584]` | `[188416, 384]` |
| Prefill | `[16384, 3584]` | `[319488, 384]` |

The [task contract](../../../tasks/headkernel/kimi-k3__moe_gemm1_stage1/config.yaml)
uses **protected_reference** scoring. Its reference includes the declared
async-LDS synchronization repair; it is not byte-identical to the stock native
image. This is an unchanged-source control and **no optimization gain is
accepted**. The [task guide](../../../tasks/headkernel/kimi-k3__moe_gemm1_stage1/README.md)
and [repair provenance](../../../tasks/headkernel/kimi-k3__moe_gemm1_stage1/provenance/STAGE1-SYNC-REPAIR.json)
describe the operator ABI, editable boundary and original failures. Historical
pending-status text in the frozen task guide predates these completed gates.

Correctness retains seeds 0, 1 and 2, no-op and wrong-output controls, and all 27
retained routing populations. Timing samples the captured work distributions;
it does not establish exhaustive timing of every histogram bin, missing routing
frequencies, full routing histories, model qualification or serving/E2E gains.

Use the [immutable OCI publication guide](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261013/qualification-evidence/kimi-k3__moe_gemm1_stage1/18bb4d67-prompt-cf563f61-job210191/README.md)
for download, full archive verification and the portable read-only audit.
That audit uses the archived validators with Python and PyYAML; it requires no
GPU or original execution node. Raw receipts retain their execution paths, and
the portable verifier explicitly remaps them within the downloaded bundle.

The 29 fixture assets totaling 1,666,534,545 bytes remain pinned by the
[external fixture manifest](../../../tasks/headkernel/kimi-k3__moe_gemm1_stage1/fixtures/EXTERNAL-MANIFEST.json).
The evidence retains their full in-place byte observations and does not duplicate
bulk tensors or runtime caches. The [portable setup guide](../../../docs/how-to/headkernel-upstream-runtime.md)
covers fixture preparation and the digest-pinned SGLang 0.5.20 MI355X runtime.
Changed candidates, shapes, runtime or framework require matching evidence or
an explicit compatibility review; this handoff claims no new execution at a
later branch tip.
