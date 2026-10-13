# Head-kernel campaign progress — October 13, 2026

The qualification snapshot at `378eac7c` retains **nine scoped qualifications out of
23 tasks; 14 remain unfinished**. The complete branch remains unqualified.
This note records partial MiniMax decode-score evidence and campaign execution
history. **It is not a ready-task handoff and claims no optimization or
serving/E2E gain.** The [qualification index](headkernel-ready-subsets.md)
remains authoritative for the nine completed scopes.

## MiniMax decode-score progress

The task has 16 required units. A complete trusted unit seal covers all six
reference/candidate compile, correctness and performance phases; an incomplete
phase or unsealed unit does not increase accepted coverage.

| Attempt | Recorded progress | Remaining gate |
| --- | --- | --- |
| Trusted four-unit wave, job 211562 | Four new complete unit seals, bringing accepted coverage to **6/16**. All 24 new phase containers exited zero and retired; all four seals retain reference/candidate timing-quality PASS. R6 portable output acceptance subsequently succeeded. | The accepted output remains partial and contains no full 16-unit measurement. |
| Framework eight-worker run, job 211621 | Compile and correctness passed for **16/16**, with negative controls and GPU identities retained. **0/16 performance units were sealed** before interruption at **08:24:12 UTC**. | Complete performance, the finalized framework result and full qualification remain outstanding. |
| Trusted wave allowing up to six new units, job 211888 | The final portability audit records **zero new complete seals** and **6/16 inherited seals** after preemption at **09:53:00 UTC**. The wave exited with SIGTERM. | Ten units remain outside accepted trusted coverage; this wave produced no output-acceptance receipt or complete task measurement. |

Complete unit seals may be retained only through the reviewed checkpoint
acceptance path. These partial trusted and framework results do not constitute
a completed full-task qualification when combined.

## Published partial evidence

The [fixed terminal progress snapshot](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261013/campaign-progress/minimax-m3__decode_score_kernel/checkpoints-through-j211888-f8-j211621-PARTIAL_NOT_QUALIFIED-20261013T0954Z/CAMPAIGN-PROGRESS.json)
is dated **09:57:19 UTC** and labeled `PARTIAL_NOT_QUALIFIED`. It preserves the
six complete seals, unresolved attempts and the separate F8 evidence without
promoting them to a full result. The
[OCI readback receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261013/campaign-progress/minimax-m3__decode_score_kernel/checkpoints-through-j211888-f8-j211621-PARTIAL_NOT_QUALIFIED-20261013T0954Z/OCI-VERIFIED.json)
records verification of all **10 objects**, all **3,736 main-archive members**,
and all **136 members of the preserved F8 archive**, including file bytes and
modes. The source snapshot was unchanged during publication.

Readback-receipt SHA-256:
`7b2ff5f2e5f933cd159e1bb50d53c2acc8ff40de66fbab674cf330196ba486ef`.

## Execution limits and campaign tooling

The scheduler records for Burst jobs 211562, 211621 and 211888 show **00:20:00**
runtimes and preemption, ending at **08:01:01 UTC**, **08:24:12 UTC** and
**09:53:00 UTC**, respectively.
GEAK jobs **210785** and **211714** never allocated: both retain zero runtime,
no start time and no assigned node before cancellation. These observations
provide no completion-time guarantee for a future allocation.

The runs used the exact SGLang 0.5.20 MI355X image:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

The trusted checkpoint waves retain execution revision `fd289d68`. The
framework run retains its `18bb4d67` base and explicitly declared setup overlays,
including a pinned read-only rclone mount. Cache-copy startup uses explicit
resource admission; the reviewed R6 continuation and NFS identity wrappers are
external campaign controls. This progress note does not describe newly merged
task-runtime changes or broaden any qualification scope.
