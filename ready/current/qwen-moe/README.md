# Qwen MXFP4 MoE scoped qualification

The original two captured MoE cases have complete framework and trusted evidence.
The framework ran at `096452d937d7e5469ff695c4094ca6837a9872ee` on job **209857**, GPU 0, and passed
all **12 checks**, current quality, the reviewed-helper replay audit, all **8 mirrored
integrity observations**, and owned-container retirement before FINAL at
**2026-10-12 19:08:01 UTC**. The trusted comparison ran at
`eed4df573cb0214750c43779f209c663223c2e2f` on job **209670**, GPU 2, and completed all **six fresh phases**
and retirement before FINAL at **2026-10-12 16:50:01 UTC**.

[READY.json](READY.json) pins the exact executions and publication hashes.
[COMPATIBILITY-096452d9.json](COMPATIBILITY-096452d9.json) records the unchanged task tree,
canonical timing helper, numerical implementation, quality/guard code and trusted
runtime between these commits. The retry changed validator guidance and added a
read-only replay auditor. No new trusted execution at 096452d9 is claimed.

| Case | Hidden states | Original capture |
| --- | --- | --- |
| decode_m64_live | BF16 [64, 8192] | SG518 TP8/EP8 |
| prefill_m8192_live | BF16 [8192, 8192] | SG518 TP8/EP8 |

Both cases preserve top-k 10, 64 local experts of 512, packed MXFP4 weights/scales,
routes, mask and native implementations. Each series uses **10 warmups and 100
ordered samples**, with **114 checked invocations and 116 fresh input sets**.
There are 200 framework samples and 400 reference/candidate trusted samples.
The selector is unchanged and **no optimization gain is accepted**.

The [task contract](../../../tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4/config.yaml)
and [task guide](../../../tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4/README.md)
define the literal native-routing boundary and complete reproduction protocol.
Use a fresh authorized MI355X allocation and the exact digest-pinned image:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

The original fixture is 2,083,145,699 bytes, SHA256
`09a71daa55bdd3683be45016f0e5139e2fd63b7c614ae5ba565f74cbd2c7e434`. It remains at its existing object under the
[original fixture registry](../../../tools/headkernel-artifacts.json). This evidence
is not a new SG520 serving capture or model/E2E qualification. Changed candidates,
other cases and later runtime changes require new evidence or explicit review.

The [immutable OCI readback receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__fused_moe_2stage_mxfp4/096452d9-framework-209857__eed4df57-trusted-209670/OCI-VERIFIED.json) pins seven uploaded
payload/metadata objects; that receipt was also uploaded and read back, for eight
objects in total. Every one of the **313 archive members** passed byte, SHA256 and
mode verification from the remote archive. Original tensors/caches were not copied
into the evidence archive.

The exact original fixture object is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_ut_0914_full/20260922/tensors/qwen3.8-2.4t__fused_moe_2stage_mxfp4/ut/reference_io.pt
```

The prior eed4 framework **11 PASS / 1 WARN** report remains unchanged. It is archived
for provenance and contributes no substitute framework PASS. Earlier failed attempts
contribute no phases or samples to this qualification.
