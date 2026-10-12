# Current qualification: Kimi Lean attention decode

The original **batch-64, page-1 decode case** is ready within its reviewed scope at
**`eed4df573cb0214750c43779f209c663223c2e2f`**. This covers the recorded **8193–9216
sequence-length distribution**, correctness seeds **0/1/2**, and **10 warmups /
100 measured samples**. It does not claim exhaustive timing of every length.

| Evidence | Actual execution | Completed scope |
| --- | --- | --- |
| Retained trusted comparison | `20c1a949`, job **200712** | The complete original six-phase comparison; all original paired timing series and reports remain intact. |
| Compiled source controls | Separate job **201533** | Unchanged-source, no-op and wrong-output controls, preserved with their original identity. |
| Fresh framework | `eed4df57`, job **209907**, Codex `gpt-5.6-sol` / `max` | **All 12 checks PASS**, one case, **100 candidate + 100 reference samples**, **110 checked pairs**, current paired quality and post-validator integrity. |

The framework exited successfully and retired its owned containers at
**2026-10-12 19:46:28 UTC**. The host wrapper then failed while importing PyYAML.
Root captured the completed canonical report, marker, integrity and phase records
at **19:49:03 UTC**, before preemption. The original `RESULT`, `STATUS` and
`SEQUENCE-EXIT` remain `INCOMPLETE_OR_REJECTED`; no original `FINAL.json` was created.

A separate **CPU-only recovery**, approved by the independent guard review, verified
the full captured evidence using the exact `eed4` consumers and an existing
YAML-capable environment. The actual post-validator receipts cover **all 39 inputs**:
**38 raw before/after harness and fixture hashes**, including **14 fixture blobs**,
plus an unchanged full-byte declared-source fingerprint. The recovery does not
invent the host input receipts that were never written or claim a new node scan.

[COMPATIBILITY-RECOVERY-eed4df57.json](COMPATIBILITY-RECOVERY-eed4df57.json) binds the
188-check retained-trusted compatibility review and the approved CPU recovery.
The separate `RECOVERY-FINAL.json` establishes completed qualification while
preserving the host failure. No GPU rerun, phase/sample splicing, or relabelling of
historical execution occurred. Reference and candidate source are unchanged:
**no optimization gain is accepted or eligible**. This does not qualify other tasks,
full-model serving, E2E speedup, global GPU idleness, or arbitrary later checkouts.

Only the body of `_lean_attention_decode_kernel` in
[`source/decode_attention.py`](../../../tasks/headkernel/kimi-k3__lean_attention_decode/source/decode_attention.py)
is the declared editable kernel. Its surrounding launch, ABI, layout, validation,
fixture and timing contracts remain protected. Changed candidates require a new
complete current trusted/framework evaluation and quality review.

## Preparation and evidence retrieval

Follow the [portable runtime setup guide](../../../docs/how-to/headkernel-upstream-runtime.md),
selecting only `headkernel/kimi-k3__lean_attention_decode` at the exact reviewed
`eed4df57` checkout. The image source of truth is `images.sglang_v0520.pull_reference`
in the [runtime catalog](../../../tools/headkernel-runtime-targets.json), on Linux
amd64 and MI355X/gfx950. Preparation is not another qualification.

The [external fixture manifest](../../../tasks/headkernel/kimi-k3__lean_attention_decode/fixtures/EXTERNAL-MANIFEST.json)
pins **15 files / 632,810,514 bytes**, including 14 tensor-storage blobs, under:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261005/kimi-actual-v4-no-stack-194550/
```

Tensor bodies remain at their original pinned objects. The evidence archive
contains their observed before/after hashes and small metadata, not the tensors
or model weights. No credentials or authentication configuration are included.

The immutable evidence prefix is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/kimi-k3__lean_attention_decode/eed4df57-fw209907-trusted20c1a949-200712-cpu-recovery/
```

| Object | SHA-256 |
| --- | --- |
| `kimi-lean-eed4df57-fw209907-retained20c1a949-cpu-recovery.tar.gz` | `345b4afca29befa36727210b79bca5318eae840bbfaa4b19dd0bb188ce038b7a` |
| `MANIFEST.json` | `54049f8db36ff01b8ed6c12a6c2075978359a0e6b42193b8c61b3a17df14027b` |
| `RECOVERY-FINAL.json` | `19c43019e72f1e43d4a14bf53d5e56eb81b2c3339f3868713b75d0089b1355eb` |
| `COMPLETE.json` | `eed91c5221863164363485f228989bd5e0b97b22e2e45bc01cc1bf823237c6d5` |

**All eight remote objects and all 384 payload files passed full streamed
readback.** The archive includes the complete retained trusted/control records,
current reports and input evidence, original host failure, independent review,
CPU verifier sources, and recovery result. [READY.json](READY.json) records the
full publication and execution identities.

With rclone configured for OCI access:

```bash
export GOMAXPROCS=1
LEAN_EVIDENCE=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/kimi-k3__lean_attention_decode/eed4df57-fw209907-trusted20c1a949-200712-cpu-recovery
rclone copy "$LEAN_EVIDENCE" lean-evidence \
  --transfers 64000 --progress --buffer-size 0
(cd lean-evidence && sha256sum --check SHA256SUMS)
```

Verify `COMPLETE.json` and its readback-receipt hashes, then every archive member
against `MANIFEST.json`. Original absolute paths in receipts are provenance;
retrieval does not require the original node or shared filesystem.
