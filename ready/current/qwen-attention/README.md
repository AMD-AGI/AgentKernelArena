# Current qualification: Qwen paged-attention decode

The single declared case **`decode_m64_live|decode`** is ready within its reviewed scope at **`18bb4d67bea577f6951f51a95cf524c9c425f952`**. The [final compatibility review](COMPATIBILITY-18bb4d67.json) passed 644 checks and connects two complete executions without relabeling either one:

| Evidence | Actual execution | Completed scope |
| --- | --- | --- |
| Trusted comparison | `18bb4d67`, job **209572** | All **six fresh phases**, one case, **100 reference + 100 candidate samples**, 10 warmups per measured leg and 118 checked invocations per leg. Twelve before/after input observations each bound 37 entries. |
| Framework | `8956e210`, job **208015** | All **12 checks**, one case, **100 samples**, 118 checked invocations, post-validator integrity and all 37 final workspace entries. The current completion checker accepts the retained report/marker/integrity chain. |

Owned trusted containers retired at **2026-10-12 15:42:09 UTC**, before `FINAL` at **15:42:10 UTC** and the reported preemption at 15:42:20 UTC. The retained framework completed on **2026-10-11 at 17:29:54 UTC**. **No new framework GPU execution at `18bb4d67` is claimed.** The earlier incomplete trusted attempt on job 209533 contributes no phases or samples.

Reference and candidate source bytes are identical. **No optimization gain is accepted or eligible.** The independent review accepted this unchanged-source qualification control after checking all six phase device bindings and the raw timing quality. It does not categorically exclude CPU/shared-system interference, establish global GPU idleness, qualify the concurrent MoE attempt, or extend to other tasks.

The task preserves the original BF16 **M64, page-1, NHD ragged-KV** contract and 19-argument ABI, including partition size 256. Fresh queries and independent CPU softmax references exercise the measured GPU graph while preserving captured KV/routing data. Broader shapes, different fixtures and a full SG0.5.20 Qwen serving capture are outside scope.

Only the bodies of `_paged_attention_kernel` and `_paged_attention_ll4mi_reduce_kernel` in `source/pa_kernels.cuh` are editable. Preserve declarations, signatures, host launch code, Python ABI wrapper, oracle aliases, source controls and the [task contract](../../../tasks/headkernel/qwen3.8-2.4t__paged_attention_decode/config.yaml). Changed candidates require complete current trusted/framework evaluation and timing-quality review. [READY.json](READY.json) records the exact source, execution, fixture and publication identities. This documentation does not extend historical verifier pin sets or automatically cover later branch tips.

## Preparation and evidence retrieval

Use the [portable setup guide](../../../docs/how-to/headkernel-upstream-runtime.md), selecting only `headkernel/qwen3.8-2.4t__paged_attention_decode` at the exact reviewed `18bb4d67` checkout. Its `headkernel_artifacts_v1` route verifies the pinned original fixture through [the artifact registry](../../../tools/headkernel-artifacts.json). The image source of truth is `images.sglang_v0520.pull_reference` in the [runtime catalog](../../../tools/headkernel-runtime-targets.json); the recorded platform is Linux amd64, MI355X/gfx950. Preparation itself does not create a new qualification.

The one external fixture is **536,480,065 bytes**, SHA256 `9c40d52c145027f8affca9e41bbf1100189fde724d42d085ac6296d57020ac56`:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_ut_0914_full/20260922/tensors/qwen3.8-2.4t__paged_attention_decode/ut/reference_io.pt
```

Its historical SG0.5.18 TP8/EP8 page-1 provenance is unchanged. The fixture body is not duplicated in the evidence archive; model weights and a serving process are not required for isolated task preparation.

The immutable combined evidence prefix is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__paged_attention_decode/18bb4d67-trusted-209572-fw8956e210-208015/
```

| Object | SHA-256 |
| --- | --- |
| `qwen-attention-18bb4d67-209572-fw8956e210-208015.tar.gz` | `3ec8176964863adbbe66f41fa571436d3583ef38384fc04f3c27f043a91a199f` |
| `MANIFEST.json` | `c936d7ee5c74ee2191d3c5207a58fc6367bcce3945b6afaf24240860b782b0a0` |
| `COMPLETE.json` | `1fffb085ccdba9ddd0304ef4d817df0e089d4085374827f00f31161d4ccd84aa` |
| Final compatibility review inside the archive | `eb172f6acbeaeb66b621006a6eabb3a4da1e9d6e134152f49cb3f77499af01de` |

All **seven remote objects** passed streamed readback, including **786 regular payload files, 15 relative symlinks and the manifest member**. The archive preserves complete selected trusted evidence and retained framework evidence, their input observations and retirement records, and the final compatibility/isolation review. Earlier pending review flags remain chronological records; the final review and [COMPLETE receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__paged_attention_decode/18bb4d67-trusted-209572-fw8956e210-208015/COMPLETE.json) establish the finished publication.

With rclone configured for OCI access:

```bash
export GOMAXPROCS=1
ATTENTION_EVIDENCE=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__paged_attention_decode/18bb4d67-trusted-209572-fw8956e210-208015
rclone copy "$ATTENTION_EVIDENCE" attention-evidence \
  --transfers 64000 --progress --buffer-size 0
(cd attention-evidence && sha256sum --check SHA256SUMS)
```

Check `COMPLETE.json` against the table and use its object hashes for the final readback receipt; verify archive members against `MANIFEST.json`. Downloading these objects requires no original node or shared storage. The [original framework-only publication](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__paged_attention_decode/8956e210-framework-208015/ARCHIVE-VERIFIED.json) remains immutable and retains its `8956e210` identity.
