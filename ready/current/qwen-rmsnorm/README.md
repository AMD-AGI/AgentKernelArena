# Current qualification: Qwen fused add + RMSNorm

The unchanged two-case task is **ready within its reviewed scope at `18bb4d67bea577f6951f51a95cf524c9c425f952`**. The [explicit compatibility review](COMPATIBILITY-18bb4d67.json) combines the fresh framework result with separately retained trusted evidence; it does not relabel either execution.

| Evidence | Original execution | Completed scope |
| --- | --- | --- |
| Framework | `18bb4d67`, job **209533** | All **12 checks**, two cases, **200 ordered samples** and 236 checked invocations; complete post-validator integrity chain and all 44 final workspace entries verified. |
| Trusted comparison | `294e65a0`, job **203064** | All **six phases**, **400 samples** and **72 compiled controls**. The numerical/timing source review approves retaining these measurements for the unchanged task. |

The new framework run and owned-container retirement completed before `FINAL` at **2026-10-12 15:09:40 UTC**. The source observation covered 43 of 44 entries: the generated `scripts/_aka_benchmark.py` is absent from the source tree by design and was present and hash-verified in the final workspace. Both report/marker/integrity records and the external attempt records passed the current read-only completion checker. The earlier `31d7bdde` framework pass retains its final input-observation gap, and `8956e210` retains its official report-format FAIL; neither is promoted by this new result.

**No optimization gain is accepted.** No new trusted comparison or SG0.5.20 model-serving capture is claimed. The review covers BF16 shapes **8192×8192** and **64×8192**, epsilon `1e-6`, tolerance `0.02`, 10 warmups and 100 samples per case. Every replay uses fresh CPU-generated inputs and CPU-computed expected outputs; device timing uses GPU graph replay without event fallback. No external tensor fixtures or model weights are required for this task.

The current task tree is `9e02b0ab5b238e8e288c05ffb958866e45e78a94`. Edit only the body of `_gemma_fused_add_rmsnorm_kernel` in `source/minimax_m3_rmsnorm.py`, preserving its signature and decorators. The host helper, public launcher, imports and other code remain frozen under the [task contract](../../../tasks/headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm/config.yaml). A changed candidate requires a new complete current trusted comparison, correctness/source controls, timing-quality review and framework qualification.

This entry records the exact reviewed checkout and archive identities in [READY.json](READY.json). It does not change the [historical guide](../../qwen-rmsnorm/README.md), its strict verifier or input pins. Those historical verifiers continue to check their original snapshots; this entry does not broaden their acceptance. Later source changes or branch tips require their own evidence or compatibility assessment.

## Preparation and evidence retrieval

Use the [portable setup guide](../../../docs/how-to/headkernel-upstream-runtime.md), selecting `headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm` and the exact reviewed commit above. The image source of truth is `images.sglang_v0520.pull_reference` in the [runtime catalog](../../../tools/headkernel-runtime-targets.json); the recorded target is Linux amd64, MI355X/gfx950. Keep generated workspaces and caches on local NVMe. The [agent configuration](../../../example_configs/ready_qwen_rmsnorm_mi355x.yaml) and [validator configuration](../../../example_configs/validate_ready_qwen_rmsnorm_mi355x.yaml) provide the entrypoints. Preparation itself is not a new qualification.

The immutable current evidence prefix is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__gemma_fused_add_rmsnorm/18bb4d67-framework-209533/
```

| Object | SHA-256 |
| --- | --- |
| `rmsnorm-18bb4d67-framework-209533-evidence.tar.gz` | `b79e7b08969c65fa99d787af22f5fdcaa1059969f59110cf08dae05fd79fcd3b` |
| `MANIFEST.json` | `f30f31d2ea307e2aa8d733feb158f5f08e4d6cc6e851de93f609d2daec82638f` |
| `COMPLETE.json` | `7607c2f6440f5eaff56f04e099363da768998c9721f5911781f29a8e093f08cd` |
| Final compatibility review, inside the archive | `8270a6d671e06d675c51d3660387b3b9fa8be0128c5ed429d4d3649d91be8859` |

All **seven remote objects** were streamed back and checked, including **300 regular payload files, two relative symlinks and the archive manifest member**. The archive contains the original mirrored framework/source evidence, selected retained trusted records, final compatibility approval, and exact grant/parent-grant/scheduler records. No credentials, tensor bodies or unrelated caches were included. Pending flags in original preparation records remain chronological; the later compatibility review and [COMPLETE receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__gemma_fused_add_rmsnorm/18bb4d67-framework-209533/COMPLETE.json) establish the completed publication.

With rclone configured for OCI access, download the bounded bundle and check its object hashes:

```bash
export GOMAXPROCS=1
RMS_EVIDENCE=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__gemma_fused_add_rmsnorm/18bb4d67-framework-209533
rclone copy "$RMS_EVIDENCE" rmsnorm-evidence \
  --transfers 64000 --progress --buffer-size 0
(cd rmsnorm-evidence && sha256sum --check SHA256SUMS)
```

Use the `COMPLETE.json` hashes to check the final metadata receipts as well. `MANIFEST.json` binds each archive member. Original filesystem paths in archived receipts describe historical placement; obtaining this bundle does not require the original node or shared storage.

The original trusted archive remains separately available at:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261009/qualification-evidence/qwen3.8-2.4t__gemma_fused_add_rmsnorm/294e65a0-203064-generation0-complete-trusted/
```

Its manifest SHA-256 is `19f89e9b94383c1e449d06bd64103e54ffd28c0abbbe7a54b4254343d65a50cb`. The new compatibility review binds that original `294e65a0` evidence to the `18bb4d67` framework result; it does not splice partial phases or samples across attempts.
