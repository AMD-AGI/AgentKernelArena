# Current qualification: Qwen recurrent gated-delta decode

The original **batch-64 and batch-1 decode cases** are ready within their reviewed scope at **`eed4df573cb0214750c43779f209c663223c2e2f`**. The [final compatibility review](COMPATIBILITY-eed4df57.json) passed **1,334 checks**, preserving two complete executions at their actual revisions:

| Evidence | Actual execution | Completed scope |
| --- | --- | --- |
| Trusted comparison | `eed4df57`, job **209670** | All **six fresh phases**, two cases and **400 ordered reference/candidate samples**. Twelve before/after observations each bind 31 entries; 348 small input files were rehashed. |
| Framework | `18bb4d67`, job **209533** | All **12 checks**, two cases and **200 samples**, with the complete post-validator report/marker/integrity chain, 31 final workspace entries and owned-container retirement. |

The trusted run retired its owned containers at **2026-10-12 16:49:32 UTC**, before `FINAL` at **16:49:33 UTC**. The retained framework completed at **2026-10-12 15:10:25 UTC**, after its own retirement. The `eed4df57` completion checker accepts the unchanged framework evidence. **No framework GPU execution at `eed4df57` is claimed.** Earlier incomplete trusted attempts, including jobs 209533 and 209572, contribute no phases or samples.

Both runs retain task tree `7bc803bae3dbd29f8955406442c92e456085796e` and the same task inputs, timing contract and enforcement core. The intervening changes concern host copy scheduling, native-routing support for other tasks, and documentation. Reference and candidate kernel sources are identical: **no optimization gain is accepted or eligible**. No global GPU-idle, other-task or model-serving qualification is claimed.

The exact cases are `decode_b64_live|decode` and `decode_b1_live|decode`. Preserve the original SG0.5.18 TP8/EP8 fixture and **FP32 recurrent state**, all captured layouts and metadata, output/state-transition checks, 10 warmups and 100 measured GPU graph replays per case and leg. Each case has 118 checked invocations. These fixture contracts do not establish a new SG0.5.20 Qwen serving capture or the later serving recipe's BF16 Mamba state.

Only the body of `fused_recurrent_gated_delta_rule_packed_decode_kernel` in `source/fused_recurrent.py` is editable. Its launcher, decorators, signatures, imports, other functions and module code remain frozen under the [task contract](../../../tasks/headkernel/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode/config.yaml). Changed candidates need complete current trusted/framework evaluation and timing-quality review. [READY.json](READY.json) records exact source, execution and fixture identities. This entry does not broaden any historical verifier's input pins or automatically qualify later checkouts.

## Preparation and evidence retrieval

Use the [portable setup guide](../../../docs/how-to/headkernel-upstream-runtime.md), selecting only `headkernel/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode` at the exact reviewed `eed4df57` checkout. The `headkernel_artifacts_v1` route verifies the original fixture through [the artifact registry](../../../tools/headkernel-artifacts.json). The image source of truth is `images.sglang_v0520.pull_reference` in the [runtime catalog](../../../tools/headkernel-runtime-targets.json); the recorded platform is Linux amd64, MI355X/gfx950. Preparation itself is not another qualification.

The external fixture is **1,044,479,494 bytes**, SHA256 `38a624a5f502279a13a09c012fababbe232cadedacd560056d8411640a04aaf1`:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_ut_0914_full/20260922/tensors/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode/ut/reference_io.pt
```

The evidence archive retains its identities and runtime before/after hash observations, not the tensor body. Model weights and a serving process are not required for isolated task preparation.

The immutable combined evidence prefix is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode/eed4df57-trusted-209670-fw18bb4d67-209533/
```

| Object | SHA-256 |
| --- | --- |
| `qwen-recurrent-eed4df57-trusted-209670-fw18bb4d67-209533.tar.gz` | `5db3b76a060d1ceb574a72c52b9af335d671b3a239e12d9905c42d32fb1addad` |
| `MANIFEST.json` | `f0e89e31f31c5a2ab348b3dabf847ac4a7d30c002c5179ed91502e6efbb3f801` |
| `COMPLETE.json` | `1717a4a5333c512fae3abd65f68ff424f7e00d529778aa8d3184a55937f23887` |
| Final compatibility review inside the archive | `9a02d998d374b99d0835aba48e012521808c34d104a789e0f7f20a1030fae951` |

All **seven remote objects** passed streamed readback, including **680 regular payload files, 15 relative symlinks and the manifest member**. The archive preserves all completed trusted phases and raw series, retained framework evidence, input/source observations, retirement records and the final independent review. Original pending flags remain chronological evidence; the final review and [COMPLETE receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode/eed4df57-trusted-209670-fw18bb4d67-209533/COMPLETE.json) establish the completed publication.

With rclone configured for OCI access:

```bash
export GOMAXPROCS=1
RECURRENT_EVIDENCE=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode/eed4df57-trusted-209670-fw18bb4d67-209533
rclone copy "$RECURRENT_EVIDENCE" recurrent-evidence \
  --transfers 64000 --progress --buffer-size 0
(cd recurrent-evidence && sha256sum --check SHA256SUMS)
```

Check `COMPLETE.json` against the table, use its hashes for the final readback receipt, and verify archive members against `MANIFEST.json`. Downloading the evidence does not require the original node or shared filesystem. No credentials, fixture tensor bodies or unrelated cache contents were included.
