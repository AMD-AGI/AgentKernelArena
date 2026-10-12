# Qwen Dense GEMM scoped qualification

The five original BF16 GEMM cases have complete framework and trusted evidence at
`eed4df573cb0214750c43779f209c663223c2e2f`. The retained framework run on **job 209670/GPU 3** passed all **12 checks**,
500 ordered samples, source/harness integrity and owned-container retirement before
FINAL at **2026-10-12 16:50:39 UTC**. The fresh trusted run on **job 210191/GPU 2** completed
all **six phases**, 1,000 ordered samples and retirement before FINAL at
**2026-10-12 23:34:02 UTC**. No partial attempt contributes phases or samples.

[READY.json](READY.json) pins both executions and immutable publication objects.
[COMPATIBILITY-230de1f5.json](COMPATIBILITY-230de1f5.json) verifies that the complete
task tree and relevant timing, numerical, quality, guard, integrity and trusted-runtime
code remain unchanged at `230de1f5897296ddd364d8811bd9fd6a83dbe602`. This documentation base is not a new GPU
execution identity. Later task/runtime changes need new evidence or explicit review.

| Case | M | N | K |
| --- | ---: | ---: | ---: |
| prefill_m16384_n4608_k8192 | 16384 | 4608 | 8192 |
| decode_m64_n32_k8192 | 64 | 32 | 8192 |
| decode_m64_n512_k8192 | 64 | 512 | 8192 |
| decode_m64_n4608_k8192 | 64 | 4608 | 8192 |
| decode_m64_n8192_k256 | 64 | 8192 | 256 |

Every performance series retains **10 warmups and 100 samples**, **114 checked
invocations** and **116 fresh input sets**. The declared sources, seeded input
preparations, shapes, strides, dtype, native dispatch rows and tolerances are unchanged.
The protected source is a literal routing policy parsed without executing editable
Python. This scope covers native backend selection, not arbitrary device-source edits.
The reference and candidate policies are identical: **no optimization gain is accepted**.

The [task contract](../../../tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster/config.yaml) and
[task guide](../../../tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster/README.md) define the complete protocol. Dense is
explicitly fixture-free: the protected generators prepare the original case shapes,
and the task requires no tensor fixture download. Use a fresh authorized MI355X allocation,
exact source revision and the digest-pinned runtime:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

The independent audit replayed the original raw-payload validators for all six
trusted phases, verified 26 before/after input entries per phase and 217 source-verifier
entries, recomputed current quality, checked both FINAL chains, and re-ran the full
join after safe archive extraction. Both results retain their original execution IDs;
owned retirement is not a claim about current global GPU availability.

The [OCI readback receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__dense_bf16_gemm_cluster/eed4df57-framework-209670__trusted-210191/OCI-VERIFIED.json) pins seven evidence objects.
That receipt was itself uploaded and read back, giving **eight verified objects**.
All **849 archive members**—655 regular files, 17 contained relative symlinks and 177 directories—
passed full remote byte/hash/mode/link verification. The archive includes both complete
runs and the qualified join; earlier interrupted comparisons are excluded.

Historical capture/profiler provenance remains SG518. This is not a new SG520 serving
capture, full-model qualification, or an accepted serving/E2E gain. Changed candidates
must pass the full original correctness, source-control, timing-quality and review gates.
