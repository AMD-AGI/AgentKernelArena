# MiniMax FP4 GEMM — actual capture sealed, qualification pending

The task preserves all 14 native case identities from the completed
`minimax-original-head-v3-194969` workload: 64 requests, ISL8192, OSL1024,
concurrency64, TP8. The exact eight-rank sum is **1,897,872 calls**. The captured
shapes include M1/M64 decode and M8189/M8192/M16365/M16384 prefill. Packed weights
are `[768,3072]` and `[6144,192]`; activation scales retain their actual
column-major strides. The M1/N768 case uses four-way split-K. No case, frequency,
weight tensor, tail shape, or launch setting is inferred from a sampled profile.

`cases.json`, `CAPTURE-ADMISSION.json`, and
`provenance/FULL-WORKLOAD-194969.json` bind the actual source capture. All 74
metadata/blob assets (2,044,395,146 bytes) are content-pinned in
`fixtures/EXTERNAL-MANIFEST.json`. The verified local mirror is the staging
source; OCI publication is still pending. The historical unbuilt marker is
preserved in `provenance/HISTORICAL-NOT-BUILT.txt`.

Only `_gemm_afp4wfp4_kernel` in `source/kernel.py` is editable. Its imports,
decorators, signature, helpers and configuration lookup are frozen. Candidate
and reference use the original wrapper body with a private kernel binding;
only its process-global registration decorator is removed in the loader AST.
Original source bytes and that adaptation are recorded in `SOURCE-PROVENANCE.json`.

The independent CPU oracle decodes low-then-high E2M1 nibbles and E8M0 group32
scales, performs FP64 products, and applies native split-K partial/final casts.
The fixed mixed-RMS limit is 0.02 and never relaxes automatically. Every captured
output must calibrate before candidate acceptance. Input refresh permutes rows
and corresponding scales and changes FP4 signs while preserving layout and
magnitudes. CPU-owned truth, complete input storage, output aliases and padding
are checked on every replay. Outputs are poisoned before execution.

All 14 cases retain 10 warmups and 100 checked graph timing samples. Correctness
also tests eager and graph callbacks plus real source-only no-op/zero-output
submissions on the smallest captured non-split case, each after a valid native
reference. Invalid reference or setup failures never count as rejection. These
additional source controls collect no performance samples and do not replace
any scoreable case.

The shared adapter smoke passed; that result establishes capture mechanics only.
A fresh framework-finalized `validation_report.yaml` with overall PASS and the
actual task's source controls are still required. No task/model optimization
or serving speedup is claimed.

The first task-validator attempt on job194969 exited the compile command after
38.9 seconds at the final package-hash check: the framework's live stderr log
under `.validator_audit` was incorrectly included in that hash. Correctness and
performance were skipped, and the outer launcher later reached its time bound.
The revised hash excludes only top-level `.validator_audit` and
`.validator_torch_extensions` runtime directories, while code, oracle, cases,
fixtures and nested same-name directories remain protected. The preserved
failure and repair are recorded in `provenance/VALIDATOR-HASH-BOUNDARY-194969.json`.
This repair has CPU regression coverage and still requires a fresh GPU pass.

For trusted staging use `src/tools/trusted_task_eval.py --stage-only` with the
committed task and verified fixture mirror. The task runs through the protected
`python3 scripts/task_runner.py compile|correctness|performance` entrypoint.
`scripts/import_capture.py` rebuilds a new dataset from pinned all-rank owner
receipts; it never alters the original capture. `capture/INTEGRATION.json`
documents capture integration and native smoke preparation.
