# minimax-m3__decode_score_kernel

This task optimizes the body of `_decode_score_kernel` in [`source/flash_with_topk_idx.py`](source/flash_with_topk_idx.py) from the exact SGLang image pinned in [`config.yaml`](config.yaml). The AST source guard freezes imports, decorators, signatures, host wrappers, allocation policy, downstream kernels and helpers. The task requires ROCm `gfx950` and the image's complete native AITER JIT cache.

The final package contains **16 structural variants**, including **0 alternate compiler configurations certified by native GPU replay with captured operands**. Full task qualification is pending an authentic framework-finalized `task_validator` PASS. Supplemental correctness receipts establish compiler-configuration parity; they contain no timing or task-qualification claim.

## Served workload and provenance

The source run `minimax-case-only-v3-194550` completed 64 requests with ISL8192, OSL1024, concurrency 64, TP8, context length 9218 and seed 42. All eight ranks have explicit sealed `manifest.profile-stop.json` receipts, despite the stop HTTP route returning 500. No Torch profiler was started and no fresh raw traces are claimed.

Across the three MiniMax tasks, all **80 observed structural variants** and **1,424,072 calls** are retained. Rank0 directly captured 52 native configurations. The other 28 configurations differ only in recorded compiler `num_warps` or `num_stages`; their tensor ABI, launch grid, constexpr values, scalars and complete proportional work distributions match uniquely paired rank0 cases. Actual GPU replay checked all 39 retained states for those variants against captured outputs, original native configurations, independent numerical math and target native configurations at tolerance 0.02.

[`provenance/COVERAGE.json`](provenance/COVERAGE.json) distinguishes direct captures from native replay with captured operands and binds the original all-rank manifests, exact audits, source schemas, original fixture hashes, supplemental reports and full state coverage. The original strict global capture verifier still cannot certify missing exact fixture keys; its code and the source captures were not changed. The supplemented certificate covers all observed variants without claiming that another rank's floating tensors were captured.

[`cases.json`](cases.json) retains every physical tensor shape, stride, storage offset, alias group, original storage capacity, scalar value/type, launch configuration, full work distribution and all-rank occurrence count. Tensor geometry determines physical query rows; a logical CPU batch annotation does not substitute for that geometry. Startup graph buffers and historical captures cannot fill gaps. Every used graph has at least two actual served replay notifications.

## External fixtures

[`fixtures/EXTERNAL-MANIFEST.json`](fixtures/EXTERNAL-MANIFEST.json) pins the portable metadata bundles and unchanged raw tensor blobs by size and SHA256. Each bundle embeds the byte-exact original first/min/max fixture JSON and declares its raw-storage closure. Raw data stays outside Git. A verified local mirror supplies these assets; the committed OCI prefix is reserved and unpublished.

Use the repository [trusted fixture materializer](../../../docs/how-to/trusted-fixture-artifacts.md) with the exact trusted commit, an explicit scratch directory and a verified local mirror. Its `--stage-only` output provides the complete task to framework validation. A Git-only checkout deliberately fails when fixtures are absent.

## Fresh inputs and correctness

The runner reconstructs original storage capacities and aliased physical views. Before every replay it refreshes Q, K, V and sink values. Alternating seeds correlate the standard-normal sink with the first query so sink omission remains detectable at long context. Fresh page translations, request-row translations and legal sparse-block permutations preserve sequence lengths, top-k counts, duplicate multiplicity, sorted right padding, partial last blocks and causal-boundary work.

The independent FP32 block-score reference checks the returned top-k indices against the valid cutoff, including padding and ties. Frozen native parity also checks the exact integer indices and optional-output structure. Floating outputs use the unchanged mixed bound `0.02 * RMS(reference) + 0.02 * abs(reference)` with explicit nonfinite checks. Every checked replay runs both independent math and frozen native parity at the recorded launch configuration.

Pure outputs are poisoned before replay. Immutable CPU snapshots cover all input storage, including padding. Candidate outputs are snapshotted before expected outputs are computed on GPU. Correctness restores every retained actual first/min/max state and checks captured output parity, then checks three fresh seeds `[0, 1, 2]`. No-op and wrong-output controls must fail.

## Evaluation

From a fully materialized task inside the pinned GPU image:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

Compilation executes the original wrapper and captures a real GPU graph for every case. The isolated task module selects the original recorded autotuner configuration and checks actual grid, constexpr, warps and stages. Production SGLang modules are never rebound.

Performance runs **10 warmups and 100 measured graph replays per case**. Every iteration resets inputs, poisons outputs and checks independent math, frozen native parity and input immutability. Reset and validation remain outside synchronized device-event timing. Each timed graph contains one complete original wrapper invocation with its frozen downstream work.

The protected runner uses the canonical `ut/evaluation_contract.py`, accepts a trusted `--request`, and finalizes reports only after exact case coverage and source/package identities are rechecked. Reports live under `build/`. The shared framework scores the arithmetic mean of matched per-case speedup ratios; occurrence counts remain provenance.

Qualification requires positive compile/correctness/performance, actual no-op and wrong-output GPU source rejection through this loader, and a fresh framework-finalized `validation_report.yaml` with `overall_status: PASS`. CPU tests, supplemental receipts and staging receipts do not replace this gate.

## Protected implementation

- `ut/served_contract.py`, `ut/minimax_work.py`: full case and work-distribution validation.
- `ut/minimax_coverage.py`: captured-versus-native-replay coverage proof validation.
- `ut/minimax_fixtures.py`: portable raw fixture loading and original-byte verification.
- `ut/minimax_data.py`: physical storage reconstruction, input refresh and immutability.
- `ut/minimax_reference.py`, `ut/reference/`: independent math and exact frozen native source.
- `ut/minimax_native.py`, `ut/source_guard.py`: runtime source pins, isolated launches and editable-body enforcement.
- `scripts/task_runner.py`: compilation, correctness and checked timing.

The superseded legacy overlays and historical geometry fallbacks were removed. Their tracked history remains provenance and is not used for evaluation.


Fresh replay covers the full recorded sequence-length histogram within every original ABI/compiler case. The original recorded-golden checks and seed-to-recorded-state mapping for seeds 0, 1, 2 and the seed-1000 negative controls remain intact. Correctness additionally checks six numeric ranks in each 128-setting class: 0, 1, 63, 64, 126 and 127. Reports distinguish all represented settings from the settings actually checked.

Performance draws from exact recorded integer occurrence weights with replacement and retains 10 warmups and 100 checked graph samples per case. Reports include all draw IDs, untimed values and the arithmetic mean of all 100 raw device samples. The trusted evaluator supplies the same private seed to reference and candidate; `ut/workload_controls.py:paired_report_costs` additionally validates complete phase requests, raw means and matched schedules for evidence review. This is sampled distribution coverage, not exhaustive numerical execution or an end-to-end gain claim. This harness change requires fresh GPU and framework qualification; existing receipts qualify their original snapshots.
