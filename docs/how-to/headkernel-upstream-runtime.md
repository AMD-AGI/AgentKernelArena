# Portable head-kernel setup and trusted retest

The [runtime catalog](../../tools/headkernel-runtime-targets.json) maps 18 refreshed tasks to one exact SGLang 0.5.20 digest and retains all five Qwen mappings. **Readiness and qualification remain false** until final, source-matched GPU reports pass. Artifact download, CPU staging and a completed capture do not establish task correctness, performance or a serving-runtime gain.

The refreshed recording workload is 64 requests, ISL8192, OSL1024, C64, TP8, context9218 and seed42. Warmup uses OSL16 and a prefix-cache flush. Replay uses the task's captured rank-local operands on a MI355X (`gfx950`), with the work and measurement policy in its local case manifest. Full model weights and the original serving cluster are not required for isolated replay.

## Prepare trusted inputs before launching an agent

On a trusted ROCm host, install Git, Python3 with PyYAML, rclone and Docker. Configure rclone's `oci` remote to access the exact prefix in each selected task's `fixtures/EXTERNAL-MANIFEST.json`. This is an OCI object-store remote, distinct from the Docker image registry. Credentials stay on the host. Choose a trusted full Git commit containing the task, case manifest, external-asset manifest and setup helper.

The following creates a separate clean Arena checkout and materializes the exact task package with the existing trusted fixture materializer. It preserves the original Git task tree, checks that tracked task content is unchanged, retains staging receipts and creates a standard validator config. The helper chooses file/byte limits from the committed manifest; DeepSeek decode has more than the default16,384 assets.

```bash
TRUSTED_COMMIT=$(git rev-parse HEAD)
PREPARED_ROOT="$PWD/../aka-headkernel-prepared"
SCRATCH_ROOT="$PWD/../aka-headkernel-scratch"
python3 tools/prepare_headkernel_run.py   --repo . --commit "$TRUSTED_COMMIT"   --output "$PREPARED_ROOT" --scratch-dir "$SCRATCH_ROOT"   --task headkernel/kimi-k3__dense_bf16_gemm_cijk   --use-manifest-oci-prefixes
```

Repeat `--task` to select several refreshed tasks, or omit it to prepare all18. For one task with an existing trusted local mirror, replace `--use-manifest-oci-prefixes` with `--fixture-local-mirror /absolute/mirror`; arrange it as `MIRROR/object_key` using that task's manifest. Prefixes are task-specific and may share a published capture bundle. The materializer requires every declared object and exact SHA256/size match; an unavailable or incomplete publication stops preparation. The new supplemental manifests do not inherit publication or qualification status from older captures.

Preparation uses `src/tools/trusted_task_eval.py --stage-only`, which extracts task files from the selected Git commit, stages data without executing an importer, validates the case/segment reference closure and emits receipts. Downloaded fixtures are installed before framework workspace copying. See [the fixture contract](trusted-fixture-artifacts.md) for the supported formats, host budgets and receipt semantics. The native DeepSeek FP8 quantization task has declared generated cases and requires no external fixture manifest.

## Run the standard Arena validator

Read `PREPARATION.json` and the per-task staging receipts. Configure the chosen validator backend and its ordinary authentication in `arena/example_configs/prepared_headkernel_run.yaml` and the normal agent configuration. Run from the prepared checkout:

```bash
cd "$PREPARED_ROOT/arena"
export AKA_DOCKER_IMAGE=$(python3 -c 'import json; print(json.load(open("tools/headkernel-runtime-targets.json"))["images"]["sglang_v0520"]["pull_reference"])')
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
make docker-smoke
make docker-check-agents CONFIG=example_configs/prepared_headkernel_run.yaml
make docker-run CONFIG=example_configs/prepared_headkernel_run.yaml
```

The image override is an exact digest. `AKA_AITER_JIT_SOURCE` asks the standard Docker runner to copy and verify the image's complete native JIT cache into a private writable cache. It preserves image-precompiled entries and avoids root-owned cache and CSV-write failures; it does not use another campaign's cache. The refreshed task sources and declared fixtures contain their replay requirements. No shared model tree, external architecture patch, cluster lock script, Spur or Slurm command is part of this route.

Inspect the **framework-finalized** `validation_report.yaml` and completion marker for the exact prepared package. Require `overall_status: PASS`; a partial report, WARN, stale report or direct UT success is not a replacement. Standard task protection preserves fixture and harness files. For optimization, use the same prepared task package with an ordinary agent run config and change only the agent selection; do not regenerate or substitute fixture data.

## Retest only the submitted source

After the optimization worker stops, run the trusted evaluator from the trusted source checkout, outside the candidate workspace. It extracts the reference and harness again from Git, rematerializes the declared fixtures, admits only configured editable source files and runs fresh comparison phases. For example:

```bash
TASK_PATH=tasks/headkernel/kimi-k3__dense_bf16_gemm_cijk
FIXTURE_PREFIX=$(git show "$TRUSTED_COMMIT:$TASK_PATH/fixtures/EXTERNAL-MANIFEST.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["oci_prefix"])')
python3 src/tools/trusted_task_eval.py   --repo . --commit "$TRUSTED_COMMIT" --task "$TASK_PATH"   --candidate-workspace /absolute/path/to/stopped-agent-workspace   --render-device /dev/dri/renderD128   --output /absolute/path/to/new-trusted-results   --scratch-dir "$SCRATCH_ROOT"   --fixture-oci-prefix "$FIXTURE_PREFIX" --fixture-max-files 32768
```

Select the available render device. A verified local mirror can replace the OCI argument. The evaluator uses the task's committed image and source/fixture contracts; its containers have no network or credential mounts. It retains diagnostics and binds materialization receipts into `trusted_measurement.json`. Keep this receipt with the framework report and source-only negative-control evidence. The dedicated [native quant retest](../../src/tools/trusted_native_eval.py) remains the supported route for `headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8`; its task README gives the explicit single-source command.

## Interpret fixture and result scope

The Kimi residual task records the complete native ABI and frequencies but stores only first-`min(4,T)` actual token-row parity samples. Its full-shape fresh-input mathematical checks remain mandatory. Other tasks document captured, generated or supplemental-native cases in their local provenance. Preserve those distinctions, output mutations, tensor aliases, tolerances, warmups and timed boundaries.

The user-provided [results table](../reference/headkernel-reported-results.md) reports historical SGLang0.5.17 numbers. Current setup and capture completion do not replace those figures with new gains. Qwen's five mappings, task files and historical workload remain unchanged; use its task-local instructions and the [historical setup guide](../reference/headkernel-historical-setup.md), not the refreshed SG0.5.20 selection.
