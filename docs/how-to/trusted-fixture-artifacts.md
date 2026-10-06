# Trusted captured-fixture artifacts

The generic [trusted host evaluator](../../src/tools/trusted_task_eval.py) can
materialize captured kernel fixtures before it copies the reference task into
the candidate leg. It accepts a manifest from the exact trusted Git commit and
data from either a verified local mirror or a host-approved OCI prefix. The
candidate workspace still supplies only the declared source files. Evaluation
containers retain their read-only task/image mounts, disabled network, and no
credential mounts. The dedicated native-quant evaluator is unchanged.

When a task requests a complete AITER JIT cache, each native evaluation phase
points `FLYDSL_RUNTIME_CACHE_DIR` at that private copy's `flydsl_cache` directory.
This preserves the copied compiled entries and lets FlyDSL create lock files
without writing to the image's root-owned cache.

This extends the [guarded case contract](guarded-case-contract.md). It does not
execute a task's importer, regenerate operands, change cases, qualify a capture,
or publish artifacts. Captured kernel weights, scales, activations, controls,
and expected output tensors are supported benchmark data. Whole model
checkpoints, raw token strings, and credentials are outside this data contract.
Publication remains a separate operation after task qualification.

## Commit the manifest and case references

The task owner commits this optional descriptor in `config.yaml`:

```yaml
trusted_evaluation:
  schema_version: 1
  fixture_manifest: fixtures/EXTERNAL-MANIFEST.json
```

`EXTERNAL-MANIFEST.json` must be a regular Git-tracked file. It uses this shape
(the uppercase SHA placeholders must be replaced with real hashes):

```json
{
  "schema": "trusted-external-fixtures-v1",
  "case_manifest_fingerprint": "CANONICAL_CASES_SHA256",
  "runtime_image": "REGISTRY/IMAGE@sha256:IMAGE_DIGEST",
  "oci_prefix": "oci:REMOTE/BUCKET/VERSION/",
  "assets": [
    {
      "path": "fixtures/case.json",
      "object_key": "capture/case.json",
      "bytes": 1234,
      "sha256": "CASE_JSON_SHA256",
      "codec": "served-tensor-fixture-v1",
      "roles": ["case_metadata"]
    },
    {
      "path": "fixtures/blobs/weight.bin",
      "object_key": "capture/blobs/weight.bin",
      "bytes": 8192,
      "sha256": "RAW_STORAGE_SHA256",
      "codec": "raw-storage-segment-v1",
      "roles": ["kernel_weight"]
    }
  ]
}
```

`case_manifest_fingerprint` is `src.task_contract.fingerprint` of the parsed
case manifest, not the SHA of its formatted JSON file. `runtime_image` must
match that case manifest and the task configuration. The OCI prefix is pinned
in Git, including its trailing slash. A local-mirror run still records the
planned or published prefix but does not contact OCI.

Each asset has a canonical destination beneath `fixtures/`, an object key
relative to the OCI prefix, exact nonnegative integer byte count, lowercase
SHA-256, codec, and roles. Object keys and destination paths must each be
unique and cannot overlap as files/directories. Destinations may not overwrite
any Git file, the artifact manifest, editable sources, or protected harness
files. Keep the external case JSON and raw blobs out of the source commit;
commit their exact manifest entries and case references instead. Existing
Git-only tasks without this descriptor keep their current behavior.

The supported codecs are `served-tensor-fixture-v1` for `.json` case metadata
and `raw-storage-segment-v1` for `.bin` storage segments. Metadata must have only
the `case_metadata` role. Storage roles are `activation`, `runtime_control`,
`oracle_output`, `kernel_weight`, and `kernel_weight_scale`; a shared blob may
have multiple roles. No archive, pickle, model-checkpoint format, executable
importer, or implicit directory download is accepted.

Cases refer to their data using either `fixture` or `live_fixture`:

```json
"fixture": {
  "path": "fixtures/case.json",
  "sha256": "CASE_JSON_SHA256"
}
```

Every selected case JSON must be referenced by at least one case. Its existing
`payload[phase][alias].segments` entries retain the original `blob`, `sha256`,
`bytes`, and `offset_bytes`. Blob references resolve relative to that case
JSON's directory. Every segment must be declared in the asset manifest with
matching size/SHA. Offsets must be nonnegative integers, fit within the
original `storage_nbytes`, and not overlap within an alias group. Every declared
blob must be referenced. The materializer preserves original bytes; the
protected task codec remains responsible for interpreting tensor shapes,
strides, aliases, attributes, and controls and performing meaningful GPU checks.

## Materialize on the trusted host

For a local mirror, arrange files as `MIRROR/object_key`. Object keys may differ
from task-relative destinations, allowing a frozen capture directory to serve
as the mirror without another preparatory copy. The mirror must be a canonical
directory outside the candidate workspace and must not contain that workspace.
Every mirror asset is hashed before copying and checked again afterward.

```bash
python3 src/tools/trusted_task_eval.py \
  --repo TRUSTED_CHECKOUT --commit FULL_TRUSTED_COMMIT \
  --task tasks/CATEGORY/TASK --candidate-workspace AGENT_WORKSPACE \
  --render-device /dev/dri/renderD128 --output NEW_RESULT_DIRECTORY \
  --scratch-dir LARGE_LOCAL_SCRATCH \
  --fixture-local-mirror TRUSTED_MIRROR
```

To download from OCI, omit the mirror and pass `--fixture-oci-prefix` with the
exact prefix committed in the manifest. The explicit host prefix is mandatory
for a download and cannot silently redirect an asset to another remote.
Credentials remain in the trusted host's rclone configuration. OCI support
adds no network or credentials to the evaluation container.

External fixtures require explicit `--scratch-dir`. Host defaults allow at
most 16,384 files and 64 GiB of declared asset bytes; adjust these with
`--fixture-max-files` and `--fixture-max-bytes` for an approved workload. The
materializer checks available space for the transaction, reference, and
candidate copies before transferring. It streams large tensor hashes rather
than loading tensors into host memory.

After GPU identity preflight, the isolated Python launcher adds only the selected
script's canonical parent directory to its import path. This gives protected
task scripts their normal sibling imports, including `production_comparison`,
while `-I` continues to exclude ambient `PYTHONPATH` and user-site packages.
The task and image mounts retain their existing read-only boundaries.

Transfers use `rclone --transfers 64000 --progress --buffer-size 0`,
`GOMAXPROCS=1`, two checkers, disabled multithread streams, and explicit lists of
at most 24 objects. Batches target at most 64 MiB; a larger individual asset is
copied alone. Local copies disable external rclone configuration. Each command
has a timeout. Transfer output is suppressed to keep host credentials and
remote diagnostics out of task artifacts; errors stop staging.

Only after all objects pass SHA/size checks and complete reference-closure
validation are they installed into the private extracted task. The evaluator
then copies the complete reference tree, admits candidate source files, runs
source guards, selects a GPU, and computes both package fingerprints. Those
fingerprints cover the materialized fixtures. A host `fixtures_receipt.json`
records the trusted commit, external-manifest SHA, cases fingerprint, image,
OCI prefix, exact asset inventory, closure counts, budgets, and batch counts.
`trusted_measurement.json` binds that receipt's SHA. Failed materialization
runs no task phase and emits no successful measurement.

## Prepare the same input for framework validation

A host-only preparation mode exposes the same Git extraction, manifest checks,
materialization, and complete-task copy without candidate edits or GPU work:

```bash
python3 src/tools/trusted_task_eval.py --stage-only \
  --repo TRUSTED_CHECKOUT --commit FULL_TRUSTED_COMMIT \
  --task tasks/CATEGORY/TASK --output NEW_VALIDATOR_STAGE \
  --scratch-dir LARGE_LOCAL_SCRATCH \
  --fixture-local-mirror TRUSTED_MIRROR
```

Use `NEW_VALIDATOR_STAGE/task` as the validator's prepared task input. Preserve
`staging_receipt.json` and `fixtures_receipt.json` with the eventual framework
report. The stage receipt records the complete package fingerprint and states
`staged_not_evaluated`. A validator adapter must consume this prepared tree
rather than recopying the Git-only task and losing its external assets. It must
preserve protected fixture paths and permit edits only to the task's declared
sources. No fixture fetch belongs inside the validator's evaluation container.
A stage receipt does not replace a framework-finalized `PASS` or a trusted GPU
measurement. Task owners should qualify their prepared input with the normal
framework validator and the trusted replay before publication.

## Prepare a normal Arena checkout

For the refreshed head-kernel selection, [prepare_headkernel_run.py](../../tools/prepare_headkernel_run.py) composes this stage-only API with a clean checkout of the trusted Git commit. It installs each complete prepared task before an agent starts, preserves the original Git task trees and per-task receipts, and writes `example_configs/prepared_headkernel_run.yaml`. It then uses the ordinary `make docker-run` workflow. See the [portable setup guide](headkernel-upstream-runtime.md) for the complete staging, standard-validator and source-only trusted-retest sequence. This preparation performs no GPU qualification and makes no ready-suite claim.
