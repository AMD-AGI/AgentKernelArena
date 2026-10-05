"""CPU checks of the trusted fixture data/transfer boundary."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.task_contract import fingerprint
from src.tools import trusted_fixtures as fixtures


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


@pytest.fixture
def fixture_bundle(tmp_path):
    task, mirror, scratch = (tmp_path / name for name in ("task", "mirror", "scratch"))
    for directory in (task / "fixtures", mirror / "capture/blobs", scratch):
        directory.mkdir(parents=True)
    data = b"captured kernel weight bytes"
    blob_sha = hashlib.sha256(data).hexdigest()
    case = {"schema": "served-tensor-fixture-v1", "payload": {"inputs": {"s0": {
        "storage_nbytes": len(data), "segments": [{"blob": "blobs/weight.bin", "bytes": len(data),
        "sha256": blob_sha, "offset_bytes": 0}]}}}}
    case_bytes = encoded(case)
    cases = {"runtime_image": "example/image@sha256:" + "a" * 64, "cases": [{"case_id": "a", "fixture": {
        "path": "fixtures/case.json", "sha256": hashlib.sha256(case_bytes).hexdigest()}}]}
    manifest = {"schema": fixtures.SCHEMA, "case_manifest_fingerprint": fingerprint(cases),
                "runtime_image": cases["runtime_image"], "oci_prefix": "oci:testremote/bucket/pinned-run/",
                "assets": [
                    {"path": "fixtures/case.json", "object_key": "capture/case.json", "bytes": len(case_bytes),
                     "sha256": hashlib.sha256(case_bytes).hexdigest(), "codec": "served-tensor-fixture-v1", "roles": ["case_metadata"]},
                    {"path": "fixtures/blobs/weight.bin", "object_key": "capture/blobs/weight.bin", "bytes": len(data),
                     "sha256": blob_sha, "codec": "raw-storage-segment-v1", "roles": ["kernel_weight"]}]}
    (mirror / "capture/case.json").write_bytes(case_bytes)
    (mirror / "capture/blobs/weight.bin").write_bytes(data)
    (task / "fixtures/EXTERNAL-MANIFEST.json").write_bytes(encoded(manifest))
    return task, mirror, scratch, cases, manifest


def descriptor(bundle, manifest=None, protected=()):
    task, _mirror, _scratch, cases, original = bundle
    (task / "fixtures/EXTERNAL-MANIFEST.json").write_bytes(encoded(original if manifest is None else manifest))
    return fixtures.load_fixture_manifest(task, "fixtures/EXTERNAL-MANIFEST.json", cases, set(protected))


def materialize(bundle, contract=None, **kwargs):
    task, mirror, scratch, cases, _manifest = bundle
    return fixtures.materialize_fixtures(task, contract or descriptor(bundle), cases,
                                        candidate_workspace=task.parent / "candidate", staging=scratch,
                                        local_mirror=mirror, **kwargs)


def test_verified_local_mirror_materializes_captured_kernel_weights(fixture_bundle):
    if shutil.which("rclone") is None:
        pytest.skip("rclone required for artifact copies")
    task, mirror, _scratch, _cases, manifest = fixture_bundle
    (mirror / "unselected-token-strings.txt").write_text("unselected private content")
    receipt = materialize(fixture_bundle)
    assert receipt["source"] == "verified_local_mirror"
    assert receipt["closure"] == {"case_fixtures": 1, "storage_blobs": 1}
    assert receipt["files"] == 2
    for asset in manifest["assets"]:
        assert fixtures.file_digest(task / asset["path"], asset["bytes"]) == asset["sha256"]
    assert not (task / "unselected-token-strings.txt").exists()


@pytest.mark.parametrize("attack", ["corrupt", "missing", "symlink", "parent_symlink"])
def test_local_mirror_corruption_missing_and_links_stop_before_copy(fixture_bundle, monkeypatch, attack):
    _task, mirror, _scratch, _cases, _manifest = fixture_bundle
    blob = mirror / "capture/blobs/weight.bin"
    if attack == "corrupt":
        blob.write_bytes(b"x" * blob.stat().st_size)
    elif attack == "missing":
        blob.unlink()
    elif attack == "symlink":
        blob.unlink()
        blob.symlink_to("../case.json")
    else:
        blob.parent.rename(mirror / "elsewhere")
        (mirror / "capture/blobs").symlink_to(mirror / "elsewhere")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unverified data reached rclone"))
    with pytest.raises((ValueError, OSError)):
        materialize(fixture_bundle)


@pytest.mark.parametrize("path", ["../escape.bin", "/tmp/escape.bin", "fixtures/../escape.bin", "fixtures//x.bin",
                                  "fixtures/./x.bin", "fixtures/x\ny.bin", "fixtures/x\\y.bin", "."])
def test_destination_traversal_rejected(fixture_bundle, path):
    manifest = copy.deepcopy(fixture_bundle[-1])
    manifest["assets"][1]["path"] = path
    with pytest.raises(ValueError):
        descriptor(fixture_bundle, manifest)


@pytest.mark.parametrize("attack", ["source", "manifest", "duplicate", "parent_overlap", "object_overlap", "existing"])
def test_protected_source_and_destination_overlap_rejected(fixture_bundle, attack):
    task = fixture_bundle[0]
    manifest = copy.deepcopy(fixture_bundle[-1])
    protected = {"fixtures/editable.bin", "source/kernel.py", "config.yaml"}
    if attack == "source":
        manifest["assets"][1]["path"] = "fixtures/editable.bin"
    elif attack == "manifest":
        manifest["assets"][0]["path"] = "fixtures/EXTERNAL-MANIFEST.json"
    elif attack == "duplicate":
        manifest["assets"].append(dict(manifest["assets"][1]))
    elif attack == "parent_overlap":
        manifest["assets"][1]["path"] = "fixtures/case.json/blob.bin"
    elif attack == "object_overlap":
        manifest["assets"][1]["object_key"] = "capture/case.json/blob.bin"
    else:
        (task / "fixtures/case.json").write_text("already committed")
    with pytest.raises(ValueError, match="overlap|duplicate|already exists"):
        descriptor(fixture_bundle, manifest, protected)


@pytest.mark.parametrize("field,value", [("case_manifest_fingerprint", "b" * 64), ("runtime_image", "wrong/image"),
                                         ("oci_prefix", "oci:testremote/bucket/../escape/"),
                                         ("oci_prefix", "https://host/?secret=yes")])
def test_case_image_and_oci_pins_rejected(fixture_bundle, field, value):
    manifest = copy.deepcopy(fixture_bundle[-1])
    manifest[field] = value
    with pytest.raises(ValueError):
        descriptor(fixture_bundle, manifest)


@pytest.mark.parametrize("field,value", [("codec", "pickle"), ("roles", ["whole_model_checkpoint"]),
                                         ("roles", ["case_metadata", "kernel_weight"]),
                                         ("bytes", True), ("sha256", "not-a-hash")])
def test_unsupported_codecs_roles_and_inexact_pins_rejected(fixture_bundle, field, value):
    manifest = copy.deepcopy(fixture_bundle[-1])
    manifest["assets"][1][field] = value
    with pytest.raises(ValueError):
        descriptor(fixture_bundle, manifest)


def test_duplicate_manifest_keys_and_oversized_json_rejected(fixture_bundle):
    task, _mirror, _scratch, cases, original = fixture_bundle
    manifest = copy.deepcopy(original)
    manifest["assets"][0]["bytes"] = (64 << 20) + 1
    with pytest.raises(ValueError, match="metadata limit"):
        descriptor(fixture_bundle, manifest)
    (task / "fixtures/EXTERNAL-MANIFEST.json").write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        fixtures.load_fixture_manifest(task, "fixtures/EXTERNAL-MANIFEST.json", cases, set())


def test_limits_and_unapproved_network_rejected_before_rclone(fixture_bundle, monkeypatch):
    task, _mirror, scratch, cases, _manifest = fixture_bundle
    contract = descriptor(fixture_bundle)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("limits checked after transfer"))
    with pytest.raises(ValueError, match="budget"):
        materialize(fixture_bundle, contract, max_bytes=1)
    with pytest.raises(ValueError, match="budget"):
        materialize(fixture_bundle, contract, max_files=1)
    with pytest.raises(ValueError, match="approved"):
        fixtures.materialize_fixtures(task, contract, cases, candidate_workspace=task.parent / "candidate", staging=scratch)
    with pytest.raises(ValueError, match="approval"):
        materialize(fixture_bundle, contract, allowed_oci_prefix="oci:testremote/bucket/another/")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("Disk", (), {"free": 1})())
    with pytest.raises(ValueError, match="scratch"):
        materialize(fixture_bundle, contract)


def test_candidate_owned_mirror_is_rejected(fixture_bundle):
    task, mirror, scratch, cases, _manifest = fixture_bundle
    with pytest.raises(ValueError, match="overlaps candidate"):
        fixtures.materialize_fixtures(task, descriptor(fixture_bundle), cases, candidate_workspace=mirror,
                                      staging=scratch, local_mirror=mirror)


@pytest.mark.parametrize("attack", ["missing_blob", "wrong_hash", "offset", "overlap", "undeclared_metadata"])
def test_segment_reference_closure_rejected(fixture_bundle, attack):
    task, mirror, _scratch, cases, manifest = fixture_bundle
    assets = copy.deepcopy(manifest["assets"])
    # Generate a small synthetic mapped data tree; no benchmark artifact is copied.
    fixture = json.loads((mirror / "capture/case.json").read_text())
    segment = fixture["payload"]["inputs"]["s0"]["segments"][0]
    if attack == "missing_blob":
        assets.pop()
    elif attack == "wrong_hash":
        segment["sha256"] = "b" * 64
    elif attack == "offset":
        segment["offset_bytes"] = 1
    elif attack == "overlap":
        fixture["payload"]["inputs"]["s0"]["segments"].append(dict(segment))
    else:
        cases = copy.deepcopy(cases)
        cases["cases"][0].pop("fixture")
    (task / "fixtures/case.json").write_bytes(encoded(fixture))
    with pytest.raises(ValueError):
        fixtures.verify_closure(task, assets, cases)


def test_remote_copy_flags_and_postdownload_corruption_rejection(fixture_bundle, monkeypatch):
    task, _mirror, scratch, cases, manifest = fixture_bundle
    contract = descriptor(fixture_bundle)
    calls = []

    def fake_rclone(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["GOMAXPROCS"] == "1"
        assert "--progress" in command
        for flag, value in (("--transfers", "64000"), ("--buffer-size", "0"), ("--checkers", "2")):
            assert command[command.index(flag) + 1] == value
        assert kwargs["timeout"] > 0
        target = Path(command[3])
        for asset in manifest["assets"]:
            p = target / asset["object_key"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * asset["bytes"])

    monkeypatch.setattr(subprocess, "run", fake_rclone)
    with pytest.raises(ValueError, match="SHA256"):
        fixtures.materialize_fixtures(task, contract, cases, candidate_workspace=task.parent / "candidate", staging=scratch,
                                      allowed_oci_prefix=manifest["oci_prefix"])
    assert len(calls) == 1
    assert not (task / "fixtures/case.json").exists()


def test_batch_bounds_and_large_single_objects():
    small = [{"object_key": str(i), "bytes": 1} for i in range(51)]
    assert [len(batch) for batch in fixtures.batches(small)] == [24, 24, 3]
    large = [{"object_key": str(i), "bytes": 40 << 20} for i in range(3)]
    assert [len(batch) for batch in fixtures.batches(large)] == [1, 1, 1]
