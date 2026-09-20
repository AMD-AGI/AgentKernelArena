"""CPU-only fixture provisioning and declaration regressions (stdlib unittest)."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from src.tools import prepare_head_kernel_artifacts as provision
from head_kernel_test_utils import task_directory


ROOT = Path(__file__).resolve().parents[1]
TASK = "example-model__kernel"
BODY = b"captured reference bytes\x00\xff\n"
DIGEST = hashlib.sha256(BODY).hexdigest()


class ProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suite = self.root / "suite"
        self.mirror = self.root / "mirror"
        self.destination = self.suite / TASK / "ut/reference_io.pt"
        self.source = self.mirror / TASK / "ut/reference_io.pt"
        self.destination.parent.mkdir(parents=True)
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(BODY)
        self.metadata = self.destination.with_name("meta.json")
        self.metadata.write_text(json.dumps({"reference_io_sha256": DIGEST}))
        self.artifact = provision.Artifact(
            TASK, f"{TASK}/ut/reference_io.pt", len(BODY), DIGEST
        )

    def install(self, **kwargs):
        return provision.prepare_artifacts(
            [self.artifact], self.suite, mirror=self.mirror, **kwargs
        )

    def assert_no_partial(self):
        self.assertEqual(list(self.destination.parent.glob(".*.partial")), [])

    def test_installs_verified_read_only_bytes_without_modifying_source(self):
        source_stat = self.source.stat()
        result = self.install()
        self.assertEqual(result, [(self.artifact.path, "installed")])
        self.assertEqual(self.destination.read_bytes(), BODY)
        self.assertEqual(stat.S_IMODE(self.destination.stat().st_mode), 0o444)
        self.assertEqual(self.destination.stat().st_nlink, 1)
        self.assertNotEqual(self.destination.stat().st_ino, self.source.stat().st_ino)
        self.assertEqual(self.source.read_bytes(), BODY)
        self.assertEqual(self.source.stat().st_mtime_ns, source_stat.st_mtime_ns)
        self.assert_no_partial()

    def test_preserves_existing_valid_reference_without_reading_mirror(self):
        self.destination.write_bytes(BODY)
        self.destination.chmod(0o640)
        before = self.destination.stat()
        self.source.unlink()
        self.assertEqual(self.install(), [(self.artifact.path, "verified existing")])
        after = self.destination.stat()
        self.assertEqual(before.st_ino, after.st_ino)
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
        self.assertEqual(before.st_mode, after.st_mode)

    def test_existing_public_source_hardlink_produces_independent_copy(self):
        other_package = self.root / "other-package-reference.pt"
        os.link(self.source, other_package)
        self.source.chmod(0o644)
        before = self.source.stat()
        self.assertEqual(before.st_nlink, 2)
        self.install()
        after = self.source.stat()
        self.assertEqual(provision._identity(before), provision._identity(after))
        self.assertEqual(before.st_mode, after.st_mode)
        self.assertEqual(before.st_uid, after.st_uid)
        self.assertEqual(after.st_nlink, 2)
        self.assertEqual(other_package.read_bytes(), BODY)
        self.assertEqual(self.destination.read_bytes(), BODY)
        self.assertEqual(self.destination.stat().st_nlink, 1)
        self.assertNotEqual(self.destination.stat().st_ino, after.st_ino)
        self.assert_no_partial()

    def test_unknown_existing_reference_is_never_overwritten(self):
        self.destination.write_bytes(b"x" * len(BODY))
        before = self.destination.stat()
        with self.assertRaisesRegex(provision.ArtifactError, "SHA-256 mismatch"):
            self.install()
        self.assertEqual(self.destination.read_bytes(), b"x" * len(BODY))
        self.assertEqual(self.destination.stat().st_ino, before.st_ino)
        self.assert_no_partial()

    def test_rejects_corrupt_source_size_and_digest_without_publication(self):
        for body, message in [(b"wrong size", "size mismatch"), (b"x" * len(BODY), "SHA-256 mismatch")]:
            with self.subTest(message=message):
                self.source.write_bytes(body)
                with self.assertRaisesRegex(provision.ArtifactError, message):
                    self.install()
                self.assertFalse(self.destination.exists())
                self.assert_no_partial()

    def test_rejects_metadata_hash_disagreement_before_copying(self):
        self.metadata.write_text(json.dumps({"reference_io_sha256": "a" * 64}))
        with self.assertRaisesRegex(provision.ArtifactError, "disagrees"):
            self.install()
        self.assertFalse(self.destination.exists())

    def test_verify_requires_no_source_and_does_not_create_missing_fixture(self):
        with self.assertRaisesRegex(provision.ArtifactError, "missing fixture"):
            provision.prepare_artifacts([self.artifact], self.suite, verify_only=True)
        self.assertFalse(self.destination.exists())
        self.destination.write_bytes(BODY)
        self.source.unlink()
        result = provision.prepare_artifacts([self.artifact], self.suite, verify_only=True)
        self.assertEqual(result, [(self.artifact.path, "verified existing")])

    def test_install_requires_explicit_source(self):
        with self.assertRaisesRegex(provision.ArtifactError, "explicit mirror or cache"):
            provision.prepare_artifacts([self.artifact], self.suite)

    def test_content_addressed_cache_installs_into_task_local_path(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / DIGEST).write_bytes(BODY)
        provision.prepare_artifacts([self.artifact], self.suite, cache=cache)
        self.assertEqual(self.destination.read_bytes(), BODY)

    def test_provisioned_task_copy_is_independent_of_mirror_and_original_suite(self):
        self.install()
        copied = self.root / "copied-suite"
        shutil.copytree(self.suite, copied)
        shutil.rmtree(self.mirror)
        shutil.rmtree(self.suite)
        result = provision.prepare_artifacts([self.artifact], copied, verify_only=True)
        self.assertEqual(result, [(self.artifact.path, "verified existing")])
        self.assertEqual((copied / self.artifact.path).read_bytes(), BODY)

    def test_file_is_complete_when_atomically_published(self):
        original_publish = provision._publish_noreplace

        def observe_publication(*args, **kwargs):
            self.assertFalse(self.destination.exists())
            original_publish(*args, **kwargs)
            self.assertEqual(self.destination.read_bytes(), BODY)
            self.assertEqual(stat.S_IMODE(self.destination.stat().st_mode), 0o444)

        with mock.patch.object(provision, "_publish_noreplace", side_effect=observe_publication):
            self.install()

    def test_concurrent_target_creation_is_preserved(self):
        original_publish = provision._publish_noreplace

        def concurrent_creation(*args, **kwargs):
            self.destination.write_bytes(b"concurrently created reference")
            original_publish(*args, **kwargs)

        with mock.patch.object(provision, "_publish_noreplace", side_effect=concurrent_creation):
            with self.assertRaises(FileExistsError):
                self.install()
        self.assertEqual(self.destination.read_bytes(), b"concurrently created reference")
        self.assert_no_partial()

    def test_publication_links_only_the_verified_private_output_inode(self):
        original_link = os.link
        source_inode = self.source.stat().st_ino

        def inspect_link(temporary, name, **kwargs):
            temporary_stat = os.stat(
                temporary, dir_fd=kwargs["src_dir_fd"], follow_symlinks=False
            )
            self.assertNotEqual(temporary_stat.st_ino, source_inode)
            self.assertEqual(temporary_stat.st_nlink, 1)
            self.assertFalse(self.destination.exists())
            original_link(temporary, name, **kwargs)
            self.assertEqual(self.destination.read_bytes(), BODY)
            self.assertEqual(self.destination.stat().st_ino, temporary_stat.st_ino)
            self.assertEqual(self.destination.stat().st_nlink, 2)

        with mock.patch.object(provision.os, "link", side_effect=inspect_link):
            self.install()
        self.assertEqual(self.destination.stat().st_nlink, 1)
        self.assertEqual(self.source.stat().st_nlink, 1)
        self.assert_no_partial()

    def test_interrupted_copy_removes_only_the_temporary_file(self):
        def interrupt(fd, artifact, output_fd=None):
            os.write(output_fd, b"partial bytes")
            raise OSError("simulated I/O failure")

        with mock.patch.object(provision, "_verify_bytes", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.install()
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.source.read_bytes(), BODY)
        self.assert_no_partial()

    def test_source_mutation_during_copy_is_detected(self):
        original_read = os.read
        changed = False

        def mutate_after_read(fd, amount):
            nonlocal changed
            data = original_read(fd, amount)
            if data and not changed:
                changed = True
                before = self.source.stat()
                self.source.write_bytes(b"x" * len(BODY))
                # Some filesystems update timestamps only once per clock tick.
                os.utime(self.source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            return data

        with mock.patch.object(provision.os, "read", side_effect=mutate_after_read):
            with self.assertRaisesRegex(provision.ArtifactError, "changed while reading"):
                self.install()
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

    def test_rejects_symlink_source_file_and_symlink_source_parent(self):
        saved_source = self.root / "saved.pt"
        self.source.rename(saved_source)
        self.source.symlink_to(saved_source)
        with self.assertRaises(OSError):
            self.install()
        self.source.unlink()
        saved_source.rename(self.source)
        saved_parent = self.root / "saved-source-ut"
        self.source.parent.rename(saved_parent)
        self.source.parent.symlink_to(saved_parent, target_is_directory=True)
        with self.assertRaises(OSError):
            self.install()
        self.assertFalse(self.destination.exists())

    def test_rejects_symlink_destination_even_when_target_has_expected_bytes(self):
        self.destination.symlink_to(self.source)
        with self.assertRaises(OSError):
            self.install()
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(self.source.read_bytes(), BODY)

    def test_rejects_dangling_symlink_destination(self):
        self.destination.symlink_to(self.root / "absent.pt")
        with self.assertRaises(OSError):
            self.install()
        self.assertTrue(self.destination.is_symlink())

    def test_rejects_symlink_destination_parent_and_root(self):
        saved_parent = self.root / "saved-ut"
        self.destination.parent.rename(saved_parent)
        self.destination.parent.symlink_to(saved_parent, target_is_directory=True)
        with self.assertRaises(OSError):
            self.install()
        self.assertFalse((saved_parent / "reference_io.pt").exists())
        alias = self.root / "suite-alias"
        alias.symlink_to(self.suite, target_is_directory=True)
        with self.assertRaises(OSError):
            provision.prepare_artifacts([self.artifact], alias, mirror=self.mirror)

    def test_rejects_symlink_metadata(self):
        saved = self.root / "saved-meta.json"
        self.metadata.rename(saved)
        self.metadata.symlink_to(saved)
        with self.assertRaises(OSError):
            self.install()
        self.assertFalse(self.destination.exists())

    def test_rejects_nonregular_files_including_fifo_without_blocking(self):
        self.source.unlink()
        self.source.mkdir()
        with self.assertRaisesRegex(provision.ArtifactError, "regular file"):
            self.install()
        self.source.rmdir()
        os.mkfifo(self.source)
        with self.assertRaisesRegex(provision.ArtifactError, "regular file"):
            self.install()
        self.assertFalse(self.destination.exists())


class NestedProvisioningTests(ProvisioningTests):
    """Run the complete byte/publication safety suite through a v2 declaration."""

    def setUp(self):
        super().setUp()
        nested = "example-model/isl8192_osl1024_conc64_tp8_mi355x/image_v1/kernel"
        task = self.suite / nested
        task.parent.mkdir(parents=True)
        (self.suite / TASK).rename(task)
        self.destination = task / "ut/reference_io.pt"
        self.metadata = self.destination.with_name("meta.json")
        manifest_path = self.root / "artifacts.json"
        manifest_path.write_text(json.dumps({"schema_version": 2, "artifacts": [{
            "task": nested, "operation_id": TASK,
            "path": f"{nested}/ut/reference_io.pt",
            "mirror_path": f"{TASK}/ut/reference_io.pt",
            "size_bytes": len(BODY), "sha256": DIGEST,
        }]}))
        self.manifest = provision.load_manifest(manifest_path)
        self.artifact, = self.manifest.artifacts

    def test_nested_task_selection_uses_the_complete_task_path(self):
        self.assertEqual(provision.select_artifacts(self.manifest, [self.artifact.task]),
                         [self.artifact])
        with self.assertRaisesRegex(provision.ArtifactError, "unknown task"):
            provision.select_artifacts(self.manifest, [TASK])
        with self.assertRaisesRegex(provision.ArtifactError, "unknown task"):
            provision.select_artifacts(self.manifest, ["kernel"])

    def test_nested_destination_ancestor_symlink_is_rejected(self):
        ancestor = self.suite / "example-model"
        saved = self.root / "saved-model"
        ancestor.rename(saved)
        ancestor.symlink_to(saved, target_is_directory=True)
        with self.assertRaises(OSError):
            self.install()
        self.assertEqual(self.source.read_bytes(), BODY)
        self.assertFalse(self.destination.exists())


class ManifestTests(unittest.TestCase):
    def test_committed_inventory_matches_all_task_metadata(self):
        root = ROOT / "tasks/head_kernels"
        manifest = provision.load_manifest(root / "artifacts.json")
        self.assertEqual(len(manifest.artifacts), 17)
        self.assertEqual(len(manifest.tasks), 18)
        self.assertEqual(sum(a.size_bytes for a in manifest.artifacts), 33139788731)
        self.assertEqual(manifest.tasks, {p.parent.relative_to(root).as_posix() for p in root.rglob("config.yaml")})
        references = [a for a in manifest.artifacts if a.path.endswith("reference_io.pt")]
        geometry = [a for a in manifest.artifacts if a.path.endswith("timing_geometry.pt")]
        self.assertEqual(len(references), 14)
        self.assertEqual(len(geometry), 3)
        self.assertEqual(sum(a.size_bytes for a in geometry), 1634639)
        for artifact in manifest.artifacts:
            metadata = json.loads((root / artifact.task / "ut/meta.json").read_text())
            self.assertEqual(metadata[provision.HASH_KEYS[Path(artifact.path).name]], artifact.sha256)
        for task in manifest.tasks_without_persistent_fixtures:
            metadata = json.loads((root / task / "ut/meta.json").read_text())
            self.assertTrue(metadata["synthesized"])
            self.assertFalse(metadata.get("reference_io_sha256"))
            self.assertFalse(metadata.get("timing_geometry_sha256"))
            self.assertFalse(any(a.task == task for a in manifest.artifacts))
        self.assertEqual(set(manifest.tasks_without_persistent_fixtures), {
            task_directory(operation).relative_to(root).as_posix() for operation in (
                "glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle",
                "glm-5.3-flash__gemm_a16w16_bf16_cijk",
                "qwen3.8-2.4t__dense_bf16_gemm_cluster",
                "qwen3.8-2.4t__gemma_fused_add_rmsnorm",
            )
        })

    def test_rejects_paths_that_escape_or_replace_other_task_inputs(self):
        invalid = [
            "../outside.pt", "/tmp/outside.pt", f"{TASK}/ut/../reference_io.pt",
            f"{TASK}//ut/reference_io.pt", f"{TASK}/ut/reference_io.pt/",
            f"{TASK}\\ut\\reference_io.pt", f"{TASK}/source/reference_io.pt",
            f"{TASK}/ut/meta.json", "other-model__kernel/ut/reference_io.pt",
        ]
        for path in invalid:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                manifest = Path(directory) / "artifacts.json"
                manifest.write_text(json.dumps({"schema_version": 1, "artifacts": [
                    {"task": TASK, "path": path, "size_bytes": len(BODY), "sha256": DIGEST}
                ]}))
                with self.assertRaises(provision.ArtifactError):
                    provision.load_manifest(manifest)

    def test_rejects_missing_hash_and_duplicate_targets(self):
        row = {"task": TASK, "path": f"{TASK}/ut/reference_io.pt", "size_bytes": len(BODY), "sha256": DIGEST}
        for rows in [[{**row, "sha256": None}], [row, row]]:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                manifest = Path(directory) / "artifacts.json"
                manifest.write_text(json.dumps({"schema_version": 1, "artifacts": rows}))
                with self.assertRaises(provision.ArtifactError):
                    provision.load_manifest(manifest)

    def test_rejects_symlink_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "artifacts.json"
            alias.symlink_to(ROOT / "tasks/head_kernels/artifacts.json")
            with self.assertRaises(OSError):
                provision.load_manifest(alias)

    def test_task_selection_includes_fixture_free_tasks_and_rejects_typos(self):
        manifest = provision.load_manifest(ROOT / "tasks/head_kernels/artifacts.json")
        task = manifest.tasks_without_persistent_fixtures[0]
        self.assertEqual(provision.select_artifacts(manifest, [task]), [])
        with self.assertRaisesRegex(provision.ArtifactError, "unknown task"):
            provision.select_artifacts(manifest, ["misspelled-task"])


class NestedManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "artifacts.json"
        self.task = "example-model/workload/image_v1/kernel"
        self.row = {
            "task": self.task, "operation_id": TASK,
            "path": f"{self.task}/ut/reference_io.pt",
            "mirror_path": f"{TASK}/ut/reference_io.pt",
            "size_bytes": len(BODY), "sha256": DIGEST,
        }

    def load(self, rows=None, **extra):
        self.path.write_text(json.dumps({
            "schema_version": 2, "artifacts": [self.row] if rows is None else rows,
            **extra,
        }))
        return provision.load_manifest(self.path)

    def test_requires_explicit_flat_mirror_path_and_stable_operation_id(self):
        for field in ("mirror_path", "operation_id"):
            row = dict(self.row)
            row.pop(field)
            with self.subTest(field=field), self.assertRaises(provision.ArtifactError):
                self.load([row])

    def test_rejects_unsafe_nested_task_destination_and_mirror_paths(self):
        bad_paths = ["../outside", "/outside", "model/../outside", "model//kernel",
                     "model/kernel/", "model\\kernel", "model/./kernel", "model/KERNEL"]
        for task in bad_paths:
            with self.subTest(task=task), self.assertRaises(provision.ArtifactError):
                self.load([{**self.row, "task": task, "path": f"{task}/ut/reference_io.pt"}])
        changes = [
            {"path": f"{self.task}/source/reference_io.pt"},
            {"path": f"{self.task}/ut/meta.json"},
            {"path": "other/task/ut/reference_io.pt"},
            {"path": f"{self.task}/ut/../reference_io.pt"},
            {"mirror_path": f"{TASK}//ut/reference_io.pt"},
            {"mirror_path": f"{TASK}/ut/../reference_io.pt"},
            {"mirror_path": f"{TASK}/ut/timing_geometry.pt"},
            {"mirror_path": f"{self.task}/ut/reference_io.pt"},
            {"mirror_path": "other-task/ut/reference_io.pt"},
            {"mirror_path": f"/{TASK}/ut/reference_io.pt"},
            {"operation_id": "model/kernel"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(provision.ArtifactError):
                self.load([{**self.row, **change}])

    def test_rejects_duplicate_destinations_and_conflicting_mirror_claims(self):
        with self.assertRaisesRegex(provision.ArtifactError, "duplicate artifact"):
            self.load([self.row, self.row])
        second_task = "example-model/workload/image_v2/kernel"
        second = {**self.row, "task": second_task,
                  "path": f"{second_task}/ut/reference_io.pt", "sha256": "a" * 64}
        with self.assertRaisesRegex(provision.ArtifactError, "conflicting declarations"):
            self.load([self.row, second])

    def test_rejects_conflicting_operation_ids_within_one_task(self):
        second = {**self.row, "operation_id": "other-task",
                  "path": f"{self.task}/ut/timing_geometry.pt",
                  "mirror_path": "other-task/ut/timing_geometry.pt"}
        with self.assertRaisesRegex(provision.ArtifactError, "conflicting operation IDs"):
            self.load([self.row, second])

    def test_fixture_free_tasks_require_valid_v2_identity(self):
        empty_task = "example-model/workload/image_v1/synthetic"
        exempt = {"task": empty_task, "operation_id": "example-model__synthetic",
                  "reason": "Independent runtime reference from deterministic inputs."}
        manifest = self.load([], tasks_without_persistent_fixtures=[exempt])
        self.assertEqual(manifest.tasks, {empty_task})
        self.assertEqual(provision.select_artifacts(manifest, [empty_task]), [])
        del exempt["operation_id"]
        with self.assertRaises(provision.ArtifactError):
            self.load([], tasks_without_persistent_fixtures=[exempt])

    def test_preserves_v1_flat_manifest_contract(self):
        row = {"task": TASK, "path": f"{TASK}/ut/reference_io.pt",
               "size_bytes": len(BODY), "sha256": DIGEST}
        artifact, = self.load([row], schema_version=1).artifacts
        self.assertEqual(artifact.path, artifact.mirror_path)
        self.assertEqual(artifact.operation_id, TASK)
        with self.assertRaises(provision.ArtifactError):
            self.load([self.row], schema_version=1)

    def test_rejects_unknown_or_noninteger_schema_versions(self):
        for version in (0, 3, True, "2", None):
            with self.subTest(version=version), self.assertRaises(provision.ArtifactError):
                self.load(schema_version=version)


if __name__ == "__main__":
    unittest.main()
