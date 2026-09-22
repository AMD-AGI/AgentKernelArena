"""CPU-only whole-object OCI delivery tests; rclone is always mocked."""
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from src.tools import prepare_head_kernel_artifacts as prepare


class OCIDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = 'example-model__kernel'
        self.ut = self.root / self.task / 'ut'
        self.ut.mkdir(parents=True)
        self.bodies = {f'{self.task}/ut/reference_io.pt': b'original exact reference',
                       f'{self.task}/ut/timing_geometry.pt': b'original exact geometry'}
        self.artifacts = [prepare.Artifact(self.task, path, len(data), hashlib.sha256(data).hexdigest())
                          for path, data in self.bodies.items()]
        (self.ut / 'meta.json').write_text(json.dumps({
            'reference_io_sha256': self.artifacts[0].sha256,
            'timing_geometry_sha256': self.artifacts[1].sha256}))
        self.remote = 'oci:bucket/owned-prefix/tensors'
        self.requested = []
        self.downloaded_inodes = {}

    def fake_copy(self, command, **kwargs):
        self.assertEqual(command[:2], ['rclone', 'copy'])
        self.assertEqual(command[2], self.remote)
        self.assertEqual(command[command.index('--transfers') + 1], '64000')
        self.assertIn('--progress', command)
        self.assertEqual(kwargs, {'check': True})
        self.requested = Path(command[command.index('--files-from-raw') + 1]).read_text().splitlines()
        for relative in self.requested:
            destination = Path(command[3]) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.bodies[relative])
            info = destination.stat()
            self.downloaded_inodes[relative] = (info.st_dev, info.st_ino)
        return subprocess.CompletedProcess(command, 0)

    def download(self):
        return prepare.download_artifacts(self.artifacts, self.root, self.remote)

    def assert_no_partial(self):
        self.assertEqual(list(self.ut.glob('.*.partial')), [])

    def test_exact_whole_objects_have_independent_readonly_destinations(self):
        with mock.patch.object(prepare.subprocess, 'run', side_effect=self.fake_copy) as run:
            result = self.download()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.requested, [row.path for row in self.artifacts])
        self.assertEqual(len(result), 2)
        for artifact in self.artifacts:
            target = self.root / artifact.path
            self.assertEqual(target.read_bytes(), self.bodies[artifact.path])
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertNotEqual((target.stat().st_dev, target.stat().st_ino), self.downloaded_inodes[artifact.path])
        self.assert_no_partial()

    def test_only_missing_files_are_requested_and_existing_file_is_preserved(self):
        artifact = self.artifacts[0]
        target = self.root / artifact.path
        target.write_bytes(self.bodies[artifact.path]); before = target.stat()
        with mock.patch.object(prepare.subprocess, 'run', side_effect=self.fake_copy):
            self.download()
        self.assertEqual(self.requested, [self.artifacts[1].path])
        after = target.stat()
        self.assertEqual((before.st_ino, before.st_mode, before.st_mtime_ns),
                         (after.st_ino, after.st_mode, after.st_mtime_ns))

    def test_all_existing_files_need_no_rclone(self):
        for artifact in self.artifacts:
            (self.root / artifact.path).write_bytes(self.bodies[artifact.path])
        with mock.patch.object(prepare.subprocess, 'run') as run:
            self.download()
        run.assert_not_called()

    def test_unknown_existing_file_is_not_overwritten(self):
        target = self.root / self.artifacts[0].path
        target.write_bytes(b'x' * self.artifacts[0].size_bytes)
        with mock.patch.object(prepare.subprocess, 'run') as run:
            with self.assertRaisesRegex(prepare.ArtifactError, 'SHA-256 mismatch'):
                self.download()
        run.assert_not_called()
        self.assertEqual(target.read_bytes(), b'x' * self.artifacts[0].size_bytes)

    def test_failed_or_interrupted_copy_installs_nothing(self):
        for error in [subprocess.CalledProcessError(2, ['rclone', 'copy']), KeyboardInterrupt()]:
            def interrupted(command, **kwargs):
                self.fake_copy(command, **kwargs)
                raise error
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(prepare.subprocess, 'run', side_effect=interrupted):
                    with self.assertRaises((prepare.ArtifactError, KeyboardInterrupt)):
                        self.download()
                self.assertTrue(all(not (self.root / row.path).exists() for row in self.artifacts))
                self.assert_no_partial()

    def test_missing_object_is_not_accepted_when_rclone_exits_zero(self):
        def missing(command, **kwargs):
            self.fake_copy(command, **kwargs)
            (Path(command[3]) / self.artifacts[0].path).unlink()
        with mock.patch.object(prepare.subprocess, 'run', side_effect=missing):
            with self.assertRaises(FileNotFoundError):
                self.download()
        self.assertTrue(all(not (self.root / row.path).exists() for row in self.artifacts))
        self.assert_no_partial()

    def test_corrupt_or_wrong_size_object_is_not_published(self):
        for body in [b'x' * self.artifacts[0].size_bytes, b'short']:
            self.bodies[self.artifacts[0].path] = body
            with self.subTest(bytes=len(body)), mock.patch.object(prepare.subprocess, 'run', side_effect=self.fake_copy):
                with self.assertRaises(prepare.ArtifactError):
                    self.download()
            self.assertFalse((self.root / self.artifacts[0].path).exists())
            self.assert_no_partial()

    def test_remote_override_uses_exact_caller_root(self):
        self.remote = 'another_oci:other-bucket/approved-prefix'
        with mock.patch.object(prepare.subprocess, 'run', side_effect=self.fake_copy):
            self.download()

    def test_bad_roots_are_rejected_before_subprocess(self):
        for value in [None, ':s3:bucket', 'oci:/absolute', 'oci:bucket/../escape', 'oci:bucket//duplicate', 'https://example.com']:
            with self.subTest(remote=value), mock.patch.object(prepare.subprocess, 'run') as run:
                with self.assertRaises(prepare.ArtifactError):
                    prepare.download_artifacts(self.artifacts, self.root, value)
                run.assert_not_called()

    def test_manifest_declares_remote_without_credentials(self):
        manifest = self.root / 'artifacts.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'oci_storage': {'remote_root': self.remote},
            'artifacts': [{'task': row.task, 'path': row.path, 'size_bytes': row.size_bytes, 'sha256': row.sha256}
                          for row in self.artifacts]}))
        self.assertEqual(prepare.load_manifest(manifest).oci_remote_root, self.remote)


if __name__ == '__main__':
    unittest.main()
