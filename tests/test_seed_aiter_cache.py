import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.scripts.seed_aiter_cache import COPY_BATCH_SIZE, manifest, seed


class AiterCacheSeedTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("rclone"), "rclone is required for the real local-copy check")
    def test_complete_copy_preserves_binary_private_files_links_and_empty_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "module_quant.so").write_bytes(bytes(range(256)))
            (source / "flydsl_cache").mkdir()
            private = source / "flydsl_cache/private.pkl"
            private.write_bytes(b"private-cache-bytes\x00\xff")
            private.chmod(0o600)
            (source / "empty").mkdir()
            (source / "module-link.so").symlink_to("module_quant.so")
            (source / "directory-link").symlink_to("flydsl_cache", target_is_directory=True)
            cache = root / "cache"
            cache.mkdir()
            seed(source, cache / "jit", os.getuid(), os.getgid(), Path(shutil.which("rclone")))
            self.assertEqual(manifest(source), manifest(cache / "jit"))
            record = json.loads((cache / "SEED-MANIFEST.json").read_text())
            self.assertIn("flydsl_cache/private.pkl", record["entries"])
            self.assertIn("module_quant.so", record["entries"])

    @unittest.skipUnless(shutil.which("rclone"), "rclone is required for the real local-copy check")
    def test_multiple_bounded_batches_cover_every_file_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for index in range(COPY_BATCH_SIZE * 2 + 1):
                (source / f"cache-{index:04}.pkl").write_bytes(str(index).encode())
            (source / "link.pkl").symlink_to("cache-0000.pkl")
            cache = root / "cache"
            cache.mkdir()
            observed = []
            actual_run = subprocess.run
            def run_batch(command, **kwargs):
                listing = Path(command[command.index("--files-from-raw") + 1]).read_text().splitlines()
                observed.append(listing)
                self.assertLessEqual(len(listing), 128)
                self.assertEqual(command[command.index("--transfers") + 1], "64000")
                self.assertEqual(kwargs["env"]["GOMAXPROCS"], "1")
                return actual_run(command, **kwargs)
            with patch("src.scripts.seed_aiter_cache.subprocess.run", side_effect=run_batch):
                seed(source, cache / "jit", os.getuid(), os.getgid(), Path(shutil.which("rclone")))
            self.assertEqual([len(batch) for batch in observed], [128, 128, 2])
            self.assertIn("link.pkl.rclonelink", [name for batch in observed for name in batch])
            self.assertEqual(manifest(source), manifest(cache / "jit"))

    def test_partial_copy_cannot_produce_verification_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "module_quant.so").write_bytes(b"image-baseline")
            cache = root / "cache"
            cache.mkdir()
            with patch("src.scripts.seed_aiter_cache.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
                with self.assertRaisesRegex(RuntimeError, "differs from complete"):
                    seed(source, cache / "jit", os.getuid(), os.getgid(), Path("rclone"))
            self.assertFalse((cache / "SEED-MANIFEST.json").exists())

    def test_empty_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "empty"):
                seed(root, root / "jit", os.getuid(), os.getgid(), Path("rclone"))


if __name__ == "__main__":
    unittest.main()
