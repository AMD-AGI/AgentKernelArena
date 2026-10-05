import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.scripts.seed_aiter_cache import manifest, seed


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
            cache = root / "cache"
            cache.mkdir()
            seed(source, cache / "jit", os.getuid(), os.getgid(), Path(shutil.which("rclone")))
            self.assertEqual(manifest(source), manifest(cache / "jit"))
            record = json.loads((cache / "SEED-MANIFEST.json").read_text())
            self.assertIn("flydsl_cache/private.pkl", record["entries"])
            self.assertIn("module_quant.so", record["entries"])

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
