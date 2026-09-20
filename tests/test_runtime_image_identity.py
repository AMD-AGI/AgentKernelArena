"""CPU tests for classic and containerd image-store identity semantics."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.tools.runtime_image_identity import MANIFEST_ROOT, verify_identity


class ImageIdentityTests(unittest.TestCase):
    def setUp(self):
        self.manifest = "sha256:1f5464829559b086eb66f9b803cb9c7a817438c43edff2d5ef59b46a186745f6"
        self.config = "sha256:ffe4af630e49b05c812db4a468bfb411c3dbb0e93124801f28349bfa31352dea"
        self.reference = "docker.io/rocm/hyperloom@" + self.manifest
        self.raw = (MANIFEST_ROOT / (self.manifest.removeprefix("sha256:") + ".json")).read_bytes()
        self.inspection = {
            "Id": self.manifest,
            "RepoDigests": ["rocm/hyperloom@" + self.manifest],
            "Descriptor": {"digest": self.manifest, "size": len(self.raw),
                           "mediaType": json.loads(self.raw)["mediaType"]},
        }

    def test_classic_config_id_is_preserved(self):
        result = verify_identity(self.reference, self.config, {"Id": self.config, "RepoDigests": []})
        self.assertEqual(result["engine_id_role"], "config_digest")
        self.assertEqual(result["verified_config_digest"], self.config)

    def test_manifest_engine_id_requires_committed_config_binding(self):
        result = verify_identity(self.reference, self.config, self.inspection)
        self.assertEqual(result["engine_image_id"], self.manifest)
        self.assertEqual(result["engine_id_role"], "manifest_digest")
        self.assertEqual(result["verified_config_digest"], self.config)
        self.assertTrue(result["manifest_config_binding_verified"])

    def test_arbitrary_engine_id_is_rejected(self):
        self.inspection["Id"] = "sha256:" + "a" * 64
        with self.assertRaisesRegex(ValueError, "neither"):
            verify_identity(self.reference, self.config, self.inspection)

    def test_manifest_identity_without_repo_digest_is_rejected(self):
        self.inspection["RepoDigests"] = []
        with self.assertRaisesRegex(ValueError, "RepoDigest"):
            verify_identity(self.reference, self.config, self.inspection)

    def test_wrong_repository_is_rejected(self):
        self.inspection["RepoDigests"] = ["example.invalid/other@" + self.manifest]
        with self.assertRaisesRegex(ValueError, "RepoDigest"):
            verify_identity(self.reference, self.config, self.inspection)

    def test_missing_descriptor_is_rejected(self):
        self.inspection.pop("Descriptor")
        with self.assertRaisesRegex(ValueError, "Descriptor"):
            verify_identity(self.reference, self.config, self.inspection)

    def test_wrong_descriptor_size_is_rejected(self):
        self.inspection["Descriptor"]["size"] += 1
        with self.assertRaisesRegex(ValueError, "Descriptor"):
            verify_identity(self.reference, self.config, self.inspection)

    def test_wrong_expected_config_cannot_match_manifest(self):
        with self.assertRaisesRegex(ValueError, "expected config"):
            verify_identity(self.reference, "sha256:" + "b" * 64, self.inspection)

    def test_missing_or_tampered_committed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No committed"):
                verify_identity(self.reference, self.config, self.inspection, root)
            (root / (self.manifest.removeprefix("sha256:") + ".json")).write_bytes(self.raw + b" ")
            with self.assertRaisesRegex(ValueError, "do not match"):
                verify_identity(self.reference, self.config, self.inspection, root)

    def test_both_committed_profile_manifests_hash_to_their_names(self):
        paths = list(MANIFEST_ROOT.glob("*.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.stem)


if __name__ == "__main__":
    unittest.main()
