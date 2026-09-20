"""Verify Docker engine identity without conflating manifest and config digests.

Docker's classic image store reports the config digest as Id. Its containerd
store can report the target manifest digest instead. A manifest-shaped Id is
accepted only with local descriptor/RepoDigest corroboration and a committed,
hash-verified registry manifest that binds it to the expected config digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


MANIFEST_ROOT = Path(__file__).resolve().parents[2] / "docker/head-kernels/manifests"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
MEDIA_TYPES = {
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
}


def canonical_repository(value: str) -> str:
    first = value.split("/", 1)[0]
    if "." not in first and ":" not in first and first != "localhost":
        value = "docker.io/" + value
    if value.startswith("index.docker.io/"):
        value = "docker.io/" + value[len("index.docker.io/"):]
    return value


def verify_identity(reference: str, expected_config: str | None, inspection: dict,
                    manifest_root: Path = MANIFEST_ROOT) -> dict:
    engine_id = inspection.get("Id")
    if not isinstance(engine_id, str) or not DIGEST.fullmatch(engine_id):
        raise ValueError("Docker returned an invalid engine image ID")
    if expected_config and not DIGEST.fullmatch(expected_config):
        raise ValueError("Expected config digest must be a complete sha256 digest")
    repo_digests = inspection.get("RepoDigests") or []
    if not isinstance(repo_digests, list) or any(not isinstance(item, str) for item in repo_digests):
        raise ValueError("Docker returned invalid RepoDigests metadata")
    result = {
        "engine_image_id": engine_id,
        "engine_id_role": "config_digest" if expected_config == engine_id else "unclassified",
        "verified_config_digest": engine_id if expected_config == engine_id else None,
        "manifest_digest": None,
        "repo_digests": repo_digests,
        "descriptor": inspection.get("Descriptor"),
        "manifest_config_binding_verified": False,
    }
    if expected_config == engine_id:
        return result
    if "@" not in reference:
        if expected_config:
            raise ValueError("Runtime image ID mismatch; manifest identity requires a pinned manifest reference")
        return result
    repository, manifest_digest = reference.rsplit("@", 1)
    if not DIGEST.fullmatch(manifest_digest) or engine_id != manifest_digest:
        raise ValueError("Runtime engine ID matches neither the expected config nor the pinned manifest")
    if not expected_config:
        raise ValueError("Manifest engine identity requires an expected config digest")
    matching_repo_digest = any(
        "@" in item and item.rsplit("@", 1)[1] == manifest_digest
        and canonical_repository(item.rsplit("@", 1)[0]) == canonical_repository(repository)
        for item in repo_digests
    )
    if not matching_repo_digest:
        raise ValueError("Manifest engine identity lacks the selected pinned RepoDigest")
    descriptor = inspection.get("Descriptor") or {}
    if descriptor.get("digest") != manifest_digest or descriptor.get("mediaType") not in MEDIA_TYPES:
        raise ValueError("Manifest engine identity lacks a matching image-manifest Descriptor")
    path = manifest_root / (manifest_digest.removeprefix("sha256:") + ".json")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("No committed manifest-to-config evidence for this engine identity") from exc
    if len(raw) > 1048576 or hashlib.sha256(raw).hexdigest() != manifest_digest.removeprefix("sha256:"):
        raise ValueError("Committed manifest bytes do not match the selected manifest digest")
    manifest = json.loads(raw)
    if (manifest.get("schemaVersion") != 2 or manifest.get("mediaType") != descriptor["mediaType"]
            or descriptor.get("size") != len(raw)):
        raise ValueError("Docker Descriptor disagrees with the committed manifest")
    if (manifest.get("config") or {}).get("digest") != expected_config:
        raise ValueError("Pinned manifest does not reference the expected config digest")
    result.update(
        engine_id_role="manifest_digest", verified_config_digest=expected_config,
        manifest_digest=manifest_digest, manifest_config_binding_verified=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-config", default="")
    args = parser.parse_args()
    try:
        result = verify_identity(args.image, args.expected_config or None, json.load(sys.stdin))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Runtime image identity verification failed: {exc}\n")
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
