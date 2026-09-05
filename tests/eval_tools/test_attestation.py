from __future__ import annotations

import hashlib
import json

import pytest

from src.eval_tools.plugins.attestation import BuildAttestation, attest_artifact


def _attestation(artifact_path, digest: str) -> BuildAttestation:
    return BuildAttestation(
        tool="gpu_asan",
        instrumented=True,
        compiler="hipcc",
        compiler_version="7.2",
        target_arch="gfx950:xnack+",
        build_command=("hipcc", "-fsanitize=address", "candidate.hip"),
        artifact_path=artifact_path,
        artifact_sha256=digest,
    )


def test_dump_uses_path_relative_to_attestation_and_load_rebinds_it(tmp_path):
    tool_dir = tmp_path / "gpu_asan"
    artifact = tool_dir / "build" / "candidate.hsaco"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"candidate")
    path = tool_dir / "build_attestation.json"

    _attestation(
        artifact,
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
    ).dump(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_path"] == "build/candidate.hsaco"
    loaded = BuildAttestation.load(path)
    assert loaded.artifact_path == artifact.resolve()
    assert loaded.validate(
        expected_tool="gpu_asan", required_flags=("-fsanitize=address",)
    ) == (True, "ok")


@pytest.mark.parametrize("artifact_path", ["/artifacts/candidate.hsaco", "../escape"])
def test_load_rejects_artifact_paths_outside_attestation_dir(
    tmp_path, artifact_path
):
    path = tmp_path / "build_attestation.json"
    payload = _attestation(None, "0" * 64).to_dict()
    payload["artifact_path"] = artifact_path
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="attestation directory"):
        BuildAttestation.load(path)


def test_dump_rejects_artifact_outside_attestation_dir(tmp_path):
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    outside = tmp_path / "outside.hsaco"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="beside or below"):
        attest_artifact(
            tool="gpu_asan",
            artifact_path=outside,
            build_command=("hipcc", "-fsanitize=address"),
            compiler="hipcc",
            compiler_version="7.2",
            target_arch="gfx950:xnack+",
            environment={"HSA_XNACK": "1"},
        ).dump(tool_dir / "build_attestation.json")


def test_validate_fails_if_relative_artifact_symlink_moves_outside(tmp_path):
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    inside = tool_dir / "candidate.hsaco"
    inside.write_bytes(b"inside")
    digest = hashlib.sha256(inside.read_bytes()).hexdigest()
    path = tool_dir / "build_attestation.json"
    _attestation(inside, digest).dump(path)
    loaded = BuildAttestation.load(path)

    inside.unlink()
    outside = tmp_path / "outside.hsaco"
    outside.write_bytes(b"inside")
    inside.symlink_to(outside)

    assert loaded.validate(expected_tool="gpu_asan") == (
        False,
        "attested_artifact_outside_attestation_dir",
    )


@pytest.mark.parametrize(
    "near_match",
    ["-DNOTE=-fsanitize=address", "--not-shared-libsan"],
)
def test_required_build_flags_reject_substring_near_matches(near_match):
    attestation = BuildAttestation(
        tool="gpu_asan",
        instrumented=True,
        compiler="hipcc",
        compiler_version="7.2",
        target_arch="gfx950:xnack+",
        build_command=("hipcc", near_match, "candidate.hip"),
    )
    assert attestation.validate(
        expected_tool="gpu_asan",
        required_flags=("-fsanitize=address",),
        require_artifact=False,
    ) == (False, "missing_build_flag:-fsanitize=address")


def test_required_build_flags_accept_explicit_split_include_form():
    attestation = BuildAttestation(
        tool="hip_fpsan",
        instrumented=True,
        compiler="hipcc",
        compiler_version="7.2",
        target_arch="gfx950",
        build_command=("hipcc", "-I", "/opt/hip-fpsan/include", "candidate.hip"),
    )
    assert attestation.validate(
        expected_tool="hip_fpsan",
        required_flags=("-I/opt/hip-fpsan/include",),
        require_artifact=False,
    ) == (True, "ok")
