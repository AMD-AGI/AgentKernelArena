"""Build and instrumentation attestations used to prevent false sanitizer PASSes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BuildAttestation:
    """Evidence that the artifact which ran was built with the requested tool.

    A zero exit code with no finding is not a sanitizer PASS unless this object
    verifies.  In particular, preloading an ASan runtime cannot instrument an
    already-built HSACO.
    """

    tool: str
    instrumented: bool
    compiler: str
    compiler_version: str
    target_arch: str
    build_command: tuple[str, ...]
    artifact_path: Optional[Path] = None
    artifact_sha256: Optional[str] = None
    environment: Mapping[str, str] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    # Set by ``load``.  It is deliberately excluded from the JSON contract and
    # binds a relative artifact path to the directory containing the
    # attestation in the scoring namespace.
    artifact_root: Optional[Path] = field(
        default=None, repr=False, compare=False
    )

    def validate(
        self,
        *,
        expected_tool: str,
        required_flags: Iterable[str] = (),
        required_env: Mapping[str, str] = {},
        require_artifact: bool = True,
    ) -> tuple[bool, str]:
        if self.tool != expected_tool:
            return False, "attestation_tool_mismatch"
        if not self.instrumented:
            return False, "artifact_not_instrumented"
        command_text = " ".join(self.build_command)
        for flag in required_flags:
            if flag not in command_text:
                return False, f"missing_build_flag:{flag}"
        for key, expected in required_env.items():
            if str(self.environment.get(key, "")) != str(expected):
                return False, f"missing_environment_attestation:{key}"
        if require_artifact:
            if self.artifact_path is None or self.artifact_sha256 is None:
                return False, "missing_artifact_attestation"
            if not _SHA256_RE.fullmatch(self.artifact_sha256):
                return False, "invalid_artifact_sha256"
            artifact_path = self.artifact_path.resolve(strict=False)
            if self.artifact_root is not None:
                root = self.artifact_root.resolve(strict=False)
                try:
                    artifact_path.relative_to(root)
                except ValueError:
                    return False, "attested_artifact_outside_attestation_dir"
            if not artifact_path.is_file():
                return False, "attested_artifact_missing"
            if sha256_file(artifact_path) != self.artifact_sha256:
                return False, "attested_artifact_hash_mismatch"
        return True, "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "instrumented": self.instrumented,
            "compiler": self.compiler,
            "compiler_version": self.compiler_version,
            "target_arch": self.target_arch,
            "build_command": list(self.build_command),
            "artifact_path": self._serialized_artifact_path(),
            "artifact_sha256": self.artifact_sha256,
            "environment": dict(self.environment),
            "evidence": dict(self.evidence),
        }

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        root = path.parent.resolve(strict=True)
        payload = self.to_dict()
        if self.artifact_path is not None:
            artifact_path = self.artifact_path
            if not artifact_path.is_absolute():
                artifact_path = root / artifact_path
            artifact_path = artifact_path.resolve(strict=False)
            try:
                relative = artifact_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "attested artifact must be stored beside or below its attestation"
                ) from error
            payload["artifact_path"] = relative.as_posix()
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _serialized_artifact_path(self) -> str | None:
        if self.artifact_path is None:
            return None
        if self.artifact_root is None:
            return str(self.artifact_path)
        root = self.artifact_root.resolve(strict=False)
        artifact_path = self.artifact_path.resolve(strict=False)
        try:
            return artifact_path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                "attested artifact is outside the attestation directory"
            ) from error

    @classmethod
    def load(cls, path: Path) -> "BuildAttestation":
        attestation_path = path.resolve(strict=True)
        raw = json.loads(attestation_path.read_text(encoding="utf-8"))
        artifact = raw.get("artifact_path")
        artifact_root = attestation_path.parent.resolve(strict=True)
        artifact_path: Path | None = None
        if artifact is not None:
            if not isinstance(artifact, str) or not artifact:
                raise ValueError("artifact_path must be a non-empty relative path")
            relative = Path(artifact)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "artifact_path must stay below the attestation directory"
                )
            artifact_path = (artifact_root / relative).resolve(strict=False)
            try:
                artifact_path.relative_to(artifact_root)
            except ValueError as error:
                raise ValueError(
                    "artifact_path must stay below the attestation directory"
                ) from error
        return cls(
            tool=str(raw["tool"]),
            instrumented=bool(raw["instrumented"]),
            compiler=str(raw.get("compiler", "")),
            compiler_version=str(raw.get("compiler_version", "")),
            target_arch=str(raw.get("target_arch", "")),
            build_command=tuple(str(v) for v in raw.get("build_command", [])),
            artifact_path=artifact_path,
            artifact_sha256=raw.get("artifact_sha256"),
            environment={str(k): str(v) for k, v in raw.get("environment", {}).items()},
            evidence=raw.get("evidence", {}),
            artifact_root=artifact_root,
        )


def attest_artifact(
    *,
    tool: str,
    artifact_path: Path,
    build_command: Iterable[str],
    compiler: str,
    compiler_version: str,
    target_arch: str,
    environment: Mapping[str, str],
    evidence: Mapping[str, Any] = {},
) -> BuildAttestation:
    return BuildAttestation(
        tool=tool,
        instrumented=True,
        compiler=compiler,
        compiler_version=compiler_version,
        target_arch=target_arch,
        build_command=tuple(str(v) for v in build_command),
        artifact_path=artifact_path,
        artifact_sha256=sha256_file(artifact_path),
        environment=dict(environment),
        evidence=dict(evidence),
    )
