"""Trusted sidecar entry point for rocJITsu replay-capsule execution.

The scoring process selects this module; a Triton/FlyDSL submission cannot
replace it with an arbitrary launcher.  Validation is repeated inside the
sidecar immediately before compilation so a capsule that changed after the
plugin capability check fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .native_launcher import NativeLauncherContract
from .replay_capsule import CapsuleValidationError, ReplayCapsule


SUPPORTED_ADAPTERS = frozenset({"triton_aot", "flydsl_aot"})
SUPPORTED_ADAPTER_VERSIONS = {"triton_aot": "1", "flydsl_aot": "1"}
SUPPORTED_ARCH = "gfx950"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_replay_identity(
    capsule: ReplayCapsule,
    *,
    expected_adapter: str,
    expected_arch: str,
    expected_kernel: str,
) -> None:
    if expected_adapter not in SUPPORTED_ADAPTERS:
        raise CapsuleValidationError(
            f"unsupported rocJITsu replay adapter {expected_adapter!r}"
        )
    if expected_arch != SUPPORTED_ARCH:
        raise CapsuleValidationError(
            f"rocJITsu AOT replay is verified only for {SUPPORTED_ARCH}"
        )
    if capsule.producer.adapter != expected_adapter:
        raise CapsuleValidationError(
            "capsule producer adapter does not match the task profile: "
            f"{capsule.producer.adapter!r} != {expected_adapter!r}"
        )
    expected_version = SUPPORTED_ADAPTER_VERSIONS[expected_adapter]
    if capsule.producer.adapter_version != expected_version:
        raise CapsuleValidationError(
            "capsule producer adapter version is not supported: "
            f"{capsule.producer.adapter_version!r} != {expected_version!r}"
        )
    if capsule.target.gpu_arch != expected_arch:
        raise CapsuleValidationError(
            "capsule target does not match the selected GPU architecture: "
            f"{capsule.target.gpu_arch!r} != {expected_arch!r}"
        )
    if capsule.code_object.kernel_name != expected_kernel:
        raise CapsuleValidationError(
            "capsule kernel changed after invocation construction: "
            f"{capsule.code_object.kernel_name!r} != {expected_kernel!r}"
        )


def execute_replay(
    *,
    capsule_path: Path,
    output_dir: Path,
    rocjitsu: Path,
    config: Path,
    hipcc: Path,
    expected_adapter: str,
    expected_arch: str,
    expected_kernel: str,
    expected_capsule_sha256: str,
) -> int:
    actual_capsule_sha256 = sha256_file(capsule_path)
    if actual_capsule_sha256 != expected_capsule_sha256:
        raise CapsuleValidationError(
            "capsule JSON changed after the trusted plugin validated it"
        )

    capsule = ReplayCapsule.load(capsule_path, verify_files=True)
    validate_replay_identity(
        capsule,
        expected_adapter=expected_adapter,
        expected_arch=expected_arch,
        expected_kernel=expected_kernel,
    )
    plan = NativeLauncherContract().materialize(
        capsule_path,
        output_dir,
        hipcc=hipcc,
    )
    print(
        "AKA_REPLAY_CAPSULE "
        f"sha256={actual_capsule_sha256} "
        f"code_sha256={capsule.code_object.sha256} "
        f"adapter={expected_adapter} arch={expected_arch} "
        f"kernel={expected_kernel}",
        flush=True,
    )

    compiled = subprocess.run(plan.compile_command, check=False)
    if compiled.returncode != 0:
        print(
            f"trusted replay launcher compilation failed with rc={compiled.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return int(compiled.returncode) or 70

    command = (
        str(rocjitsu),
        "--config",
        str(config),
        "--",
        *plan.run_command,
    )
    replayed = subprocess.run(command, check=False)
    return int(replayed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rocjitsu", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hipcc", type=Path, default=Path("/opt/rocm/bin/hipcc"))
    parser.add_argument("--expected-adapter", required=True)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--expected-kernel", required=True)
    parser.add_argument("--expected-capsule-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute_replay(
            capsule_path=args.capsule.resolve(),
            output_dir=args.output_dir.resolve(),
            rocjitsu=args.rocjitsu,
            config=args.config,
            hipcc=args.hipcc,
            expected_adapter=args.expected_adapter,
            expected_arch=args.expected_arch,
            expected_kernel=args.expected_kernel,
            expected_capsule_sha256=args.expected_capsule_sha256,
        )
    except (CapsuleValidationError, OSError, ValueError) as error:
        print(f"trusted rocJITsu replay rejected: {error}", file=sys.stderr)
        return 64


if __name__ == "__main__":  # pragma: no cover - exercised via main() and sidecar smoke tests.
    raise SystemExit(main())


__all__ = [
    "SUPPORTED_ADAPTERS",
    "SUPPORTED_ADAPTER_VERSIONS",
    "SUPPORTED_ARCH",
    "execute_replay",
    "main",
    "sha256_file",
    "validate_replay_identity",
]
