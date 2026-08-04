"""Trusted synthetic candidate harness used by real sidecar integration tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


FRAMEWORK = Path("/opt/aka-eval-tools/src/eval_tools")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attest(
    *,
    tool: str,
    artifact: Path,
    build_command: list[str],
    environment: dict[str, str],
    evidence: dict[str, bool] | None = None,
) -> None:
    path = Path(os.environ["AKA_BUILD_ATTESTATION_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = artifact.resolve(strict=True)
    relative = artifact.relative_to(path.parent.resolve(strict=True))
    payload = {
        "tool": tool,
        "instrumented": True,
        "compiler": build_command[0],
        "compiler_version": "runtime-qualified",
        "target_arch": "gfx950",
        "build_command": build_command,
        "artifact_path": relative.as_posix(),
        "artifact_sha256": _sha256(artifact),
        "environment": environment,
        "evidence": evidence or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result


def _cached_code_object(cache: Path) -> Path:
    for path in sorted(cache.rglob("*")):
        if path.is_file() and path.read_bytes()[:4] == b"\x7fELF":
            return path
    raise RuntimeError(f"no cached ELF code object below {cache}")


def _triton(mode: str, bug: bool) -> int:
    probe = FRAMEWORK / (
        "probes/triton_fpsan_probe.py"
        if mode == "triton_fpsan"
        else "probes/triton_asan_probe.py"
    )
    command = [sys.executable, str(probe), *( ["wrong"] if bug and mode == "triton_fpsan" else ["oob"] if bug else [] )]
    result = _run(command)
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    artifact = _cached_code_object(cache)
    if mode == "triton_fpsan":
        environment = {"TRITON_INSTRUMENTATION_MODE": "fpsan"}
        evidence = {"reference_instrumented": True, "candidate_instrumented": True}
    else:
        environment = {
            "TRITON_ENABLE_ASAN": os.environ["TRITON_ENABLE_ASAN"],
            "HSA_XNACK": os.environ["HSA_XNACK"],
        }
        evidence = None
    _attest(
        tool=mode,
        artifact=artifact,
        build_command=["triton-jit", str(probe)],
        environment=environment,
        evidence=evidence,
    )
    return result.returncode


def _hip(mode: str, bug: bool) -> int:
    attestation = Path(os.environ["AKA_BUILD_ATTESTATION_PATH"])
    build_dir = attestation.parent / "candidate-build"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / mode.replace("_", "-")
    hipcc = "/opt/rocm/bin/hipcc"
    if mode == "gpu_asan":
        source = FRAMEWORK / "probes/gpu_asan_probe.hip"
        flags = [
            "-O2",
            "-fsanitize=address",
            "-shared-libsan",
            "--offload-arch=gfx950:xnack+",
        ]
        environment = {"HSA_XNACK": os.environ["HSA_XNACK"]}
    else:
        source = FRAMEWORK / "probes/hip_fpsan_probe.hip"
        include = os.environ["FPSAN_INCLUDE_DIR"]
        flags = ["-O2", f"-I{include}", "-DAKA_HIP_FPSAN=1", "--offload-arch=gfx950"]
        environment = {"AKA_HIP_FPSAN": os.environ["AKA_HIP_FPSAN"]}
    build_command = [hipcc, *flags, str(source), "-o", str(binary)]
    compiled = _run(build_command)
    if compiled.returncode != 0:
        return compiled.returncode
    _attest(
        tool=mode,
        artifact=binary,
        build_command=build_command,
        environment=environment,
        evidence=(
            {"reference_instrumented": True, "candidate_instrumented": True}
            if mode == "hip_fpsan"
            else None
        ),
    )
    return _run([str(binary), *( ["wrong"] if bug and mode == "hip_fpsan" else ["oob"] if bug else [] )]).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("triton_fpsan", "gpu_asan_triton", "gpu_asan", "hip_fpsan"),
    )
    parser.add_argument("--bug", action="store_true")
    args = parser.parse_args()
    if args.mode in {"triton_fpsan", "gpu_asan_triton"}:
        return _triton("gpu_asan" if args.mode == "gpu_asan_triton" else args.mode, args.bug)
    return _hip(args.mode, args.bug)


if __name__ == "__main__":
    raise SystemExit(main())
