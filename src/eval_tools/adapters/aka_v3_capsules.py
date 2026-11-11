"""Build rocJITsu replay capsules from a schema-v3 capture.

The workspace runner's ``sanitize capture`` lane records each Triton/Gluon
launch of one call: the compiled kernel's HSACO and metadata, the launch grid,
every argument, and each allocation the launch can reach before and after it
ran. This module turns one launch record into a validated
:class:`~.replay_capsule.ReplayCapsule` using the same ABI lowering as the
``triton_aot`` adapter, so a replay reconstructs that exact dispatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .replay_capsule import (
    AllocationSpec,
    CapsuleValidationError,
    CaseSpec,
    CodeObjectSpec,
    ProducerSpec,
    ReplayCapsule,
    TargetSpec,
)
from .triton_aot import extract_triton_aot

CAPTURE_SCHEMA = 1
# Code object ABI version byte (ELF e_ident[EI_ABIVERSION]) -> AMDHSA version.
_ABI_TO_CODE_OBJECT = {2: 4, 3: 5, 4: 6}
_EF_AMDGPU_FEATURE_XNACK = 0x300
_EF_AMDGPU_FEATURE_XNACK_ON = 0x300


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_object_target(data: bytes) -> tuple[int, bool]:
    """(AMDHSA code object version, xnack-on) from an AMDGPU ELF header."""

    if data[:4] != b"\x7fELF":
        raise CapsuleValidationError("code object is not an ELF file")
    version = _ABI_TO_CODE_OBJECT.get(data[8])
    if version is None:
        raise CapsuleValidationError(f"unsupported AMDHSA ABI version byte {data[8]}")
    flags = int.from_bytes(data[0x30:0x34], "little")
    return version, (flags & _EF_AMDGPU_FEATURE_XNACK) == _EF_AMDGPU_FEATURE_XNACK_ON


def load_manifest(capture_dir: Path) -> dict:
    manifest = json.loads((capture_dir / "capture.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CAPTURE_SCHEMA:
        raise CapsuleValidationError("capture.json has an unsupported schema")
    return manifest


def _constant_key(raw: str) -> Any:
    value = json.loads(raw)
    return tuple(value) if isinstance(value, list) else value


def build_capsule(
    launch_dir: Path,
    *,
    manifest: Mapping[str, Any],
    task_id: str,
    image_digest: str,
    rocm_version: str,
    framework_version: str,
    gpu_arch: str = "gfx950",
) -> Path:
    """Write ``capsule.json`` beside one launch record and return its path."""

    record = json.loads((launch_dir / "launch.json").read_text(encoding="utf-8"))
    hsaco = launch_dir / record["hsaco"]
    data = hsaco.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["hsaco_sha256"]:
        raise CapsuleValidationError("captured HSACO hash does not match its record")
    code_object_version, xnack = code_object_target(data)
    pointer_bindings: dict[int, tuple[str, int]] = {}
    scalar_values: dict[int, Any] = {}
    for argument in record["arguments"]:
        index = int(argument["index"])
        if argument["kind"] == "tensor":
            pointer_bindings[index] = (str(argument["allocation"]), int(argument["byte_offset"]))
        elif argument["kind"] == "value":
            value = argument["value"]
            scalar_values[index] = int(value) if isinstance(value, bool) else value
    compiled = SimpleNamespace(
        asm={"hsaco": data},
        name=record["kernel_name"],
        metadata=dict(record["metadata"]),
        src=SimpleNamespace(
            signature=dict(record["signature"]),
            constants={_constant_key(key): value for key, value in record["constants"].items()},
            fn=SimpleNamespace(arg_names=list(record["arg_names"])),
        ),
    )
    staging = launch_dir / "aot"
    artifact = extract_triton_aot(compiled, staging, grid=record["grid"], pointer_bindings=pointer_bindings,
                                  scalar_values=scalar_values)
    if artifact.hsaco_sha256 != record["hsaco_sha256"]:
        raise CapsuleValidationError("lowered HSACO differs from the captured one")
    referenced = {argument.ref for argument in artifact.abi if argument.kind == "pointer"}
    allocations = []
    for item in record["allocations"]:
        if item["id"] not in referenced:
            continue
        before = launch_dir / item["before_blob"]
        expected = launch_dir / item["expected_blob"]
        allocations.append(AllocationSpec(
            id=item["id"], byte_size=int(item["byte_size"]), before_blob=item["before_blob"],
            before_sha256=_sha256(before), expected_blob=item["expected_blob"],
            expected_sha256=_sha256(expected), alignment=256))
    if not allocations:
        raise CapsuleValidationError("the launch reaches no captured allocation")
    capsule = ReplayCapsule(
        schema_version=1,
        producer=ProducerSpec(adapter="triton_aot", adapter_version="1", framework_version=framework_version,
                              image_digest=image_digest, rocm_version=rocm_version),
        target=TargetSpec(gpu_arch=gpu_arch, xnack=xnack, code_object_version=code_object_version),
        case=CaseSpec(task_id=task_id, case_id=str(manifest.get("case")), seed=manifest.get("seed"),
                      candidate_scope=str(manifest.get("role", "candidate"))),
        code_object=CodeObjectSpec(path=record["hsaco"], sha256=record["hsaco_sha256"],
                                   kernel_name=record["kernel_name"]),
        launch=artifact.launch,
        abi=artifact.abi,
        allocations=tuple(allocations),
        scratch=artifact.scratch,
        dispatch_count=1,
        base_dir=launch_dir,
    )
    capsule.validate()
    path = launch_dir / "capsule.json"
    capsule.dump(path)
    ReplayCapsule.load(path, verify_files=True)
    return path


__all__ = ["CAPTURE_SCHEMA", "build_capsule", "code_object_target", "load_manifest"]
