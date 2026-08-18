"""Validated, framework-neutral AOT kernel replay capsule.

A capsule is intentionally more explicit than a framework cache entry.  It
contains the exact code object, full lowered ABI (including hidden arguments),
launch geometry, and allocation snapshots required by a native HIP launcher.
Unknown descriptors, pointer graphs without relocations, and multi-dispatch
pipelines fail closed instead of being guessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


SCHEMA_VERSION = 1
_UINT32_MAX = (1 << 32) - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_.$][A-Za-z0-9_.$@-]*$")
_ALLOCATION_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_ABI_KINDS = {"pointer", "scalar", "implicit", "bytes", "descriptor"}
_SCALAR_TYPES = {
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}
_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
}


class CapsuleValidationError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapsuleValidationError(f"{name} must be a mapping")
    return value


def _exact_int(
    value: Any,
    name: str,
    *,
    lower: int,
    upper: int = _UINT32_MAX,
) -> int:
    if type(value) is not int:
        raise CapsuleValidationError(f"{name} must be an exact JSON integer")
    if value < lower or value > upper:
        raise CapsuleValidationError(f"{name} must be in [{lower}, {upper}]")
    return value


def _tuple3(value: Any, name: str, *, positive: bool) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CapsuleValidationError(f"{name} must contain exactly three integers")
    lower = 1 if positive else 0
    result = tuple(
        _exact_int(item, f"{name}[{index}]", lower=lower)
        for index, item in enumerate(value)
    )
    return result  # type: ignore[return-value]


def _safe_relative_path(value: Any, name: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CapsuleValidationError(f"{name} must be a capsule-relative path")
    return path.as_posix()


def _valid_sha(value: Optional[str], name: str, *, required: bool) -> Optional[str]:
    if value is None:
        if required:
            raise CapsuleValidationError(f"{name} is required")
        return None
    normalized = str(value).lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise CapsuleValidationError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class ProducerSpec:
    adapter: str
    adapter_version: str
    framework_version: str
    image_digest: str
    rocm_version: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProducerSpec":
        return cls(*(str(raw.get(key, "")).strip() for key in (
            "adapter", "adapter_version", "framework_version", "image_digest", "rocm_version"
        )))

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("adapter", self.adapter),
                ("adapter_version", self.adapter_version),
                ("framework_version", self.framework_version),
                ("image_digest", self.image_digest),
                ("rocm_version", self.rocm_version),
            )
            if not value
        ]
        if missing:
            raise CapsuleValidationError(
                "producer metadata is incomplete: " + ", ".join(missing)
            )


@dataclass(frozen=True)
class TargetSpec:
    gpu_arch: str
    xnack: bool
    code_object_version: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TargetSpec":
        return cls(str(raw.get("gpu_arch", "")), bool(raw.get("xnack", False)), int(raw.get("code_object_version", 0)))

    def validate(self) -> None:
        if not re.fullmatch(r"gfx[0-9a-z]+", self.gpu_arch):
            raise CapsuleValidationError(f"invalid target gpu_arch: {self.gpu_arch!r}")
        if self.code_object_version not in {4, 5, 6}:
            raise CapsuleValidationError("unsupported AMD code object version")


@dataclass(frozen=True)
class CaseSpec:
    task_id: str
    case_id: str
    seed: Optional[int]
    candidate_scope: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaseSpec":
        seed = raw.get("seed")
        return cls(str(raw.get("task_id", "")), str(raw.get("case_id", "")), int(seed) if seed is not None else None, str(raw.get("candidate_scope", "")))

    def validate(self) -> None:
        if not self.task_id or not self.case_id or not self.candidate_scope:
            raise CapsuleValidationError("case task_id, case_id, and candidate_scope are required")


@dataclass(frozen=True)
class CodeObjectSpec:
    path: str
    sha256: str
    kernel_name: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CodeObjectSpec":
        return cls(
            _safe_relative_path(raw.get("path", ""), "code_object.path"),
            _valid_sha(raw.get("sha256"), "code_object.sha256", required=True) or "",
            str(raw.get("kernel_name", "")),
        )

    def validate(self) -> None:
        if not _IDENTIFIER_RE.fullmatch(self.kernel_name):
            raise CapsuleValidationError(f"invalid kernel_name: {self.kernel_name!r}")


@dataclass(frozen=True)
class LaunchSpec:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_smem_bytes: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LaunchSpec":
        return cls(
            _tuple3(raw.get("grid"), "launch.grid", positive=True),
            _tuple3(raw.get("block"), "launch.block", positive=True),
            _exact_int(
                raw.get("dynamic_smem_bytes", 0),
                "launch.dynamic_smem_bytes",
                lower=0,
            ),
        )

    def validate(self) -> None:
        _tuple3(self.grid, "launch.grid", positive=True)
        _tuple3(self.block, "launch.block", positive=True)
        _exact_int(
            self.dynamic_smem_bytes,
            "launch.dynamic_smem_bytes",
            lower=0,
        )
        threads = math.prod(self.block)
        if threads > 1024:
            raise CapsuleValidationError(f"launch.block has too many threads: {threads}")


@dataclass(frozen=True)
class AbiArgument:
    index: int
    name: str
    kind: str
    c_type: str
    size: int
    ref: Optional[str] = None
    byte_offset: int = 0
    value: Any = None
    bytes_hex: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AbiArgument":
        return cls(
            index=int(raw.get("index", -1)),
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            c_type=str(raw.get("c_type", "")),
            size=int(raw.get("size", 0)),
            ref=str(raw["ref"]) if raw.get("ref") is not None else None,
            byte_offset=int(raw.get("byte_offset", 0)),
            value=raw.get("value"),
            bytes_hex=str(raw["bytes_hex"]) if raw.get("bytes_hex") is not None else None,
        )

    def validate(self, *, allow_descriptors: bool) -> None:
        if self.index < 0 or not self.name:
            raise CapsuleValidationError("ABI argument index/name are required")
        if self.kind not in _ABI_KINDS:
            raise CapsuleValidationError(f"unsupported ABI kind {self.kind!r}")
        if self.size <= 0:
            raise CapsuleValidationError(f"ABI argument {self.name} has invalid size")
        if self.byte_offset < 0:
            raise CapsuleValidationError(f"ABI argument {self.name} has negative byte_offset")
        if self.kind == "descriptor" and not allow_descriptors:
            raise CapsuleValidationError("opaque/tensor descriptor ABI is not replayable by the native launcher")
        if self.kind in {"pointer", "implicit"} and not self.ref:
            raise CapsuleValidationError(f"pointer ABI argument {self.name} is missing ref")
        if self.kind == "scalar":
            expected = _SCALAR_TYPES.get(self.c_type)
            if expected is None or expected != self.size:
                raise CapsuleValidationError(f"scalar ABI argument {self.name} has unsupported c_type/size")
            if self.value is None:
                raise CapsuleValidationError(f"scalar ABI argument {self.name} is missing value")
        if self.kind == "bytes":
            if self.bytes_hex is None or len(self.bytes_hex) != self.size * 2:
                raise CapsuleValidationError(f"byte ABI argument {self.name} has the wrong payload length")
            try:
                bytes.fromhex(self.bytes_hex)
            except ValueError as exc:
                raise CapsuleValidationError(f"byte ABI argument {self.name} is not hex") from exc


@dataclass(frozen=True)
class AllocationSpec:
    id: str
    byte_size: int
    before_blob: str
    before_sha256: str
    expected_blob: Optional[str] = None
    expected_sha256: Optional[str] = None
    alignment: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AllocationSpec":
        expected_blob = raw.get("expected_blob")
        return cls(
            id=str(raw.get("id", "")),
            byte_size=int(raw.get("byte_size", -1)),
            before_blob=_safe_relative_path(raw.get("before_blob", ""), "allocation.before_blob"),
            before_sha256=_valid_sha(raw.get("before_sha256"), "allocation.before_sha256", required=True) or "",
            expected_blob=_safe_relative_path(expected_blob, "allocation.expected_blob") if expected_blob else None,
            expected_sha256=_valid_sha(raw.get("expected_sha256"), "allocation.expected_sha256", required=bool(expected_blob)),
            alignment=int(raw.get("alignment", 1)),
        )

    def validate(self) -> None:
        if not _ALLOCATION_ID_RE.fullmatch(self.id):
            raise CapsuleValidationError(f"invalid allocation id: {self.id!r}")
        if self.byte_size <= 0:
            raise CapsuleValidationError(f"allocation {self.id} byte_size must be positive")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise CapsuleValidationError(f"allocation {self.id} alignment must be a power of two")
        if bool(self.expected_blob) != bool(self.expected_sha256):
            raise CapsuleValidationError(f"allocation {self.id} expected blob/hash must be supplied together")


@dataclass(frozen=True)
class ViewSpec:
    arg_index: int
    allocation_id: str
    byte_offset: int
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ViewSpec":
        return cls(
            int(raw.get("arg_index", -1)),
            str(raw.get("allocation_id", "")),
            int(raw.get("byte_offset", 0)),
            str(raw.get("dtype", "")),
            tuple(int(v) for v in raw.get("shape", ())),
            tuple(int(v) for v in raw.get("stride", ())),
        )

    def validate(self, allocation: AllocationSpec) -> None:
        if self.arg_index < 0 or self.byte_offset < 0:
            raise CapsuleValidationError("view arg_index/byte_offset cannot be negative")
        if self.dtype not in _DTYPE_BYTES:
            raise CapsuleValidationError(f"view has unsupported dtype {self.dtype!r}")
        if not self.shape or len(self.shape) != len(self.stride):
            raise CapsuleValidationError("view shape/stride must have equal nonzero rank")
        if any(dim < 0 for dim in self.shape) or any(step < 0 for step in self.stride):
            raise CapsuleValidationError("negative shape/stride replay is not supported")
        max_element = sum((dim - 1) * step for dim, step in zip(self.shape, self.stride) if dim)
        end = self.byte_offset + (max_element + 1) * _DTYPE_BYTES[self.dtype]
        if end > allocation.byte_size:
            raise CapsuleValidationError(f"view for arg {self.arg_index} exceeds allocation {allocation.id}")


@dataclass(frozen=True)
class RelocationSpec:
    allocation_id: str
    byte_offset: int
    target_allocation_id: str
    target_byte_offset: int = 0
    pointer_size: int = 8

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RelocationSpec":
        return cls(
            str(raw.get("allocation_id", "")),
            int(raw.get("byte_offset", -1)),
            str(raw.get("target_allocation_id", "")),
            int(raw.get("target_byte_offset", 0)),
            int(raw.get("pointer_size", 8)),
        )

    def validate(self, allocations: Mapping[str, AllocationSpec]) -> None:
        if self.allocation_id not in allocations or self.target_allocation_id not in allocations:
            raise CapsuleValidationError("relocation references an unknown allocation")
        if self.pointer_size != 8:
            raise CapsuleValidationError("only 64-bit GPU pointers are supported")
        source = allocations[self.allocation_id]
        target = allocations[self.target_allocation_id]
        if self.byte_offset < 0 or self.byte_offset + self.pointer_size > source.byte_size:
            raise CapsuleValidationError("relocation write lies outside source allocation")
        if self.target_byte_offset < 0 or self.target_byte_offset >= target.byte_size:
            raise CapsuleValidationError("relocation target lies outside target allocation")


@dataclass(frozen=True)
class ScratchSpec:
    global_bytes: int = 0
    profile_bytes: int = 0
    profile_alignment: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScratchSpec":
        return cls(int(raw.get("global_bytes", 0)), int(raw.get("profile_bytes", 0)), int(raw.get("profile_alignment", 1)))

    def validate(self) -> None:
        if self.global_bytes < 0 or self.profile_bytes < 0:
            raise CapsuleValidationError("scratch sizes cannot be negative")
        if self.profile_alignment <= 0 or self.profile_alignment & (self.profile_alignment - 1):
            raise CapsuleValidationError("profile scratch alignment must be a power of two")


@dataclass(frozen=True)
class ReplayCapsule:
    schema_version: int
    producer: ProducerSpec
    target: TargetSpec
    case: CaseSpec
    code_object: CodeObjectSpec
    launch: LaunchSpec
    abi: tuple[AbiArgument, ...]
    allocations: tuple[AllocationSpec, ...]
    views: tuple[ViewSpec, ...] = ()
    relocations: tuple[RelocationSpec, ...] = ()
    scratch: ScratchSpec = field(default_factory=ScratchSpec)
    dispatch_count: int = 1
    base_dir: Optional[Path] = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> "ReplayCapsule":
        raw = _require_mapping(raw, "capsule")
        capsule = cls(
            schema_version=int(raw.get("schema_version", 0)),
            producer=ProducerSpec.from_dict(_require_mapping(raw.get("producer"), "producer")),
            target=TargetSpec.from_dict(_require_mapping(raw.get("target"), "target")),
            case=CaseSpec.from_dict(_require_mapping(raw.get("case"), "case")),
            code_object=CodeObjectSpec.from_dict(_require_mapping(raw.get("code_object"), "code_object")),
            launch=LaunchSpec.from_dict(_require_mapping(raw.get("launch"), "launch")),
            abi=tuple(AbiArgument.from_dict(_require_mapping(v, "abi item")) for v in raw.get("abi", ())),
            allocations=tuple(AllocationSpec.from_dict(_require_mapping(v, "allocation")) for v in raw.get("allocations", ())),
            views=tuple(ViewSpec.from_dict(_require_mapping(v, "view")) for v in raw.get("views", ())),
            relocations=tuple(RelocationSpec.from_dict(_require_mapping(v, "relocation")) for v in raw.get("relocations", ())),
            scratch=ScratchSpec.from_dict(_require_mapping(raw.get("scratch", {}), "scratch")),
            dispatch_count=int(raw.get("dispatch_count", 1)),
            base_dir=base_dir.resolve() if base_dir else None,
        )
        capsule.validate()
        return capsule

    @classmethod
    def load(cls, path: Path, *, verify_files: bool = True) -> "ReplayCapsule":
        path = path.resolve()
        capsule = cls.from_dict(json.loads(path.read_text(encoding="utf-8")), base_dir=path.parent)
        if verify_files:
            capsule.verify_files()
        return capsule

    def validate(self, *, allow_descriptors: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CapsuleValidationError(f"unsupported capsule schema_version {self.schema_version}")
        if self.dispatch_count != 1:
            raise CapsuleValidationError("native replay MVP supports exactly one kernel dispatch")
        self.producer.validate()
        self.target.validate()
        self.case.validate()
        self.code_object.validate()
        self.launch.validate()
        self.scratch.validate()
        if not self.abi:
            raise CapsuleValidationError("capsule ABI cannot be empty")
        if [arg.index for arg in self.abi] != list(range(len(self.abi))):
            raise CapsuleValidationError("ABI argument indices must be contiguous and ordered")
        if len({arg.name for arg in self.abi}) != len(self.abi):
            raise CapsuleValidationError("ABI argument names must be unique")
        for arg in self.abi:
            arg.validate(allow_descriptors=allow_descriptors)

        allocations = {item.id: item for item in self.allocations}
        if len(allocations) != len(self.allocations):
            raise CapsuleValidationError("allocation ids must be unique")
        for allocation in self.allocations:
            allocation.validate()
        if not any(allocation.expected_blob for allocation in self.allocations):
            raise CapsuleValidationError(
                "replay capsule requires at least one golden expected output"
            )
        for arg in self.abi:
            if arg.kind == "pointer" and arg.ref not in allocations:
                raise CapsuleValidationError(f"ABI pointer {arg.name} references unknown allocation {arg.ref!r}")
            if arg.kind == "pointer" and arg.ref is not None:
                if arg.byte_offset >= allocations[arg.ref].byte_size:
                    raise CapsuleValidationError(f"ABI pointer {arg.name} offset exceeds allocation")
            if arg.kind == "implicit" and arg.ref not in {"scratch:global", "scratch:profile", "null"}:
                raise CapsuleValidationError(f"implicit ABI argument {arg.name} has unknown ref {arg.ref!r}")

        seen_view_args: set[int] = set()
        for view in self.views:
            if view.arg_index in seen_view_args:
                raise CapsuleValidationError(f"multiple views describe ABI argument {view.arg_index}")
            seen_view_args.add(view.arg_index)
            allocation = allocations.get(view.allocation_id)
            if allocation is None:
                raise CapsuleValidationError("view references an unknown allocation")
            if view.arg_index >= len(self.abi) or self.abi[view.arg_index].kind != "pointer":
                raise CapsuleValidationError("view must describe a pointer ABI argument")
            if self.abi[view.arg_index].ref != view.allocation_id:
                raise CapsuleValidationError("view allocation disagrees with ABI pointer ref")
            view.validate(allocation)
        for relocation in self.relocations:
            relocation.validate(allocations)

    def resolve(self, relative: str) -> Path:
        if self.base_dir is None:
            raise CapsuleValidationError("capsule has no base_dir; load it from a file before replay")
        resolved = (self.base_dir / relative).resolve()
        if not resolved.is_relative_to(self.base_dir):
            raise CapsuleValidationError(f"capsule path escapes base directory: {relative}")
        return resolved

    def verify_files(self) -> None:
        code = self.resolve(self.code_object.path)
        if not code.is_file() or _sha256_file(code) != self.code_object.sha256:
            raise CapsuleValidationError("code object is missing or its SHA-256 does not match")
        if code.read_bytes()[:4] != b"\x7fELF":
            raise CapsuleValidationError("code object is not an ELF/HSACO file")
        for allocation in self.allocations:
            before = self.resolve(allocation.before_blob)
            if not before.is_file() or before.stat().st_size != allocation.byte_size:
                raise CapsuleValidationError(f"allocation {allocation.id} before blob size mismatch")
            if _sha256_file(before) != allocation.before_sha256:
                raise CapsuleValidationError(f"allocation {allocation.id} before blob hash mismatch")
            if allocation.expected_blob:
                expected = self.resolve(allocation.expected_blob)
                if not expected.is_file() or expected.stat().st_size != allocation.byte_size:
                    raise CapsuleValidationError(f"allocation {allocation.id} expected blob size mismatch")
                if _sha256_file(expected) != allocation.expected_sha256:
                    raise CapsuleValidationError(f"allocation {allocation.id} expected blob hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": vars(self.producer),
            "target": vars(self.target),
            "case": vars(self.case),
            "code_object": vars(self.code_object),
            "launch": {"grid": list(self.launch.grid), "block": list(self.launch.block), "dynamic_smem_bytes": self.launch.dynamic_smem_bytes},
            "abi": [vars(v) for v in self.abi],
            "allocations": [vars(v) for v in self.allocations],
            "views": [{**vars(v), "shape": list(v.shape), "stride": list(v.stride)} for v in self.views],
            "relocations": [vars(v) for v in self.relocations],
            "scratch": vars(self.scratch),
            "dispatch_count": self.dispatch_count,
        }

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "AbiArgument",
    "AllocationSpec",
    "CapsuleValidationError",
    "CaseSpec",
    "CodeObjectSpec",
    "LaunchSpec",
    "ProducerSpec",
    "RelocationSpec",
    "ReplayCapsule",
    "SCHEMA_VERSION",
    "ScratchSpec",
    "TargetSpec",
    "ViewSpec",
]
