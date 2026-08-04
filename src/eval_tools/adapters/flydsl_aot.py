"""Fail-closed extraction helpers for FlyDSL ``CompiledArtifact`` objects."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .replay_capsule import CapsuleValidationError, LaunchSpec


_CONST_RE = re.compile(r"(%[A-Za-z0-9_.$-]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*(?:index|i\d+)")
_LAUNCH_RE = re.compile(r"gpu\.launch_func\s+@(?:[A-Za-z0-9_.$-]+::)?@(?P<kernel>[A-Za-z0-9_.$-]+)(?P<body>.*?)(?=\n\s*[%}]|$)", re.S)
_BIN_RE = re.compile(r'\bbin\s*=\s*"((?:\\.|[^"\\])*)"', re.S)


@dataclass(frozen=True)
class FlyDslStaticLaunch:
    kernel_name: str
    launch: LaunchSpec


@dataclass(frozen=True)
class FlyDslAotArtifact:
    hsaco_path: Path
    hsaco_sha256: str
    static_launch: FlyDslStaticLaunch
    host_entry: str


def _resolve_dim(token: str, constants: Mapping[str, int]) -> int:
    clean = token.strip()
    if clean in constants:
        return constants[clean]
    if re.fullmatch(r"\d+", clean):
        return int(clean)
    raise CapsuleValidationError(f"FlyDSL launch dimension is dynamic or unknown: {clean!r}")


def _parse_dims(body: str, label: str, constants: Mapping[str, int]) -> tuple[int, int, int]:
    patterns = (
        rf"{label}\s+in\s*\(([^)]*)\)",
        rf"{label}\s*\(([^)]*)\)",
    )
    match = next((m for p in patterns if (m := re.search(p, body))), None)
    if match is None:
        raise CapsuleValidationError(f"FlyDSL gpu.launch_func is missing {label}")
    tokens = [part.strip() for part in match.group(1).split(",")]
    if len(tokens) != 3:
        raise CapsuleValidationError(f"FlyDSL {label} must contain exactly three values")
    return tuple(_resolve_dim(token, constants) for token in tokens)  # type: ignore[return-value]


def parse_flydsl_static_launch(source_ir: str) -> FlyDslStaticLaunch:
    constants = {name: int(value) for name, value in _CONST_RE.findall(source_ir)}
    launches = list(_LAUNCH_RE.finditer(source_ir))
    if len(launches) != 1:
        raise CapsuleValidationError(f"FlyDSL replay MVP requires one gpu.launch_func, found {len(launches)}")
    match = launches[0]
    body = match.group("body")
    grid = _parse_dims(body, "blocks", constants)
    block = _parse_dims(body, "threads", constants)
    smem = 0
    smem_match = re.search(r"dynamic_shared_memory_size\s+(%[A-Za-z0-9_.$-]+|\d+)", body)
    if smem_match:
        smem = _resolve_dim(smem_match.group(1), constants)
    launch = LaunchSpec(grid, block, smem)
    launch.validate()
    return FlyDslStaticLaunch(match.group("kernel"), launch)


def _decode_mlir_string(value: str) -> bytes:
    result = bytearray()
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\":
            result.extend(char.encode("utf-8"))
            index += 1
            continue
        if index + 2 < len(value) and re.fullmatch(r"[0-9A-Fa-f]{2}", value[index + 1 : index + 3]):
            result.append(int(value[index + 1 : index + 3], 16))
            index += 3
            continue
        if index + 1 >= len(value):
            raise CapsuleValidationError("unterminated MLIR string escape")
        escapes = {"n": 10, "r": 13, "t": 9, "\\": 92, '"': 34}
        escaped = value[index + 1]
        if escaped not in escapes:
            raise CapsuleValidationError(f"unsupported MLIR string escape \\{escaped}")
        result.append(escapes[escaped])
        index += 2
    return bytes(result)


def extract_embedded_hsaco(ir_text: str) -> bytes:
    objects = []
    for match in _BIN_RE.finditer(ir_text):
        decoded = _decode_mlir_string(match.group(1))
        if decoded.startswith(b"\x7fELF"):
            objects.append(decoded)
    if len(objects) != 1:
        raise CapsuleValidationError(f"FlyDSL artifact must contain exactly one embedded ELF, found {len(objects)}")
    return objects[0]


def pack_dynamic_layout(
    shape: Sequence[int],
    stride: Sequence[int],
    *,
    dynamic_shape_indices: Sequence[int],
    dynamic_stride_indices: Sequence[int],
    use_32bit_stride: bool,
) -> bytes:
    """Match FlyDSL's no-padding ``_LayoutPlan`` C ABI exactly."""

    shape_values = [int(shape[index]) for index in dynamic_shape_indices]
    stride_values = [int(stride[index]) for index in dynamic_stride_indices]
    fmt = "<" + "i" * len(shape_values) + ("i" if use_32bit_stride else "q") * len(stride_values)
    try:
        return struct.pack(fmt, *shape_values, *stride_values)
    except struct.error as exc:
        raise CapsuleValidationError(f"FlyDSL dynamic layout value does not fit ABI: {exc}") from exc


def extract_flydsl_aot(artifact: object, output_dir: Path) -> FlyDslAotArtifact:
    ir_text = getattr(artifact, "_ir_text", None)
    source_ir = getattr(artifact, "_source_ir", None)
    entry = str(getattr(artifact, "_entry", ""))
    if not isinstance(ir_text, str) or not isinstance(source_ir, str) or not entry:
        raise CapsuleValidationError("FlyDSL CompiledArtifact IR/source/entry are unavailable")
    if "#fly.explicit_module" in source_ir or bool(getattr(artifact, "_uses_explicit_module", False)):
        raise CapsuleValidationError("FlyDSL extern-linked/post-load artifacts are not replayable yet")
    static_launch = parse_flydsl_static_launch(source_ir)
    hsaco = extract_embedded_hsaco(ir_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    hsaco_path = output_dir / f"{static_launch.kernel_name}.hsaco"
    hsaco_path.write_bytes(hsaco)
    return FlyDslAotArtifact(hsaco_path, hashlib.sha256(hsaco).hexdigest(), static_launch, entry)


__all__ = [
    "FlyDslAotArtifact",
    "FlyDslStaticLaunch",
    "extract_embedded_hsaco",
    "extract_flydsl_aot",
    "pack_dynamic_layout",
    "parse_flydsl_static_launch",
]
