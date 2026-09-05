"""Extraction helpers for Triton ``CompiledKernel`` AOT replay."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .replay_capsule import AbiArgument, CapsuleValidationError, LaunchSpec, ScratchSpec


_TRITON_SCALARS = {
    "i1": ("u8", 1),
    "i8": ("i8", 1),
    "u8": ("u8", 1),
    "i16": ("i16", 2),
    "u16": ("u16", 2),
    "fp16": ("f16", 2),
    "f16": ("f16", 2),
    "bf16": ("bf16", 2),
    "i32": ("i32", 4),
    "u32": ("u32", 4),
    "fp32": ("f32", 4),
    "f32": ("f32", 4),
    "i64": ("i64", 8),
    "u64": ("u64", 8),
    "fp64": ("f64", 8),
    "f64": ("f64", 8),
}


def _metadata_value(metadata: object, name: str, default: Any = 0) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


@dataclass(frozen=True)
class TritonAotArtifact:
    hsaco_path: Path
    hsaco_sha256: str
    kernel_name: str
    launch: LaunchSpec
    abi: tuple[AbiArgument, ...]
    scratch: ScratchSpec


def build_triton_abi(
    signature: Mapping[int, str],
    *,
    constants: Mapping[Any, Any] = {},
    pointer_bindings: Mapping[int, tuple[str, int]] = {},
    scalar_values: Mapping[int, Any] = {},
) -> tuple[AbiArgument, ...]:
    """Build the *lowered AMD launcher ABI*, including its two hidden args."""

    constant_indices: set[int] = set()
    for key in constants:
        if isinstance(key, int):
            constant_indices.add(key)
        elif isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], int):
            constant_indices.add(key[0])

    result: list[AbiArgument] = []
    for source_index, raw_type in sorted(signature.items(), key=lambda item: int(item[0])):
        index = int(source_index)
        ty = str(raw_type).lower()
        if index in constant_indices or ty == "constexpr":
            continue
        abi_index = len(result)
        if ty.startswith("tensordesc"):
            raise CapsuleValidationError("Triton tensor descriptors are not supported by the AOT replay MVP")
        if ty.startswith("*"):
            binding = pointer_bindings.get(index)
            if binding is None:
                raise CapsuleValidationError(f"Triton pointer argument {index} has no captured allocation binding")
            result.append(AbiArgument(abi_index, f"arg{index}", "pointer", "pointer", 8, binding[0], binding[1]))
            continue
        scalar = _TRITON_SCALARS.get(ty)
        if scalar is None:
            raise CapsuleValidationError(f"unsupported Triton ABI type {raw_type!r}")
        if index not in scalar_values:
            raise CapsuleValidationError(f"Triton scalar argument {index} has no captured value")
        result.append(AbiArgument(abi_index, f"arg{index}", "scalar", scalar[0], scalar[1], value=scalar_values[index]))

    # AMD's generated launcher appends these even when the kernel does not use
    # scratch.  Omitting them shifts the kernarg layout and corrupts replay.
    result.append(AbiArgument(len(result), "global_scratch", "implicit", "pointer", 8, "scratch:global"))
    result.append(AbiArgument(len(result), "profile_scratch", "implicit", "pointer", 8, "scratch:profile"))
    return tuple(result)


def extract_triton_aot(
    compiled_kernel: object,
    output_dir: Path,
    *,
    grid: Sequence[int],
    pointer_bindings: Mapping[int, tuple[str, int]],
    scalar_values: Mapping[int, Any],
) -> TritonAotArtifact:
    asm = getattr(compiled_kernel, "asm", None)
    if not isinstance(asm, Mapping) or not isinstance(asm.get("hsaco"), (bytes, bytearray)):
        raise CapsuleValidationError("CompiledKernel.asm['hsaco'] bytes are unavailable")
    name = str(getattr(compiled_kernel, "name", ""))
    if not name:
        raise CapsuleValidationError("CompiledKernel kernel name is unavailable")
    metadata = getattr(compiled_kernel, "metadata", None)
    if metadata is None:
        raise CapsuleValidationError("CompiledKernel metadata is unavailable")
    src = getattr(compiled_kernel, "src", None)
    signature = getattr(src, "signature", None)
    constants = getattr(src, "constants", {})
    if not isinstance(signature, Mapping):
        raise CapsuleValidationError("CompiledKernel source signature is unavailable")

    dims = tuple(int(v) for v in grid)
    if len(dims) > 3 or not dims:
        raise CapsuleValidationError("Triton grid must have one to three dimensions")
    grid3 = dims + (1,) * (3 - len(dims))
    num_warps = int(_metadata_value(metadata, "num_warps", 0))
    warp_size = int(_metadata_value(metadata, "warp_size", 64))
    if num_warps <= 0 or warp_size <= 0:
        raise CapsuleValidationError("Triton num_warps/warp_size metadata is invalid")
    launch = LaunchSpec(grid3, (num_warps * warp_size, 1, 1), int(_metadata_value(metadata, "shared", 0)))
    launch.validate()
    profile_per_grid = int(_metadata_value(metadata, "profile_scratch_size", 0))
    scratch = ScratchSpec(
        global_bytes=0,
        profile_bytes=profile_per_grid * grid3[0] * grid3[1] * grid3[2],
        profile_alignment=int(_metadata_value(metadata, "profile_scratch_align", 1)),
    )
    scratch.validate()
    abi = build_triton_abi(
        {int(k): str(v) for k, v in signature.items()},
        constants=constants if isinstance(constants, Mapping) else {},
        pointer_bindings=pointer_bindings,
        scalar_values=scalar_values,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hsaco_path = output_dir / f"{name}.hsaco"
    hsaco = bytes(asm["hsaco"])
    hsaco_path.write_bytes(hsaco)
    return TritonAotArtifact(
        hsaco_path,
        hashlib.sha256(hsaco).hexdigest(),
        name,
        launch,
        abi,
        scratch,
    )


__all__ = ["TritonAotArtifact", "build_triton_abi", "extract_triton_aot"]
