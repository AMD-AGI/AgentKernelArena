"""Scoped HIP source rebuild recipes for sanitizer lanes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# GPU ASan's runtime environment and HIP build flags, shared by the gpu_asan
# plugin and its HIP rebuild recipe.
GPU_ASAN_ENV = {
    "HSA_XNACK": "1",
    "HSA_DISABLE_FRAGMENT_ALLOCATOR": "1",
    "AMD_PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
    "PYTORCH_NO_HIP_MEMORY_CACHING": "1",
    "AMDGCN_USE_BUFFER_OPS": "0",
    "ASAN_OPTIONS": "detect_leaks=0,alloc_dealloc_mismatch=0",
}


def hip_asan_build_flags(target_arch: str) -> tuple[str, ...]:
    arch = target_arch.split(":", 1)[0].strip()
    if not arch.startswith("gfx"):
        raise ValueError(f"invalid AMD GPU architecture: {target_arch!r}")
    return ("-fsanitize=address", "-shared-libsan", f"--offload-arch={arch}:xnack+")


@dataclass(frozen=True)
class HipSourceBuildRecipe:
    tool: str
    source: Path
    output: Path
    compiler: Path
    flags: tuple[str, ...]
    env: dict[str, str]

    @property
    def command(self) -> tuple[str, ...]:
        return (str(self.compiler), "-O2", *self.flags, str(self.source), "-o", str(self.output))


def gpu_asan_recipe(
    source: Path,
    output: Path,
    *,
    target_arch: str,
    compiler: Path = Path("/opt/rocm/bin/hipcc"),
) -> HipSourceBuildRecipe:
    return HipSourceBuildRecipe(
        tool="gpu_asan",
        source=source,
        output=output,
        compiler=compiler,
        flags=hip_asan_build_flags(target_arch),
        env=dict(GPU_ASAN_ENV),
    )


def hip_fpsan_recipe(
    source: Path,
    output: Path,
    *,
    include_dir: Path,
    target_arch: str,
    compiler: Path = Path("/opt/rocm/bin/hipcc"),
) -> HipSourceBuildRecipe:
    arch = target_arch.split(":", 1)[0]
    return HipSourceBuildRecipe(
        tool="hip_fpsan",
        source=source,
        output=output,
        compiler=compiler,
        flags=(f"-I{include_dir}", "-DAKA_HIP_FPSAN=1", f"--offload-arch={arch}"),
        env={"AKA_HIP_FPSAN": "1", "FPSAN_INCLUDE_DIR": str(include_dir)},
    )


__all__ = ["HipSourceBuildRecipe", "gpu_asan_recipe", "hip_fpsan_recipe"]
