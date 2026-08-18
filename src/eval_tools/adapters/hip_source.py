"""Scoped HIP source rebuild recipes for sanitizer lanes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..plugins.gpu_asan import GPU_ASAN_ENV, hip_asan_build_flags


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
