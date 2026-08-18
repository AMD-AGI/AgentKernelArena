"""Generate a minimal native HIP module launcher from a replay capsule."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .replay_capsule import AbiArgument, CapsuleValidationError, ReplayCapsule


def _cpp_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _byte_initializer(value: bytes) -> str:
    return ", ".join(f"0x{byte:02x}" for byte in value)


def _scalar_bytes(arg: AbiArgument) -> bytes:
    if arg.kind == "bytes":
        assert arg.bytes_hex is not None
        return bytes.fromhex(arg.bytes_hex)
    if arg.kind != "scalar":
        raise CapsuleValidationError(f"{arg.name} is not a scalar/bytes argument")
    value = arg.value
    if arg.c_type.startswith("i"):
        bits = int(arg.c_type[1:])
        integer = int(value)
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if integer < lower or integer > upper:
            raise CapsuleValidationError(f"scalar {arg.name} is out of range for {arg.c_type}")
        return integer.to_bytes(arg.size, "little", signed=True)
    if arg.c_type.startswith("u"):
        bits = int(arg.c_type[1:])
        integer = int(value)
        if integer < 0 or integer >= 1 << bits:
            raise CapsuleValidationError(f"scalar {arg.name} is out of range for {arg.c_type}")
        return integer.to_bytes(arg.size, "little", signed=False)
    if arg.c_type == "f16":
        return struct.pack("<e", float(value))
    if arg.c_type == "bf16":
        # JSON has no native bf16 representation.  Require the exact u16 payload.
        payload = int(value)
        if payload < 0 or payload > 0xFFFF:
            raise CapsuleValidationError(f"bf16 scalar {arg.name} must be a u16 payload")
        return payload.to_bytes(2, "little")
    if arg.c_type == "f32":
        return struct.pack("<f", float(value))
    if arg.c_type == "f64":
        return struct.pack("<d", float(value))
    raise CapsuleValidationError(f"native launcher cannot pack scalar type {arg.c_type!r}")


@dataclass(frozen=True)
class NativeLauncherPlan:
    source_path: Path
    binary_path: Path
    compile_command: tuple[str, ...]
    run_command: tuple[str, ...]


class NativeLauncherContract:
    """The subset of capsules supported by the generated HIP launcher."""

    version = "3"

    def validate(self, capsule: ReplayCapsule) -> None:
        capsule.validate(allow_descriptors=False)
        if capsule.dispatch_count != 1:
            raise CapsuleValidationError("native launcher supports one dispatch only")
        if any(arg.kind == "descriptor" for arg in capsule.abi):
            raise CapsuleValidationError("native launcher does not support tensor descriptors")

    def render(self, capsule: ReplayCapsule) -> str:
        self.validate(capsule)
        capsule.verify_files()
        allocation_index = {allocation.id: i for i, allocation in enumerate(capsule.allocations)}

        lines = [
            "// Generated from an AgentKernelArena replay capsule. Do not edit.",
            "#include <hip/hip_runtime.h>",
            "#include <hip/hip_runtime_api.h>",
            "#include <cstdint>",
            "#include <cstdio>",
            "#include <cstring>",
            "#include <fstream>",
            "#include <iterator>",
            "#include <string>",
            "#include <vector>",
            "",
            "static bool hip_ok(hipError_t value, const char* expression) {",
            "  if (value == hipSuccess) return true;",
            '  std::fprintf(stderr, "HIP failure: %s: %s\\n", expression, hipGetErrorString(value));',
            "  return false;",
            "}",
            "#define HIP_OK(expr) do { if (!hip_ok((expr), #expr)) return 2; } while (0)",
            "",
            "static std::string capsule_path(const std::string& root, const char* relative) {",
            '  return root + "/" + relative;',
            "}",
            "",
            "static std::vector<unsigned char> read_exact(const std::string& path, std::size_t expected) {",
            "  std::ifstream input(path, std::ios::binary);",
            "  if (!input) return {};",
            "  std::vector<unsigned char> result((std::istreambuf_iterator<char>(input)), {});",
            "  if (result.size() != expected) return {};",
            "  return result;",
            "}",
            "",
            "int main(int argc, char** argv) {",
            '  if (argc != 2) { std::fprintf(stderr, "usage: replay_launcher CAPSULE_ROOT\\n"); return 5; }',
            "  const std::string capsule_root(argv[1]);",
            f"  const std::string code_object_path = capsule_path(capsule_root, {_cpp_string(capsule.code_object.path)});",
            "  hipDeviceProp_t device_properties{};",
            "  HIP_OK(hipGetDeviceProperties(&device_properties, 0));",
            (
                "  if ("
                f"{capsule.launch.grid[0]}ULL > static_cast<std::uint64_t>(device_properties.maxGridSize[0]) || "
                f"{capsule.launch.grid[1]}ULL > static_cast<std::uint64_t>(device_properties.maxGridSize[1]) || "
                f"{capsule.launch.grid[2]}ULL > static_cast<std::uint64_t>(device_properties.maxGridSize[2]) || "
                f"{capsule.launch.block[0]}ULL > static_cast<std::uint64_t>(device_properties.maxThreadsDim[0]) || "
                f"{capsule.launch.block[1]}ULL > static_cast<std::uint64_t>(device_properties.maxThreadsDim[1]) || "
                f"{capsule.launch.block[2]}ULL > static_cast<std::uint64_t>(device_properties.maxThreadsDim[2]) || "
                f"{capsule.launch.block[0] * capsule.launch.block[1] * capsule.launch.block[2]}ULL > static_cast<std::uint64_t>(device_properties.maxThreadsPerBlock) || "
                f"{capsule.launch.dynamic_smem_bytes}ULL > static_cast<std::uint64_t>(device_properties.sharedMemPerBlock)) {{"
            ),
            '    std::fprintf(stderr, "capsule launch geometry exceeds device limits\\n");',
            "    return 6;",
            "  }",
            "  hipModule_t module = nullptr;",
            "  hipFunction_t function = nullptr;",
            "  HIP_OK(hipModuleLoad(&module, code_object_path.c_str()));",
            f"  HIP_OK(hipModuleGetFunction(&function, module, {_cpp_string(capsule.code_object.kernel_name)}));",
        ]

        for i, allocation in enumerate(capsule.allocations):
            lines.extend(
                [
                    f"  void* device_{i} = nullptr;",
                    f"  HIP_OK(hipMalloc(&device_{i}, {allocation.byte_size}));",
                    f"  auto host_{i} = read_exact(capsule_path(capsule_root, {_cpp_string(allocation.before_blob)}), {allocation.byte_size});",
                    f"  if (host_{i}.size() != {allocation.byte_size}) {{ std::fprintf(stderr, \"invalid input blob {i}\\n\"); return 3; }}",
                ]
            )

        for relocation in capsule.relocations:
            source = allocation_index[relocation.allocation_id]
            target = allocation_index[relocation.target_allocation_id]
            lines.extend(
                [
                    f"  std::uintptr_t relocation_{source}_{relocation.byte_offset} =",
                    f"      reinterpret_cast<std::uintptr_t>(device_{target}) + {relocation.target_byte_offset};",
                    f"  std::memcpy(host_{source}.data() + {relocation.byte_offset}, &relocation_{source}_{relocation.byte_offset}, 8);",
                ]
            )

        for i, allocation in enumerate(capsule.allocations):
            lines.append(
                f"  HIP_OK(hipMemcpy(device_{i}, host_{i}.data(), {allocation.byte_size}, hipMemcpyHostToDevice));"
            )

        lines.extend(["  void* global_scratch = nullptr;", "  void* profile_scratch = nullptr;"])
        if capsule.scratch.global_bytes:
            lines.append(f"  HIP_OK(hipMalloc(&global_scratch, {capsule.scratch.global_bytes}));")
        if capsule.scratch.profile_bytes:
            lines.append(f"  HIP_OK(hipMalloc(&profile_scratch, {capsule.scratch.profile_bytes}));")

        for arg in capsule.abi:
            if arg.kind == "pointer":
                assert arg.ref is not None
                allocation = allocation_index[arg.ref]
                lines.append(
                    f"  void* arg_{arg.index} = static_cast<unsigned char*>(device_{allocation}) + {arg.byte_offset};"
                )
            elif arg.kind == "implicit":
                ref = {"scratch:global": "global_scratch", "scratch:profile": "profile_scratch", "null": "nullptr"}[arg.ref or "null"]
                lines.append(f"  void* arg_{arg.index} = {ref};")
            else:
                payload = _scalar_bytes(arg)
                lines.append(
                    f"  unsigned char arg_{arg.index}[{len(payload)}] = {{{_byte_initializer(payload)}}};"
                )

        params = []
        for arg in capsule.abi:
            params.append(f"&arg_{arg.index}" if arg.kind in {"pointer", "implicit"} else f"arg_{arg.index}")
        lines.extend(
            [
                f"  void* params[{len(params)}] = {{{', '.join(params)}}};",
                f'  std::puts("AKA_REPLAY_DISPATCH kernel={capsule.code_object.kernel_name}");',
                "  HIP_OK(hipModuleLaunchKernel(",
                f"      function, {capsule.launch.grid[0]}, {capsule.launch.grid[1]}, {capsule.launch.grid[2]},",
                f"      {capsule.launch.block[0]}, {capsule.launch.block[1]}, {capsule.launch.block[2]},",
                f"      {capsule.launch.dynamic_smem_bytes}, nullptr, params, nullptr));",
                "  HIP_OK(hipDeviceSynchronize());",
            ]
        )

        for i, allocation in enumerate(capsule.allocations):
            if allocation.expected_blob:
                lines.extend(
                    [
                        f"  std::vector<unsigned char> actual_{i}({allocation.byte_size});",
                        f"  HIP_OK(hipMemcpy(actual_{i}.data(), device_{i}, {allocation.byte_size}, hipMemcpyDeviceToHost));",
                        f"  auto expected_{i} = read_exact(capsule_path(capsule_root, {_cpp_string(allocation.expected_blob)}), {allocation.byte_size});",
                        f"  if (expected_{i}.size() != {allocation.byte_size} || actual_{i} != expected_{i}) {{",
                        f'    std::fprintf(stderr, "output allocation {allocation.id} mismatched golden blob\\n");',
                        "    return 4;",
                        "  }",
                    ]
                )

        for i in range(len(capsule.allocations)):
            lines.append(f"  HIP_OK(hipFree(device_{i}));")
        lines.extend(
            [
                "  if (global_scratch) HIP_OK(hipFree(global_scratch));",
                "  if (profile_scratch) HIP_OK(hipFree(profile_scratch));",
                "  HIP_OK(hipModuleUnload(module));",
                '  std::puts("AKA_REPLAY_RESULT pass");',
                "  return 0;",
                "}",
                "",
            ]
        )
        return "\n".join(lines)

    def materialize(
        self,
        capsule_path: Path,
        output_dir: Path,
        *,
        hipcc: Path = Path("/opt/rocm/bin/hipcc"),
    ) -> NativeLauncherPlan:
        capsule = ReplayCapsule.load(capsule_path, verify_files=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = output_dir / "replay_launcher.cpp"
        binary = output_dir / "replay_launcher"
        source.write_text(self.render(capsule), encoding="utf-8")
        if capsule.base_dir is None:  # load() above always sets this; keep the plan contract explicit.
            raise CapsuleValidationError("capsule has no base directory")
        return NativeLauncherPlan(
            source_path=source,
            binary_path=binary,
            compile_command=(str(hipcc), "-std=c++17", "-O2", str(source), "-o", str(binary)),
            run_command=(str(binary), str(capsule.base_dir)),
        )


def render_native_launcher(capsule: ReplayCapsule) -> str:
    return NativeLauncherContract().render(capsule)


__all__ = ["NativeLauncherContract", "NativeLauncherPlan", "render_native_launcher"]
