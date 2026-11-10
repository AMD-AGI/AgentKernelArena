# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Assemble, load and launch AMDGPU assembly kernels for ``target_language: asm``.

Adapted from the assembly candidate runner of the AgentKernelArena assembly
task copies (``asm_candidate.py``). An assembly implementation is a flat Python
package like the other Python-family languages: ``kernel.py`` defines
``run(*, inputs, outputs)`` and launches kernels written in ``.s`` files beside
it. The package imports this module as ``aka_asm`` (the loader registers that
name) and uses only:

    kernel = aka_asm.load("gemm.s", "gemm_kernel")      # at import time
    kernel.launch(grid=(gx, gy, gz), block=(256, 1, 1),
                  args=[a, b, out, aka_asm.u32(m)], shared_bytes=0)

``load`` assembles the file with the image's ROCm ``clang`` for the current GPU
architecture, links it with ``ld.lld`` into a code object, caches the result by
content and loads it with ``hipModuleLoadData``. ``launch`` packs the arguments
in order with natural alignment (tensors as 64-bit device pointers, scalars
only through the typed wrappers below) and calls ``hipModuleLaunchKernel`` on
the current PyTorch stream, which is what CUDA-graph capture records.

The kernel argument layout is the kernel's own contract: the ``.s`` file's
``.amdgpu_metadata`` must declare arguments that match what ``launch`` packs.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ALIAS = "aka_asm"
SOURCE_SUFFIX = ".s"
# .incbin and .include pull other files or raw bytes into the code object; a
# kernel's whole source must be in the .s file the evaluator reads and audits.
FORBIDDEN_DIRECTIVE = re.compile(r"^[ \t]*\.(?:incbin|include)\b", re.IGNORECASE | re.MULTILINE)
SYMBOL = re.compile(r"^[A-Za-z_.$][A-Za-z0-9_.$]*$")
MAX_THREADS_PER_BLOCK = 1024
MAX_SHARED_BYTES = 160 * 1024          # LDS per workgroup on gfx950
MAX_GRID = 2 ** 31 - 1

# hipModuleLaunchKernel reads these markers from its ``extra`` array.
HIP_LAUNCH_PARAM_BUFFER_POINTER = 1
HIP_LAUNCH_PARAM_BUFFER_SIZE = 2
HIP_LAUNCH_PARAM_END = 3

# Build evidence, copied into the action report: tool versions and the SHA256
# of every source and code object.
BUILD_EVIDENCE: dict[str, Any] = {"files": []}
# Every launch since import, by kernel symbol; the launch observer reads its length.
LAUNCHES: list[str] = []
# SHA256 of the code objects an asm baseline vendors; the loader sets it from
# the contract. A prebuilt code object loads only if its hash is listed.
ALLOWED_CODE_OBJECTS: frozenset = frozenset()
CODE_OBJECT_SUFFIX = ".co"


class AsmError(RuntimeError):
    """An assembly source, its build or a launch is invalid."""


class _Scalar:
    __slots__ = ("format", "value")

    def __init__(self, format: str, value: Any):
        self.format, self.value = format, value


def u32(value: int) -> _Scalar:
    return _Scalar("I", int(value))


def i32(value: int) -> _Scalar:
    return _Scalar("i", int(value))


def u64(value: int) -> _Scalar:
    return _Scalar("Q", int(value))


def i64(value: int) -> _Scalar:
    return _Scalar("q", int(value))


def f32(value: float) -> _Scalar:
    return _Scalar("f", float(value))


def pack_arguments(args: list) -> bytes:
    """Kernel arguments in order, each at its natural alignment; the total padded to 8 bytes."""
    data = bytearray()
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if arg.device.type != "cuda":
                raise AsmError(f"argument {index} is a {arg.device.type} tensor; pass GPU tensors")
            fmt, value = "Q", arg.data_ptr()
        elif isinstance(arg, _Scalar):
            fmt, value = arg.format, arg.value
        else:
            raise AsmError(f"argument {index} is a {type(arg).__name__}; pass a tensor or a scalar "
                           "wrapped in aka_asm.u32/i32/u64/i64/f32")
        size = struct.calcsize(fmt)
        data.extend(b"\0" * (-len(data) % size))
        try:
            data.extend(struct.pack("<" + fmt, value))
        except struct.error as exc:
            raise AsmError(f"argument {index} does not fit its type: {exc}") from exc
    data.extend(b"\0" * (-len(data) % 8))
    return bytes(data)


def _triple(value, what: str, limit: int) -> tuple[int, int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(type(x) is not int or x <= 0 or x > limit for x in value)):
        raise AsmError(f"{what} must be three positive integers no larger than {limit}")
    return tuple(value)


def _toolchain() -> tuple[Path, Path]:
    roots = [os.environ.get("ROCM_PATH"), "/opt/rocm"]
    try:
        from torch.utils import cpp_extension

        roots.insert(1, cpp_extension.ROCM_HOME)
    except Exception:
        pass
    for root in filter(None, roots):
        clang, lld = Path(root, "llvm/bin/clang"), Path(root, "llvm/bin/ld.lld")
        if clang.is_file() and lld.is_file():
            return clang, lld
    raise AsmError("The ROCm assembler (llvm/bin/clang and ld.lld) is not in this image")


def available() -> tuple[bool, str]:
    if torch.version.hip is None:
        return False, "assembly kernels need a ROCm PyTorch"
    try:
        _toolchain()
    except AsmError as exc:
        return False, str(exc)
    return True, ""


def current_arch() -> str:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (getattr(properties, "gcnArchName", "") or "").split(":")[0]


def _run(argv: list[str]) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        raise AsmError(f"{Path(argv[0]).name} failed with exit code {done.returncode}: "
                       f"{(done.stderr or done.stdout)[-4000:]}")
    return done.stdout


def _cache_dir() -> Path:
    configured = os.environ.get("AKA_KERNEL_CACHE_DIR")
    root = Path(configured) if configured else Path(os.environ.get("TMPDIR", "/tmp")) / "aka-kernel-cache"
    path = root.resolve() / "asm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_source(path: str, text: str) -> None:
    """The static rules for one ``.s`` file."""
    if not path.endswith(SOURCE_SUFFIX):
        raise AsmError(f"{path}: assembly sources end in {SOURCE_SUFFIX}")
    found = FORBIDDEN_DIRECTIVE.search(text)
    if found:
        line = text.count("\n", 0, found.start()) + 1
        raise AsmError(f"{path}:{line}: .incbin and .include are not allowed; keep a kernel's whole "
                       "source in its .s file")


def assemble(source: Path, label: str, arch: str) -> Path:
    """Assemble and link ``source`` for ``arch``; return the cached code object."""
    text = source.read_text(encoding="utf-8")
    check_source(label, text)
    clang, lld = _toolchain()
    version = _run([str(clang), "--version"]).splitlines()[0]
    key = hashlib.sha256("\0".join([text, arch, version]).encode()).hexdigest()[:24]
    code_object = _cache_dir() / f"{key}.co"
    if not code_object.is_file():
        work = code_object.with_suffix(f".{os.getpid()}.tmp")
        obj = work.with_suffix(".o")
        try:
            _run([str(clang), "-x", "assembler", "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch}",
                  "-c", str(source), "-o", str(obj)])
            _run([str(lld), "-shared", str(obj), "-o", str(work)])
            os.replace(work, code_object)
        finally:
            for leftover in (work, obj):
                leftover.unlink(missing_ok=True)
    BUILD_EVIDENCE.update(assembler=str(clang), linker=str(lld), assembler_version=version, arch=arch)
    BUILD_EVIDENCE["files"].append({
        "source": label, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "code_object_sha256": hashlib.sha256(code_object.read_bytes()).hexdigest()})
    return code_object


_HIP = None


def _hip():
    """The HIP runtime PyTorch already loaded; a second copy would not know its streams."""
    global _HIP
    if _HIP is None:
        torch.cuda.init()
        maps = Path("/proc/self/maps").read_text().splitlines()
        loaded = sorted({line.split()[-1] for line in maps if "/libamdhip64.so" in line})
        if len(loaded) > 1:
            raise AsmError(f"Two HIP runtimes are loaded in this process: {loaded}")
        hip = ctypes.CDLL(loaded[0] if loaded else "libamdhip64.so")
        hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        hip.hipModuleLaunchKernel.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 7
                                              + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
        for name in ("hipModuleLoadData", "hipModuleGetFunction", "hipModuleLaunchKernel"):
            getattr(hip, name).restype = ctypes.c_int
        _HIP = hip
    return _HIP


def _check(status: int, what: str) -> None:
    if status != 0:
        raise AsmError(f"{what} failed with hipError {status}")


class Kernel:
    """One function of one loaded code object."""

    def __init__(self, code_object: Path, symbol: str, label: str):
        self.symbol, self.label = symbol, label
        image = code_object.read_bytes()
        # HIP keeps reading the image after hipModuleLoadData returns.
        self._image = ctypes.create_string_buffer(image, len(image))
        self._module, self._function = ctypes.c_void_p(), ctypes.c_void_p()
        hip = _hip()
        _check(hip.hipModuleLoadData(ctypes.byref(self._module), ctypes.cast(self._image, ctypes.c_void_p)),
               f"hipModuleLoadData({label})")
        _check(hip.hipModuleGetFunction(ctypes.byref(self._function), self._module, symbol.encode()),
               f"hipModuleGetFunction({label}: {symbol})")

    def launch(self, *, grid, block, args: list, shared_bytes: int = 0) -> None:
        grid = _triple(grid, "grid", MAX_GRID)
        block = _triple(block, "block", MAX_THREADS_PER_BLOCK)
        if block[0] * block[1] * block[2] > MAX_THREADS_PER_BLOCK:
            raise AsmError(f"block {block} has more than {MAX_THREADS_PER_BLOCK} threads")
        if type(shared_bytes) is not int or not 0 <= shared_bytes <= MAX_SHARED_BYTES:
            raise AsmError(f"shared_bytes must be an int in [0, {MAX_SHARED_BYTES}]")
        packed = pack_arguments(list(args))
        buffer = ctypes.create_string_buffer(packed, max(len(packed), 8))
        size = ctypes.c_size_t(len(packed))
        extra = (ctypes.c_void_p * 5)(HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.cast(buffer, ctypes.c_void_p),
                                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                      ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), HIP_LAUNCH_PARAM_END)
        # HIP copies the argument buffer when the launch is enqueued or captured
        # (verified on MI355X), so the buffer need not outlive this call.
        stream = torch.cuda.current_stream().cuda_stream
        _check(_hip().hipModuleLaunchKernel(self._function, *grid, *block, shared_bytes, ctypes.c_void_p(stream),
                                            None, ctypes.cast(extra, ctypes.c_void_p)),
               f"hipModuleLaunchKernel({self.label}: {self.symbol})")
        LAUNCHES.append(self.symbol)


def _package_root(depth: int) -> Path:
    """The directory of the implementation package that called ``load``."""
    frame = sys._getframe(depth)
    package = frame.f_globals.get("__package__") or ""
    module = sys.modules.get(package)
    path = getattr(module, "__path__", None)
    if not package or not path:
        raise AsmError("aka_asm.load is called from an implementation package's kernel.py or helpers")
    return Path(list(path)[0]).resolve()


def load(source: str, symbol: str) -> Kernel:
    """Return ``symbol`` from ``source``, a file in the implementation directory.

    A ``.s`` file is assembled for the current GPU. A ``.co`` file loads only if
    it is one of the code objects the task's asm baseline vendors, by SHA256.
    """
    if not isinstance(source, str) or not isinstance(symbol, str) or not SYMBOL.match(symbol):
        raise AsmError("aka_asm.load(source, symbol) takes a relative .s or .co path and a kernel symbol")
    root = _package_root(2)
    path = (root / source).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise AsmError(f"{source} is not a file in the implementation directory")
    label = path.relative_to(root).as_posix()
    if path.suffix == CODE_OBJECT_SUFFIX:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in ALLOWED_CODE_OBJECTS:
            raise AsmError(f"{label} is not one of the baseline's vendored code objects; write kernels as .s source")
        BUILD_EVIDENCE["files"].append({"source": label, "code_object_sha256": digest, "prebuilt": True})
        return Kernel(path, symbol, label)
    return Kernel(assemble(path, label, current_arch()), symbol, label)
