# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Assemble, load and launch AMDGPU assembly kernels for ``target_language: asm``.

Adapted from the assembly candidate runner of the AgentKernelArena assembly
task copies (``asm_candidate.py``). An assembly implementation is a flat Python
package like the other Python-family languages: ``kernel.py`` defines
``run(*, inputs, outputs)`` and launches kernels written in ``.s`` files beside
it, through the ``aka_asm`` facade:

    kernel = aka_asm.load("gemm.s", "gemm_kernel")      # at import time
    kernel.launch(grid=(gx, gy, gz), block=(256, 1, 1),
                  args=[a, b, out, aka_asm.u32(m)], shared_bytes=0)

Candidates reach only the facade (``FACADE``, registered as ``aka_asm``): ``load``,
the typed scalars ``u32``/``i32``/``u64``/``i64``/``f32`` and ``AsmError``. The
evaluator keeps everything else in this module, and the static audit allows no
other ``aka_asm`` attribute.

- ``load`` assembles an audited ``.s`` source (the exact text the loader
  registered, not whatever is on disk later) with the image's ROCm ``clang`` and
  ``ld.lld`` for the current GPU, or reads a vendored ``.co`` file.
- Every code object goes through ``_open``, the only caller of
  ``hipModuleLoadData``. It hashes the exact bytes it hands to HIP and accepts
  them only if the digest is one of the baseline's vendored code objects or one
  that ``_assemble`` produced in this process from an audited source.
- ``load`` returns a handle whose only capability is a ``launch`` closure; its
  type is a plain namespace, so constructing another one loads nothing.
- Launch counts and build evidence live in ``_STATE``, which no object handed to
  a candidate refers to.

``launch`` packs the arguments in order with natural alignment (tensors as 64-bit
device pointers, scalars only through the typed wrappers) and calls
``hipModuleLaunchKernel`` on the current PyTorch stream, which is what CUDA-graph
capture records. The ``.s`` file's ``.amdgpu_metadata`` must declare the same
argument layout.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import struct
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch

ALIAS = "aka_asm"
#: The names a candidate may use on ``aka_asm``; the audit rejects any other attribute.
API = frozenset({"load", "u32", "i32", "u64", "i64", "f32", "AsmError"})
SOURCE_SUFFIX = ".s"
CODE_OBJECT_SUFFIX = ".co"
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


class AsmError(RuntimeError):
    """An assembly source, its build or a launch is invalid."""


class _State:
    """Evaluator-private state. Its attributes change in place; the object is never rebound."""

    def __init__(self) -> None:
        self.allowed_code_objects: set[str] = set()      # vendored code objects (SHA256)
        self.assembled_code_objects: set[str] = set()    # produced by _assemble in this process
        self.audited_sources: dict[tuple[str, str], str] = {}   # (package root, relative .s) -> text
        self.launches: list[str] = []                    # one symbol per successful launch
        self.evidence: dict[str, Any] = {"files": []}
        self.hip = None


_STATE = _State()


# --- evaluator-side controls (never exposed to candidates) ---------------------------------------

def allow_code_objects(digests) -> None:
    """The baseline's vendored code objects, by SHA256; replaces the previous set."""
    _STATE.allowed_code_objects.clear()
    _STATE.allowed_code_objects.update(digests)


def register_sources(root: Path, sources: dict[str, str]) -> None:
    """Record the audited ``.s`` texts of an implementation package rooted at ``root``."""
    root = str(Path(root).resolve())
    for relative, text in sources.items():
        if relative.endswith(SOURCE_SUFFIX):
            _STATE.audited_sources[(root, relative)] = text


def launch_count() -> int:
    return len(_STATE.launches)


def launches_since(start: int) -> list[str]:
    return list(_STATE.launches[start:])


def build_evidence() -> dict[str, Any]:
    """A copy of the build evidence: tool versions and every source and code object SHA256."""
    return {**{k: v for k, v in _STATE.evidence.items() if k != "files"},
            "files": [dict(item) for item in _STATE.evidence["files"]]}


# --- typed scalars ---------------------------------------------------------------------------------

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


# --- toolchain and build --------------------------------------------------------------------------

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


def _assemble(text: str, label: str, arch: str) -> bytes:
    """Assemble and link ``text`` (an audited source) for ``arch``; return the code object bytes.

    The assembler reads a private copy of exactly this text, never the candidate's
    file, so what runs is what the audit read.
    """
    check_source(label, text)
    clang, lld = _toolchain()
    version = _run([str(clang), "--version"]).splitlines()[0]
    key = hashlib.sha256("\0".join([text, arch, version]).encode()).hexdigest()[:24]
    cache = _cache_dir()
    code_object = cache / f"{key}.co"
    if not code_object.is_file():
        stem = cache / f"{key}.{os.getpid()}"
        source, obj, work = stem.with_suffix(".s"), stem.with_suffix(".o"), stem.with_suffix(".tmp")
        try:
            source.write_text(text, encoding="utf-8")
            _run([str(clang), "-x", "assembler", "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch}",
                  "-c", str(source), "-o", str(obj)])
            _run([str(lld), "-shared", str(obj), "-o", str(work)])
            os.replace(work, code_object)
        finally:
            for leftover in (source, obj, work):
                leftover.unlink(missing_ok=True)
    image = code_object.read_bytes()
    digest = hashlib.sha256(image).hexdigest()
    _STATE.assembled_code_objects.add(digest)
    _STATE.evidence.update(assembler=str(clang), linker=str(lld), assembler_version=version, arch=arch)
    _STATE.evidence["files"].append({"source": label, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                     "code_object_sha256": digest})
    return image


# --- HIP --------------------------------------------------------------------------------------------

def _hip():
    """The HIP runtime PyTorch already loaded; a second copy would not know its streams."""
    if _STATE.hip is None:
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
        _STATE.hip = hip
    return _STATE.hip


def _check(status: int, what: str) -> None:
    if status != 0:
        raise AsmError(f"{what} failed with hipError {status}")


def _open(image: bytes, symbol: str, label: str) -> types.SimpleNamespace:
    """Load ``image`` and return a launch handle for ``symbol``.

    The only caller of ``hipModuleLoadData``: it hashes the exact bytes HIP
    receives and accepts only vendored or freshly assembled code objects.
    """
    if not isinstance(image, (bytes, bytearray)):
        raise AsmError("a code object is bytes")
    image = bytes(image)
    digest = hashlib.sha256(image).hexdigest()
    if digest not in _STATE.allowed_code_objects and digest not in _STATE.assembled_code_objects:
        raise AsmError(f"{label} is neither a vendored code object of the baseline nor assembled from an "
                       "audited .s source in this process")
    # HIP keeps reading the image after hipModuleLoadData returns, so the closure owns it.
    buffer = ctypes.create_string_buffer(image, len(image))
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    hip = _hip()
    _check(hip.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buffer, ctypes.c_void_p)),
           f"hipModuleLoadData({label})")
    _check(hip.hipModuleGetFunction(ctypes.byref(function), module, symbol.encode()),
           f"hipModuleGetFunction({label}: {symbol})")

    def launch(*, grid, block, args: list, shared_bytes: int = 0) -> None:
        grid_ = _triple(grid, "grid", MAX_GRID)
        block_ = _triple(block, "block", MAX_THREADS_PER_BLOCK)
        if block_[0] * block_[1] * block_[2] > MAX_THREADS_PER_BLOCK:
            raise AsmError(f"block {block_} has more than {MAX_THREADS_PER_BLOCK} threads")
        if type(shared_bytes) is not int or not 0 <= shared_bytes <= MAX_SHARED_BYTES:
            raise AsmError(f"shared_bytes must be an int in [0, {MAX_SHARED_BYTES}]")
        packed = pack_arguments(list(args))
        arguments = ctypes.create_string_buffer(packed, max(len(packed), 8))
        size = ctypes.c_size_t(len(packed))
        extra = (ctypes.c_void_p * 5)(HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.cast(arguments, ctypes.c_void_p),
                                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                      ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), HIP_LAUNCH_PARAM_END)
        # HIP copies the argument buffer when the launch is enqueued or captured
        # (verified on MI355X), so the buffer need not outlive this call.
        stream = torch.cuda.current_stream().cuda_stream
        _check(_hip().hipModuleLaunchKernel(function, *grid_, *block_, shared_bytes, ctypes.c_void_p(stream),
                                            None, ctypes.cast(extra, ctypes.c_void_p)),
               f"hipModuleLaunchKernel({label}: {symbol})")
        _STATE.launches.append(symbol)
        _ = buffer      # keep the image alive with the handle

    return types.SimpleNamespace(symbol=symbol, launch=launch)


# --- the candidate-facing entry point ---------------------------------------------------------------

def _package_root(depth: int) -> Path:
    """The directory of the implementation package that called ``load``."""
    frame = sys._getframe(depth)
    package = frame.f_globals.get("__package__") or ""
    module = sys.modules.get(package)
    path = getattr(module, "__path__", None)
    if not package or not path:
        raise AsmError("aka_asm.load is called from an implementation package's kernel.py or helpers")
    return Path(list(path)[0]).resolve()


def load(source: str, symbol: str) -> types.SimpleNamespace:
    """Return a launch handle for ``symbol`` in ``source``, a file of the implementation package.

    A ``.s`` file must be one the loader audited and is assembled from that
    exact text. A ``.co`` file loads only if it is one of the code objects the
    task's asm baseline vendors, by SHA256 (checked where it is loaded).
    """
    if not isinstance(source, str) or not isinstance(symbol, str) or not SYMBOL.match(symbol):
        raise AsmError("aka_asm.load(source, symbol) takes a relative .s or .co path and a kernel symbol")
    root = _package_root(2)
    path = (root / source).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise AsmError(f"{source} is not a file in the implementation directory")
    label = path.relative_to(root).as_posix()
    if path.suffix == CODE_OBJECT_SUFFIX:
        image = path.read_bytes()
        handle = _open(image, symbol, label)
        _STATE.evidence["files"].append({"source": label, "code_object_sha256": hashlib.sha256(image).hexdigest(),
                                         "prebuilt": True})
        return handle
    text = _STATE.audited_sources.get((str(root), label))
    if text is None:
        raise AsmError(f"{label} is not an audited .s source of this implementation")
    return _open(_assemble(text, label, current_arch()), symbol, label)


def _facade() -> types.ModuleType:
    facade = types.ModuleType(ALIAS, "AMDGPU assembly kernels for target_language asm: load and typed scalars.")
    for name in sorted(API):
        setattr(facade, name, globals()[name])
    facade.__all__ = sorted(API)
    return facade


#: The module candidates import as ``aka_asm``; registered at import so the
#: integrity snapshot, taken before candidate code loads, covers it.
FACADE = _facade()
sys.modules[ALIAS] = FACADE
