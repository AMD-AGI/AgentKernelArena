# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Candidate implementation policy: static source rules and runtime evidence.

Every candidate, whatever its language, must compute the operator in kernels
written in the target language. Host code may only allocate, take views and
launch. Static rules reject other backends, library operators, evaluator access
and side effects; the runtime audit rejects PyTorch computation and requires
actual target-language kernel launches. These checks are task-contract
enforcement, not a Python security sandbox.
"""
from __future__ import annotations

import ast
import re
import sys
from contextlib import contextmanager
from typing import Any, Iterator

STDLIB_ALLOWED = frozenset({
    "__future__", "math", "functools", "itertools", "operator", "typing", "dataclasses",
    "enum", "collections", "abc", "numbers", "contextlib", "copy", "warnings",
})
LANGUAGE_IMPORTS = {
    "triton": ("triton", "triton.language"),
    "gluon": ("triton", "triton.language", "triton.experimental", "triton.experimental.gluon"),
    "flydsl": ("flydsl",),
}
# Triton subpackages that are other kernel languages or library kernels.
TRITON_EXCLUDED = ("triton.experimental", "triton.tools", "triton.ops", "triton.testing",
                   "triton.profiler", "triton_kernels")
FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "setattr", "delattr",
    "breakpoint", "open", "input", "print", "exit", "quit", "help", "memoryview",
})
FORBIDDEN_ATTRIBUTES = frozenset({
    "__globals__", "__code__", "__closure__", "__builtins__", "__dict__", "__subclasses__",
    "__bases__", "__mro__", "__class__", "__getattribute__", "__setattr__", "__delattr__",
    "__import__", "__loader__", "__spec__", "__self__", "__func__", "__wrapped__",
    "f_globals", "f_locals", "f_back", "f_builtins", "f_code", "gi_frame", "cr_frame",
    "ag_frame", "tb_frame", "gi_code", "cr_code",
    # Synchronization, host round trips and timing inside launch code. Stream,
    # event and graph objects are reached only through torch.cuda, which the
    # allowlist below already restricts; kernel languages use these names.
    "synchronize", "item", "cpu", "tolist", "numpy", "elapsed_time", "record_stream",
    # Dispatch and evaluator state.
    "_python_dispatch", "_disable_current_modes", "_DisableTorchDispatch", "_pop_mode",
    "_pop_mode_temporarily", "_unset_dispatch_mode", "_DisablePythonDispatcher",
})
TORCH_FORBIDDEN_PREFIXES = (
    "torch.ops", "torch._C", "torch._ops", "torch.library", "torch.utils", "torch.nn",
    "torch.compile", "torch._dynamo", "torch._inductor", "torch.autograd", "torch.backends",
    "torch.fx", "torch.jit", "torch.distributed", "torch.profiler", "torch.hub", "torch.cuda.graphs",
    "torch.set_", "torch.cuda.set_", "torch.manual_seed", "torch.seed", "torch.cuda.manual_seed",
    "torch.cuda.seed", "torch.use_deterministic_algorithms", "torch.cuda.stream",
    "torch.save", "torch.load", "torch.rand", "torch.randn", "torch.randint", "torch.randperm",
    "torch.normal", "torch.bernoulli", "torch.multinomial", "torch.tensor", "torch.as_tensor",
    "torch.from_numpy", "torch.matmul", "torch.mm", "torch.bmm", "torch.einsum", "torch.linalg",
    "torch.fft", "torch.special", "torch.sparse", "torch._scaled_mm", "torch._int_mm",
)
TORCH_CUDA_ALLOWED = frozenset({
    "torch.cuda.current_stream", "torch.cuda.current_device", "torch.cuda.get_device_properties",
    "torch.cuda.device_count", "torch.cuda.is_available",
})


class PolicyError(ValueError):
    """The candidate source violates the implementation policy."""


def _bindings(tree: ast.Module) -> dict[str, str]:
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _qualified(node: ast.AST, names: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, names)
        return f"{base}.{node.attr}" if base else None
    return None


def _allowed_import(module: str, language: str) -> bool:
    top = module.split(".")[0]
    if top in STDLIB_ALLOWED and module == top:
        return True
    if module in ("typing_extensions",):
        return True
    if top == "torch":
        return module == "torch"
    if language in ("triton", "gluon") and top == "triton":
        if language == "triton" and module.startswith(TRITON_EXCLUDED):
            return False
        if language == "gluon":
            excluded = tuple(p for p in TRITON_EXCLUDED if p != "triton.experimental")
            if module.startswith(excluded):
                return False
            if module.startswith("triton.experimental") and not (
                    module in ("triton.experimental", "triton.experimental.gluon")
                    or module.startswith("triton.experimental.gluon.")):
                return False
        return True
    if language == "flydsl" and top == "flydsl":
        return True
    return False


def audit_python_sources(sources: dict[str, str], language: str) -> dict:
    """Reject disallowed imports, reflection, side effects and library operators."""
    if language not in LANGUAGE_IMPORTS:
        raise PolicyError(f"{language} is not a Python target language")
    python = {path: text for path, text in sources.items() if path.endswith(".py")}
    if set(sources) - set(python):
        others = sorted(set(sources) - set(python))
        raise PolicyError(f"A {language} candidate contains only .py sources; found {others[:4]}")
    local = {path.removesuffix(".py") for path in python}
    backend_seen = False
    for path, text in python.items():
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError as exc:
            raise PolicyError(f"{path}: {exc}") from exc
        names = _bindings(tree)
        for node in ast.walk(tree):
            where = f"{path}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise PolicyError(f"{where}: global/nonlocal state is not allowed")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _allowed_import(alias.name, language):
                        raise PolicyError(f"{where}: import {alias.name} is not allowed for {language}")
                    backend_seen |= alias.name.split(".")[0] in ("triton", "flydsl")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level != 1:
                        raise PolicyError(f"{where}: relative imports name a sibling candidate file")
                    siblings = ([node.module.split(".")[0]] if node.module
                                else [alias.name for alias in node.names])
                    if any(name not in local for name in siblings):
                        raise PolicyError(f"{where}: relative imports name a sibling candidate file")
                    continue
                module = node.module or ""
                if not _allowed_import(module, language):
                    raise PolicyError(f"{where}: import from {module} is not allowed for {language}")
                for alias in node.names:
                    if alias.name == "*":
                        raise PolicyError(f"{where}: star imports are not allowed")
                    full = f"{module}.{alias.name}"
                    if module == "torch" and any(full.startswith(p) for p in TORCH_FORBIDDEN_PREFIXES):
                        raise PolicyError(f"{where}: {full} is not a candidate implementation dependency")
                backend_seen |= module.split(".")[0] in ("triton", "flydsl")
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRIBUTES:
                    raise PolicyError(f"{where}: attribute {node.attr} is not allowed in candidate code")
                qualified = _qualified(node, names)
                if qualified and qualified.startswith("torch."):
                    if qualified.startswith("torch.cuda.") and qualified not in TORCH_CUDA_ALLOWED and \
                            not any(qualified.startswith(a + ".") for a in TORCH_CUDA_ALLOWED):
                        raise PolicyError(f"{where}: {qualified} is not allowed in candidate code")
                    if any(qualified == p or qualified.startswith(p + ".") or
                           (p.endswith("_") and qualified.startswith(p)) for p in TORCH_FORBIDDEN_PREFIXES):
                        raise PolicyError(f"{where}: {qualified} is not a candidate implementation dependency")
            elif isinstance(node, ast.Name):
                if node.id in FORBIDDEN_CALLS and isinstance(node.ctx, ast.Load):
                    raise PolicyError(f"{where}: {node.id} is not allowed in candidate code")
                if node.id.startswith("_aka"):
                    raise PolicyError(f"{where}: evaluator names are not available to candidates")
            elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
                raise PolicyError(f"{where}: the @ operator delegates to a library GEMM")
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                for target in targets:
                    base = target
                    while isinstance(base, (ast.Attribute, ast.Subscript)):
                        base = base.value
                    if base is not target and isinstance(base, ast.Name) and base.id in names:
                        raise PolicyError(f"{where}: candidates cannot mutate an imported module")
    if not backend_seen:
        raise PolicyError(f"The candidate imports no {language} kernel language")
    return {"python_sources": sorted(python), "language": language}


_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(?:<([^>\n]+)>|"([^"\n]+)")', re.MULTILINE)
_HIP_ALLOWED_HEADERS = ("hip/", "ATen/", "c10/", "torch/library.h", "torch/extension.h")
# ATen headers that declare operator or backend kernels callable without the
# dispatcher (for example ATen/ops/mm_cuda_dispatch.h -> at::cuda::mm_out), so
# the run-time PyTorch-operation check never sees them. Only the device and
# stream contexts are allowed from the backend directories.
_HIP_FORBIDDEN_HEADERS = re.compile(
    r"^ATen/(?:ops|native|functorch)/|^ATen/\w*Functions(?:_inl)?\.h$"
    r"|^ATen/(?:cuda|hip)/(?!(?:CUDAContext|HIPContext)\.h$)")
_STD_HEADERS = frozenset({
    "algorithm", "array", "bit", "cfloat", "climits", "cmath", "cstddef", "cstdint", "cstdio",
    "cstdlib", "cstring", "functional", "initializer_list", "limits", "memory", "numeric",
    "optional", "string", "tuple", "type_traits", "utility", "vector", "stdint.h", "math.h",
    "float.h", "limits.h", "stddef.h", "assert.h", "cassert",
})
_HIP_FORBIDDEN = re.compile(
    r"\b(hip(?:DeviceSynchronize|StreamSynchronize|EventSynchronize|SetDevice|Malloc|Free|HostMalloc|"
    r"HostFree|ExtMalloc|Ipc|StreamCreate|Graph|Module|MemcpyPeer|DeviceReset)\w*"
    r"|hip(?:Memcpy|Memset)(?!\w*Async)\w*|cuda[A-Z]\w*|rocblas_\w+|hipblas\w*|hipblasLt\w*|"
    r"miopen\w+|rccl\w+|getenv|fopen|system|popen|dlopen|std::thread|std::async|"
    r"(?:at|torch)::(?:matmul|mm|bmm|addmm|baddbmm|einsum|linear|conv\w*|softmax|log_softmax|layer_norm|"
    r"rms_norm|group_norm|batch_norm|sort|topk|cumsum|sum|mean|max|min|argmax|argmin|exp|log|"
    r"tanh|sigmoid|relu|gelu|silu|mul|add|sub|div|pow|sqrt|rsqrt|where|gather|scatter\w*|"
    r"index_select|_scaled_mm|_int_mm)\b)")
# Backend-specific operator namespaces call a library kernel directly, bypassing
# the dispatcher and so the run-time check; dispatcher redispatch does the same.
# The device, stream and guard helpers kernels need stay allowed.
_HIP_BACKEND_CALL = re.compile(
    r"\b(?:at|torch)::(?:cuda|hip|native|cpu|meta|mps|xpu|_ops|detail|"
    r"compositeexplicitautograd\w*|compositeimplicitautograd\w*)::(?!(?:getCurrentDeviceProperties|"
    r"getDeviceProperties|getCurrentCUDAStream|getCurrentHIPStream|getDefaultCUDAStream|"
    r"getDefaultHIPStream|CUDAGuard|HIPGuard|OptionalCUDAGuard|OptionalHIPGuard|CUDAStream|HIPStream)\b)\w+"
    r"|\b(?:Dispatcher|DispatchKeySet|redispatch\w*|ExcludeDispatchKeyGuard|IncludeDispatchKeyGuard|"
    r"_AutoDispatchBelow\w*|AutoDispatchBelow\w*)\b")
_HIP_STATIC_MUTABLE = re.compile(
    r"^[ \t]*(?:static[ \t]+|__device__[ \t]+)+(?![^;]*\b(?:const|constexpr)\b)[^;(){}\n]*;", re.MULTILINE)


def _blank_cpp(source: str) -> str:
    """Comments and string/char literals replaced by spaces, newlines kept."""
    out, i, n = [], 0, len(source)
    while i < n:
        if source.startswith("//", i):
            j = source.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif source.startswith("/*", i):
            j = source.find("*/", i + 2)
            if j < 0:
                raise PolicyError("Unterminated C++ comment")
            out.append("".join(c if c == "\n" else " " for c in source[i:j + 2]))
            i = j + 2
        elif source[i] in "\"'":
            quote, j = source[i], i + 1
            while j < n and source[j] != quote:
                j += 2 if source[j] == "\\" else 1
            if j >= n:
                raise PolicyError("Unterminated C++ literal")
            out.append(quote + " " * (j - i - 1) + quote)
            i = j + 1
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


def audit_hip_sources(sources: dict[str, str], entry: str) -> dict:
    """Allowed headers, the single op registration and no host-side side effects."""
    registrations = 0
    for path, text in sources.items():
        for match in _INCLUDE.finditer(text):
            header = match.group(1) or match.group(2)
            local = match.group(2) is not None and header in sources
            if not (local or header.startswith(_HIP_ALLOWED_HEADERS) or header in _STD_HEADERS):
                raise PolicyError(f"{path}: #include {header} is neither a candidate file nor an allowed header")
            if not local and _HIP_FORBIDDEN_HEADERS.search(header):
                raise PolicyError(f"{path}: #include {header} declares library operator kernels; "
                                  "it is not an allowed header")
        code = _blank_cpp(text)
        found = _HIP_FORBIDDEN.search(code)
        if found:
            line = code.count("\n", 0, found.start()) + 1
            raise PolicyError(f"{path}:{line}: {found.group(0)} is not allowed in a HIP candidate")
        backend = _HIP_BACKEND_CALL.search(code)
        if backend:
            line = code.count("\n", 0, backend.start()) + 1
            raise PolicyError(f"{path}:{line}: {backend.group(0)} calls a library kernel outside the "
                              "dispatcher; it is not allowed in a HIP candidate")
        mutable = _HIP_STATIC_MUTABLE.search(code)
        if mutable:
            line = code.count("\n", 0, mutable.start()) + 1
            raise PolicyError(f"{path}:{line}: non-const static/__device__ state persists between calls")
        if "TORCH_LIBRARY_IMPL" in code or "PYBIND11_MODULE" in code:
            raise PolicyError(f"{path}: register exactly one op with TORCH_LIBRARY_EXPAND")
        registrations += len(re.findall(r"\bTORCH_LIBRARY_EXPAND\s*\(\s*TORCH_EXTENSION_NAME\s*,", code))
    if registrations != 1 or "TORCH_LIBRARY_EXPAND" not in _blank_cpp(sources[entry]):
        raise PolicyError("The entry .hip file must contain the single "
                          "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) registration")
    if "hipLaunchKernelGGL" not in "".join(sources.values()) and "<<<" not in "".join(sources.values()):
        raise PolicyError("The HIP candidate launches no __global__ kernel")
    return {"hip_sources": sorted(sources)}


# Allocation, metadata/view changes and same-dtype copies are launch preparation.
PREPARATION_OPS = frozenset({
    "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::new_empty",
    "aten::new_empty_strided", "aten::zeros", "aten::zeros_like", "aten::new_zeros",
    "aten::ones", "aten::ones_like", "aten::new_ones", "aten::full", "aten::full_like",
    "aten::new_full", "aten::zero_", "aten::fill_", "aten::view", "aten::_unsafe_view",
    "aten::reshape", "aten::alias", "aten::detach", "aten::as_strided", "aten::transpose",
    "aten::t", "aten::permute", "aten::unsqueeze", "aten::squeeze", "aten::slice",
    "aten::select", "aten::expand", "aten::narrow", "aten::view_as_real",
    "aten::view_as_complex", "aten::split", "aten::split_with_sizes", "aten::chunk",
    "aten::unbind", "aten::unflatten", "aten::flatten", "aten::lift_fresh", "aten::resize_",
    "aten::set_",
})
DTYPE_PRESERVING_COPIES = frozenset({"aten::copy_", "aten::clone", "aten::_to_copy", "aten::contiguous"})


def _dtype_preserved(name: str, args: tuple, kwargs: dict) -> bool:
    import torch

    if name == "aten::copy_":
        return (len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor)
                and args[0].dtype == args[1].dtype)
    if name == "aten::_to_copy":
        dtype = kwargs.get("dtype")
        return isinstance(args[0], torch.Tensor) and (dtype is None or dtype == args[0].dtype)
    return True


@contextmanager
def preparation_only(label: str) -> Iterator[None]:
    """Reject PyTorch arithmetic, reductions and library operators."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class PreparationOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func._schema.name
            allowed = name in PREPARATION_OPS or (
                name in DTYPE_PRESERVING_COPIES and _dtype_preserved(name, args, kwargs))
            if not allowed:
                raise RuntimeError(f"{label} issued PyTorch operation {name}; operator computation "
                                   "must run in target-language kernels")
            return func(*args, **kwargs)

    with PreparationOnly():
        yield


class LaunchObserver:
    """Kernel launches made through Python-family runtimes during one call."""

    def __init__(self, language: str, package: str):
        self.language = language
        self.package = package
        self.launches: list[str] = []
        self.foreign: list[str] = []

    @contextmanager
    def observe(self) -> Iterator["LaunchObserver"]:
        if self.language in ("triton", "gluon"):
            from triton.runtime.jit import JITFunction

            original = JITFunction.run
            observer = self

            def run(self, *args, **kwargs):
                fn = getattr(self, "fn", None)
                module = getattr(fn, "__module__", "") or ""
                gluon = "gluon" in type(self).__name__.lower() or "gluon" in type(self).__module__
                label = f"{module}.{getattr(fn, '__name__', '?')}"
                if not module.startswith(observer.package) or gluon != (observer.language == "gluon"):
                    observer.foreign.append(f"{type(self).__name__} {label}")
                else:
                    observer.launches.append(label)
                return original(self, *args, **kwargs)

            JITFunction.run = run
            try:
                yield self
            finally:
                JITFunction.run = original
        elif self.language == "flydsl":
            previous = sys.getprofile()
            observer = self

            def profile(frame, event, arg):
                if event != "call":
                    return
                module = frame.f_globals.get("__name__", "")
                if module.startswith("flydsl.") and frame.f_code.co_name == "__call__":
                    owner = type(frame.f_locals.get("self")).__name__
                    if any(part in owner for part in ("Jit", "Compiled", "Kernel")):
                        observer.launches.append(f"{module}.{owner}")

            sys.setprofile(profile)
            try:
                yield self
            finally:
                sys.setprofile(previous)
        else:
            yield self

    def require(self, role: str) -> None:
        if self.foreign:
            raise RuntimeError(f"{role} launched kernels that are not its own {self.language} kernels: "
                               f"{sorted(set(self.foreign))[:4]}")
        if self.language in ("triton", "gluon", "flydsl") and not self.launches:
            raise RuntimeError(f"No {self.language} kernel launch was observed in the {role} call")


def audited_call(call, *, language: str, package: str, role: str, enforce: bool, integrity=None):
    """Wrap ``call`` so each eager invocation is policy-checked.

    ``integrity``, when given, runs after each call, once the launch observer
    has restored what it temporarily replaced.
    """
    def run(**arguments: Any) -> None:
        if not enforce:
            result = call(**arguments)
        else:
            observer = LaunchObserver(language, package)
            with observer.observe(), preparation_only(role):
                result = call(**arguments)
            observer.require(role)
        if integrity is not None:
            integrity(f"a {role} call")
        if result is not None:
            raise RuntimeError(f"{role} run must write its outputs and return None")
    return run
