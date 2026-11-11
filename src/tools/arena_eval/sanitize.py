# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Framework-owned sanitizer lanes for schema-v3 tasks.

    python3 -m _aka_eval.evaluate {baseline|candidate} sanitize LANE

An evaluation-tool sidecar (src/eval_tools) runs each lane in a fresh process
and sets the lane's environment first: allocator switches, compiler
instrumentation and runtime preloads only take effect before PyTorch, HIP and
Triton initialize. A lane checks that environment instead of trusting it, runs
the role's implementation on the task's own correctness cases and prints one
``ARENA_EVAL_RESULT=`` report whose action is ``sanitize``.

The report is evidence, not the verdict. A hardware memory fault or a GPU
AddressSanitizer report terminates the process before any report is printed,
so the sidecar's plugin judges each lane from the exit status, stderr and the
report together.

Lanes:

``page_fault_guard``
    Every correctness case with the caching allocators off, so an access past
    a tensor's allocation reaches an unmapped page and faults instead of
    landing silently in a neighbouring tensor.
``csan``
    Every correctness case under PyTorch's CUDA Sanitizer (stream races between
    PyTorch-visible operations, including a HIP candidate's custom op).
``tilelens``
    The smallest case with each Triton or Gluon kernel run by TileLens's
    symbolic out-of-bounds checker in Triton's CPU interpreter.
``fpsan``
    The smallest case through Triton FpSan for the candidate and the baseline
    (both Triton or Gluon); a differing digest means a different floating-point
    expression. A signal only: correct rewrites also change the digest.
``gpu_asan``
    Every correctness case with GPU AddressSanitizer instrumentation
    (bare-metal hosts only: the ASan runtime needs XNACK host-memory access).
``capture``
    One call on the smallest case that records each kernel the role dispatched:
    its final code object and, for Triton and Gluon, a replay record (launch
    geometry, arguments and memory before and after) from which the framework
    builds rocJITsu replay capsules.
"""
from __future__ import annotations

import hashlib
import json
import re
import math
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

LANES = ("page_fault_guard", "csan", "tilelens", "fpsan", "gpu_asan", "capture")

# The allocator switches a lane requires; the sidecar sets them before start-up.
NO_CACHING = {"PYTORCH_NO_HIP_MEMORY_CACHING": "1", "AMD_PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
              "HSA_DISABLE_FRAGMENT_ALLOCATOR": "1"}
ATTESTATION_ENV = "AKA_BUILD_ATTESTATION_PATH"
OUTPUT_ENV = "AKA_SANITIZE_OUTPUT_DIR"
FPSAN_PREFIX = "AKA_FPSAN_RESULT "
CAPTURE_SCHEMA = 1
# Capture is bounded: a replay record holds every allocation a launch touches.
CAPTURE_MAX_LAUNCHES = 16
CAPTURE_MAX_BYTES = 256 << 20


def _evaluate():
    from . import evaluate

    return evaluate


def _require_env(required: dict) -> dict:
    evaluate = _evaluate()
    missing = {key: value for key, value in required.items() if os.environ.get(key) != value}
    if missing:
        raise evaluate.ActionFailure(f"The sanitizer environment is not set: {sorted(missing)}", "environment")
    return dict(required)


def _case_size(contract, case) -> int:
    symbols = contract.symbols(case)
    total = 0
    for declaration in list(contract.inputs.values()) + list(contract.outputs.values()):
        if declaration.get("kind", "tensor") == "tensor" or "shape" in declaration:
            total += math.prod(contract.shape(declaration, symbols))
    return total


def smallest_case(contract):
    return min(contract.cases, key=lambda case: (_case_size(contract, case), case.case_id))


def _load_role(contract, role: str, *, enforce: bool):
    """The role's audited call; ``enforce=False`` keeps the integrity check but skips launch auditing."""
    evaluate = _evaluate()
    from .audit import PolicyError
    from .loader import BuildError

    if role == "candidate" and evaluate.candidate_state(contract) == "unimplemented":
        raise evaluate.ActionFailure("The candidate is still the unimplemented template", "unimplemented")
    check = evaluate._integrity(role)
    try:
        call, evidence = evaluate._load(contract, role, audit=enforce and evaluate._policy_enforced(role),
                                        integrity=check)
    except PolicyError as exc:
        raise evaluate.ActionFailure(str(exc), "policy_violation")
    except BuildError as exc:
        raise evaluate.ActionFailure(str(exc), "compile_error")
    return call, evidence, check


def _run_once(contract, call, case, device, seed_index: int = 0) -> dict:
    """One call on fresh inputs; returns the written tensors."""
    from . import values

    inputs = values.make_inputs(contract, case, case.seeds[seed_index], device)
    kernel = values.clone_inputs(inputs.kernel)
    outputs = values.allocate_outputs(contract, case, device)
    values.poison(outputs, 31 + seed_index)
    _synchronize()
    call(**kernel, **outputs)
    _synchronize()
    return values.checked_values(contract, kernel, outputs)


def _synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _report(role: str, status: str, rows: list, lane: str, metadata: dict, reason=None, kind=None) -> dict:
    evaluate = _evaluate()
    return evaluate._report(role, "sanitize", status, rows, reason, kind, metadata={"lane": lane, **metadata})


def _crash_rows(contract, role, call, device, cases) -> tuple[list, list]:
    """Run each case once without checking values; return rows and errors."""
    rows, errors = [], []
    for case in cases:
        row = case.identity()
        try:
            _run_once(contract, call, case, device)
            row["status"] = "PASS"
        except Exception as exc:
            row.update(status="FAIL", failure_kind="execution_error",
                       metadata={"error": f"{type(exc).__name__}: {exc}"[:2000]})
            errors.append(row["metadata"]["error"])
        rows.append(row)
    return rows, errors


# ---------------------------------------------------------------------------
# page_fault_guard


def page_fault_guard(contract, role: str) -> dict:
    evaluate = _evaluate()
    environment = _require_env(NO_CACHING)
    device = evaluate._device()
    call, baseline_call, need_baseline, check = evaluate._role_calls(contract, role)
    rows, failures, kinds = evaluate.check_cases(contract, role, call, baseline_call, need_baseline, device)
    if check is not None:
        check("sanitize")
    metadata = {"environment": environment, "cases": len(rows)}
    if failures:
        return _report(role, "FAIL", rows, "page_fault_guard", metadata,
                       f"{len(failures)} case(s) failed with the caching allocators off: {failures[:8]}",
                       kinds.pop() if len(kinds) == 1 else "mixed")
    return _report(role, "PASS", rows, "page_fault_guard", metadata)


# ---------------------------------------------------------------------------
# csan


def csan(contract, role: str) -> dict:
    evaluate = _evaluate()
    import torch.cuda._sanitizer as cuda_sanitizer

    device = evaluate._device()
    call, _, check = _load_role(contract, role, enforce=True)
    cuda_sanitizer.enable_cuda_sanitizer()
    rows, errors = _crash_rows(contract, role, call, device, contract.cases)
    if check is not None:
        check("sanitize")
    races = [error for error in errors if error.startswith("CUDASanitizerErrors")]
    metadata = {"cases": len(rows), "races": races[:8],
                "enabled": bool(getattr(cuda_sanitizer.cuda_sanitizer, "enabled", False))}
    if races:
        return _report(role, "FAIL", rows, "csan", metadata, f"CSAN reported a stream race in {len(races)} case(s)",
                       "data_race")
    if errors:
        return _report(role, "FAIL", rows, "csan", metadata, f"{len(errors)} case(s) failed: {errors[:2]}",
                       "execution_error")
    return _report(role, "PASS", rows, "csan", metadata)


# ---------------------------------------------------------------------------
# tilelens


def _tilelens_patch(sanitizers: list):
    """Patch triton.jit/autotune and gluon.jit as TileLens's CLI does, with abort_on_error off."""
    import triton
    import triton.runtime.interpreter as interpreter
    import tilelens
    from tilelens.clients import Sanitizer
    from tilelens.core.config import config
    from triton.experimental import gluon

    original = {"jit": triton.jit, "autotune": triton.autotune, "gluon": gluon.jit}

    def wrap(kernel, *, frontend="triton"):
        client = Sanitizer(abort_on_error=False)
        sanitizers.append(client)
        return tilelens.trace(client=client, frontend=frontend)(kernel)

    def patched_jit(base, frontend):
        def jit(fn=None, **options):
            if fn is None:
                return lambda function: wrap(base(**options)(function), frontend=frontend)
            return wrap(base(fn), frontend=frontend)
        return jit

    def autotune(fn=None, **options):
        if fn is None:
            return lambda function: wrap(original["autotune"](**options)(function))
        return wrap(original["autotune"](fn))

    config.cli_active = True
    triton.jit = triton.language.jit = interpreter.jit = patched_jit(original["jit"], "triton")
    triton.autotune = autotune
    gluon.jit = patched_jit(original["gluon"], "gluon")


def _tilelens_record(record) -> dict:
    def text(value, limit=400):
        return str(value)[:limit] if value is not None else None

    tracebacks = getattr(record, "user_code_tracebacks", None) or []
    return {"op": getattr(getattr(record, "op_type", None), "__name__", text(getattr(record, "op_type", None))),
            "tensor": text(getattr(record, "tensor_name", None)),
            "address": text(getattr(record, "violation_address", None)),
            "location": [text(item, 300) for item in list(tracebacks)[:3]]}


def tilelens(contract, role: str) -> dict:
    evaluate = _evaluate()
    implementation = contract.implementation(role)
    if implementation.language not in ("triton", "gluon"):
        raise evaluate.ActionFailure(f"TileLens checks Triton and Gluon, not {implementation.language}",
                                     "unsupported")
    device = evaluate._device()
    sanitizers: list = []
    try:
        _tilelens_patch(sanitizers)
    except ImportError as exc:
        raise evaluate.ActionFailure(f"TileLens is not installed: {exc}", "environment")
    # Kernels execute in the CPU interpreter, which the GPU launch audit cannot
    # observe; the call keeps its integrity check and ran audited in correctness.
    call, _, check = _load_role(contract, role, enforce=False)
    case = smallest_case(contract)
    row = case.identity()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _run_once(contract, call, case, device)
            row["status"] = "PASS"
            error = None
        except SystemExit as exc:  # abort_on_error is off; any exit is unexpected
            error = f"SystemExit: {exc.code}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
    if check is not None:
        check("sanitize")
    violations = [_tilelens_record(record) for client in sanitizers for record in getattr(client, "records", [])]
    approximations = sorted({str(item.message)[:200] for item in caught
                             if "approximat" in str(item.message).lower()})
    metadata = {"case": case.case_id, "kernels": len(sanitizers), "violations": violations[:16],
                "violation_count": len(violations), "approximations": approximations}
    if violations:
        row.update(status="FAIL", failure_kind="out_of_bounds")
        return _report(role, "FAIL", [row], "tilelens", metadata,
                       f"TileLens found {len(violations)} out-of-bounds access(es)", "out_of_bounds")
    if error is not None:
        row.update(status="FAIL", failure_kind="unanalyzable", metadata={"error": error})
        return _report(role, "FAIL", [row], "tilelens", metadata, f"TileLens could not analyze the call: {error}",
                       "unanalyzable")
    if not sanitizers:
        return _report(role, "FAIL", [row], "tilelens", metadata, "No Triton or Gluon kernel was traced",
                       "unanalyzable")
    return _report(role, "PASS", [row], "tilelens", metadata)


# ---------------------------------------------------------------------------
# fpsan


def _launch_names():
    """Kernel names launched through Triton while the context is active."""
    from triton import knobs

    names: list[str] = []
    previous = knobs.runtime.launch_enter_hook

    def hook(metadata):
        data = getattr(metadata, "data", metadata)
        name = data.get("name") if isinstance(data, dict) else None
        if name:
            names.append(str(name))
        if previous is not None:
            previous(metadata)

    return names, previous, hook


def _cache_kernels(cache: Path) -> dict:
    """Compiled kernels in a Triton cache: name -> (metadata, hsaco path).

    Triton names a cache entry's files after the Python function, truncated to
    150 characters, while ``metadata["name"]`` is the kernel symbol, which a
    ``@triton.jit(repr=...)`` kernel extends with its constants. The code
    object is therefore the metadata file's sibling, not ``<name>.hsaco``.
    """
    kernels = {}
    for metadata_path in sorted(cache.rglob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = metadata.get("name") if isinstance(metadata, dict) else None
        hsaco = metadata_path.with_suffix(".hsaco")
        if name and hsaco.is_file():
            kernels.setdefault(name, []).append((metadata, hsaco))
    return kernels


def _digest(written: dict) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(written):
        tensor = written[name].detach().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes() if tensor.numel() else b"")
    return digest.hexdigest()


def fpsan(contract, role: str) -> dict:
    evaluate = _evaluate()
    from triton import knobs

    environment = _require_env({"TRITON_INSTRUMENTATION_MODE": "fpsan"})
    if role != "candidate":
        raise evaluate.ActionFailure("FpSan compares the candidate with the baseline", "usage")
    languages = {contract.baseline.language, contract.candidate.language}
    if not languages <= {"triton", "gluon"}:
        raise evaluate.ActionFailure(f"FpSan compares Triton/Gluon implementations, not {sorted(languages)}",
                                     "unsupported")
    cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
    if not cache.is_dir():
        raise evaluate.ActionFailure("FpSan needs a fresh TRITON_CACHE_DIR", "environment")
    device = evaluate._device()
    case = smallest_case(contract)
    baseline_call = evaluate._load(contract, "baseline", audit=False)[0]
    call, _, check = _load_role(contract, "candidate", enforce=True)
    launched = {}
    for label, function in (("baseline", baseline_call), ("candidate", call)):
        names, previous, hook = _launch_names()
        knobs.runtime.launch_enter_hook = hook
        try:
            written = _run_once(contract, function, case, device)
        finally:
            knobs.runtime.launch_enter_hook = previous
        launched[label] = {"names": sorted(set(names)), "digest": _digest(written)}
    if check is not None:
        check("sanitize")
    compiled = _cache_kernels(cache)
    instrumented = {}
    for label, record in launched.items():
        modes = [entry[0].get("instrumentation_mode") for name in record["names"] for entry in compiled.get(name, [])]
        instrumented[label] = bool(record["names"]) and bool(modes) and all(mode == "fpsan" for mode in modes) and all(
            name in compiled for name in record["names"])
    payload = {"instrumented": all(instrumented.values()), "reference_digest": launched["baseline"]["digest"],
               "candidate_digest": launched["candidate"]["digest"], "case": case.case_id,
               "reference_kernels": launched["baseline"]["names"], "candidate_kernels": launched["candidate"]["names"]}
    attestation = os.environ.get(ATTESTATION_ENV)
    if attestation and instrumented.get("candidate"):
        name = launched["candidate"]["names"][0]
        _write_attestation(Path(attestation), tool="triton_fpsan", artifact=compiled[name][0][1],
                           build_command=("triton", "TRITON_INSTRUMENTATION_MODE=fpsan"),
                           environment=environment,
                           evidence={"reference_instrumented": instrumented["baseline"],
                                     "candidate_instrumented": instrumented["candidate"],
                                     "kernels": payload["candidate_kernels"]})
    print(FPSAN_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return _report(role, "PASS", [case.identity() | {"status": "PASS"}], "fpsan",
                   {"environment": environment, **payload})


# ---------------------------------------------------------------------------
# gpu_asan


def _write_attestation(path: Path, *, tool: str, artifact: Path, build_command, environment: dict,
                       evidence: dict, compiler: str = "", compiler_version: str = "", target_arch: str = "") -> None:
    """A src/eval_tools BuildAttestation, written without importing the framework."""
    path.parent.mkdir(parents=True, exist_ok=True)
    copy = path.parent / f"attested-{artifact.name}"
    shutil.copyfile(artifact, copy)
    payload = {"tool": tool, "instrumented": True, "compiler": compiler, "compiler_version": compiler_version,
               "target_arch": target_arch, "build_command": [str(item) for item in build_command],
               "artifact_path": copy.name, "artifact_sha256": hashlib.sha256(copy.read_bytes()).hexdigest(),
               "environment": dict(environment), "evidence": dict(evidence)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _offload_bundler() -> Path | None:
    from torch.utils import cpp_extension

    roots = [Path(cpp_extension.ROCM_HOME or "/opt/rocm")]
    hipcc = shutil.which("hipcc")
    if hipcc:
        roots.append(Path(hipcc).resolve().parent.parent)
    for root in roots:
        for relative in ("llvm/bin/clang-offload-bundler", "lib/llvm/bin/clang-offload-bundler"):
            if (root / relative).is_file():
                return root / relative
    found = shutil.which("clang-offload-bundler")
    return Path(found) if found else None


def _elf_section(path: Path, wanted: str) -> bytes | None:
    """The bytes of one section of a 64-bit little-endian ELF file."""
    import struct

    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + index * shentsize) for index in range(shnum)]
    names = sections[shstrndx]
    for name_offset, _, _, _, offset, size, *_ in sections:
        start = names[4] + name_offset
        name = data[start:data.index(b"\0", start)].decode()
        if name == wanted:
            return data[offset:offset + size]
    return None


def hip_code_objects(library: Path, output: Path) -> list[Path]:
    """Unbundle the device code objects embedded in a HIP torch extension."""
    fatbin = _elf_section(library, ".hip_fatbin")
    bundler = _offload_bundler()
    if not fatbin or bundler is None:
        return []
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"{library.stem}.hip_fatbin"
    bundle.write_bytes(fatbin)
    listed = subprocess.run([str(bundler), "--list", "--type=o", f"--input={bundle}"], capture_output=True,
                            text=True, check=False, timeout=120)
    objects = []
    for index, target in enumerate(line.strip() for line in listed.stdout.splitlines()):
        if not target.startswith("hip") or "gfx" not in target:
            continue
        code_object = output / f"{library.stem}-{index}.co"
        unbundled = subprocess.run([str(bundler), "--unbundle", "--type=o", f"--input={bundle}",
                                    f"--targets={target}", f"--output={code_object}"],
                                   capture_output=True, text=True, check=False, timeout=120)
        if unbundled.returncode == 0 and code_object.is_file() and code_object.read_bytes()[:4] == b"\x7fELF":
            objects.append(code_object)
    return objects


def gpu_asan(contract, role: str) -> dict:
    evaluate = _evaluate()
    from .loader import SANITIZER_ARCH_ENV, SANITIZER_CFLAGS_ENV, hip_build_arch, hip_build_flags

    language = contract.implementation(role).language
    required = {"HSA_XNACK": "1", **NO_CACHING}
    if language == "hip":
        required.update({SANITIZER_CFLAGS_ENV: "-fsanitize=address -shared-libsan"})
    elif language in ("triton", "gluon"):
        required.update({"TRITON_ENABLE_ASAN": "1", "AMDGCN_USE_BUFFER_OPS": "0"})
    else:
        raise evaluate.ActionFailure(f"GPU ASan does not instrument {language}", "unsupported")
    environment = _require_env(required)
    device = evaluate._device()
    call, implementation, check = _load_role(contract, role, enforce=True)
    rows, errors = _crash_rows(contract, role, call, device, contract.cases)
    if check is not None:
        check("sanitize")
    evidence: dict = {"language": language}
    artifact = None
    command: tuple = ()
    arch = ""
    if language == "hip":
        arch = hip_build_arch()
        command = ("hipcc", *hip_build_flags(), f"--offload-arch={arch}")
        environment[SANITIZER_ARCH_ENV] = arch
        library = Path(implementation.get("library") or "")
        objects = hip_code_objects(library, Path(os.environ.get("TMPDIR", "/tmp")) / "aka-asan-co") \
            if library.is_file() else []
        evidence["device_code_objects"] = len(objects)
        evidence["device_instrumented"] = bool(objects) and all(b"__asan" in item.read_bytes() for item in objects)
        artifact = library if library.is_file() else None
    else:
        cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
        compiled = [entry for entries in _cache_kernels(cache).values() for entry in entries] if cache.is_dir() else []
        evidence["kernels"] = len(compiled)
        evidence["device_instrumented"] = bool(compiled) and all(b"__asan" in entry[1].read_bytes()
                                                                  for entry in compiled)
        artifact = compiled[0][1] if compiled else None
        command = ("triton", "TRITON_ENABLE_ASAN=1", "AMDGCN_USE_BUFFER_OPS=0")
    attestation = os.environ.get(ATTESTATION_ENV)
    if attestation and artifact is not None and evidence["device_instrumented"]:
        _write_attestation(Path(attestation), tool="gpu_asan", artifact=artifact, build_command=command,
                           environment=environment, evidence=evidence, target_arch=arch)
    metadata = {"environment": environment, "cases": len(rows), **evidence}
    if errors:
        return _report(role, "FAIL", rows, "gpu_asan", metadata, f"{len(errors)} case(s) failed: {errors[:2]}",
                       "execution_error")
    return _report(role, "PASS", rows, "gpu_asan", metadata)


# ---------------------------------------------------------------------------
# capture


class _Capture:
    """Records Triton/Gluon launches: compiled kernel, grid, arguments, memory before and after."""

    def __init__(self, output: Path):
        self.output = output
        self.launches: list[dict] = []
        self.skipped: list[str] = []
        self.bytes = 0
        self.active = False

    def _storage(self, tensor):
        storage = tensor.untyped_storage()
        return storage.data_ptr(), storage.nbytes(), storage

    def record(self, jit, args, kwargs, grid, run):
        import torch

        if not self.active:
            return run()
        if len(self.launches) >= CAPTURE_MAX_LAUNCHES:
            self.skipped.append(f"launch limit {CAPTURE_MAX_LAUNCHES}")
            return run()
        try:
            # Launch options (num_warps, num_stages, ...) are not kernel parameters.
            parameters = jit.signature.parameters
            bound = jit.signature.bind(*args, **{k: v for k, v in kwargs.items() if k in parameters})
            bound.apply_defaults()
        except TypeError as exc:
            self.skipped.append(f"cannot bind arguments: {exc}")
            return run()
        storages, arguments = {}, []
        for index, (name, value) in enumerate(bound.arguments.items()):
            if isinstance(value, torch.Tensor):
                base, nbytes, storage = self._storage(value)
                key = f"a{base:x}"
                storages.setdefault(key, (base, nbytes, storage))
                arguments.append({"index": index, "name": name, "kind": "tensor", "allocation": key,
                                  "byte_offset": value.data_ptr() - base})
            elif isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
                arguments.append({"index": index, "name": name, "kind": "value", "value": value})
            else:
                arguments.append({"index": index, "name": name, "kind": "opaque", "type": type(value).__name__})
        total = sum(entry[1] for entry in storages.values())
        if self.bytes + 2 * total > CAPTURE_MAX_BYTES:
            self.skipped.append(f"memory limit {CAPTURE_MAX_BYTES} bytes")
            return run()
        torch.cuda.synchronize()
        before = {key: self._bytes(storage, nbytes) for key, (_, nbytes, storage) in storages.items()}
        kernel = run()
        torch.cuda.synchronize()
        if kernel is None or not hasattr(kernel, "asm"):
            self.skipped.append(f"{getattr(jit, '__name__', '?')}: the launch returned no compiled kernel")
            return kernel
        after = {key: self._bytes(storage, nbytes) for key, (_, nbytes, storage) in storages.items()}
        self.bytes += 2 * total
        meta = dict(bound.arguments)
        launch_grid = grid(meta) if callable(grid) else grid
        launch_grid = tuple(int(item) for item in launch_grid) + (1,) * (3 - len(tuple(launch_grid)))
        self._save(kernel, launch_grid, arguments, before, after)
        return kernel

    @staticmethod
    def _bytes(storage, nbytes: int) -> bytes:
        import torch

        if nbytes == 0:
            return b""
        view = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage, 0, (nbytes,), (1,))
        return view.cpu().numpy().tobytes()

    def _save(self, kernel, grid, arguments, before, after) -> None:
        index = len(self.launches)
        directory = self.output / f"launch-{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        hsaco = kernel.asm.get("hsaco") if hasattr(kernel, "asm") else None
        if not isinstance(hsaco, (bytes, bytearray)):
            self.skipped.append(f"{getattr(kernel, 'name', '?')}: no hsaco")
            return
        name = str(kernel.name)
        (directory / f"{name}.hsaco").write_bytes(bytes(hsaco))
        allocations = []
        for key in before:
            (directory / f"{key}.before.bin").write_bytes(before[key])
            (directory / f"{key}.after.bin").write_bytes(after[key])
            allocations.append({"id": key, "byte_size": len(before[key]), "before_blob": f"{key}.before.bin",
                                "expected_blob": f"{key}.after.bin", "changed": before[key] != after[key]})
        metadata = getattr(kernel, "metadata", None)

        def field(name, default=0):
            value = metadata.get(name, default) if isinstance(metadata, dict) else getattr(metadata, name, default)
            return value if isinstance(value, (int, float, str, bool)) or value is None else str(value)

        source = getattr(kernel, "src", None)
        signature = {str(key): str(value) for key, value in dict(getattr(source, "signature", {}) or {}).items()}
        constants = {}
        for key, value in dict(getattr(source, "constants", {}) or {}).items():
            constants[json.dumps(key if not isinstance(key, tuple) else list(key))] = (
                value if isinstance(value, (int, float, str, bool)) or value is None else str(value))
        record = {"kernel_name": name, "hsaco": f"{name}.hsaco",
                  "hsaco_sha256": hashlib.sha256(bytes(hsaco)).hexdigest(), "grid": list(grid),
                  "metadata": {key: field(key, default) for key, default in (
                      ("num_warps", 4), ("warp_size", 64), ("shared", 0), ("profile_scratch_size", 0),
                      ("profile_scratch_align", 1), ("global_scratch_size", 0), ("global_scratch_align", 1),
                      ("target", ""))},
                  "signature": signature, "constants": constants,
                  "arg_names": list(getattr(getattr(source, "fn", None), "arg_names", []) or []),
                  "arguments": arguments, "allocations": allocations}
        (directory / "launch.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        self.launches.append({"directory": directory.name, "kernel_name": name,
                              "hsaco": f"{directory.name}/{name}.hsaco", "hsaco_sha256": record["hsaco_sha256"]})

    def install(self):
        from triton.runtime.jit import JITFunction

        original = JITFunction.run
        capture = self

        def run(self, *args, grid, warmup, **kwargs):
            if warmup:
                return original(self, *args, grid=grid, warmup=warmup, **kwargs)
            return capture.record(self, args, kwargs, grid,
                                  lambda: original(self, *args, grid=grid, warmup=warmup, **kwargs))

        JITFunction.run = run
        return original


_MLIR_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)


def _mlir_bytes(body: str) -> bytes:
    """Decode the body of an MLIR string literal: ``\\XX`` hex escapes, ``\\\\`` and ``\\"``."""
    out, position = bytearray(), 0
    while position < len(body):
        char = body[position]
        if char == "\\" and position + 1 < len(body):
            pair = body[position + 1:position + 3]
            if len(pair) == 2 and all(c in "0123456789abcdefABCDEF" for c in pair):
                out.append(int(pair, 16))
                position += 3
                continue
            out.extend(body[position + 1].encode("utf-8"))
            position += 2
            continue
        out.extend(char.encode("utf-8"))
        position += 1
    return bytes(out)


def _flydsl_code_objects(observed: list, output: Path) -> list[dict]:
    """The HSACO embedded in each FlyDSL compiled artifact's ``gpu.binary`` op."""
    objects, seen = [], set()
    for artifact in observed:
        text = vars(artifact).get("_ir_text")
        if not isinstance(text, str) or "gpu.binary" not in text:
            continue
        for match in _MLIR_STRING_RE.finditer(text):
            if match.group(1)[:6].upper() != "\\7FELF":
                continue
            data = _mlir_bytes(match.group(1))
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            path = output / f"flydsl-{len(objects):02d}.hsaco"
            path.write_bytes(data)
            objects.append({"code_object": path.name, "sha256": digest, "kernels": None})
    return objects


def _flydsl_observer(observed: list):
    """A profile hook that records each FlyDSL compiled artifact the call uses.

    FlyDSL's ``CompiledArtifact`` holds the lowered module (with its embedded
    code object) in ``_ir_text``; a launch reaches it through its executor
    methods (``_get_func_exe``, ``_ensure_engine``), not ``__call__``. Only
    FlyDSL compiler frames are inspected, through the instance dictionary, so
    no property or ``__getattr__`` of the role's objects runs.
    """

    def profile(frame, event, arg):
        if event != "call" or not (frame.f_globals.get("__name__") or "").startswith("flydsl.compiler"):
            return
        owner = frame.f_locals.get("self")
        state = getattr(owner, "__dict__", None) if owner is not None else None
        if isinstance(state, dict) and isinstance(state.get("_ir_text"), str):
            if all(owner is not item for item in observed):
                observed.append(owner)

    return profile


def capture(contract, role: str) -> dict:
    evaluate = _evaluate()
    output = Path(os.environ.get(OUTPUT_ENV, ""))
    if not output.is_dir():
        raise evaluate.ActionFailure(f"{OUTPUT_ENV} must name an existing directory", "environment")
    language = contract.implementation(role).language
    if language not in ("triton", "gluon", "hip", "flydsl"):
        raise evaluate.ActionFailure(f"Capture does not support {language}", "unsupported")
    device = evaluate._device()
    case = smallest_case(contract)
    manifest = {"schema_version": CAPTURE_SCHEMA, "role": role, "language": language, "case": case.case_id,
                "seed": case.seeds[0], "code_objects": [], "launches": [], "skipped": []}
    # The recording hooks go in before the integrity snapshot that _load_role
    # takes and stay until after its last check, so the evaluator's bindings
    # are identical at every comparison.
    recorder, original, observed, previous_profile = None, None, [], None
    if language in ("triton", "gluon"):
        recorder = _Capture(output)
        original = recorder.install()
    elif language == "flydsl":
        previous_profile = sys.getprofile()
        sys.setprofile(_flydsl_observer(observed))
    try:
        # FlyDSL's launch audit would replace this lane's profile hook; the
        # call keeps its integrity check and ran audited in correctness.
        call, implementation, check = _load_role(contract, role, enforce=language != "flydsl")
        if recorder is not None:
            _run_once(contract, call, case, device)  # warm-up: compile and finish any autotuning
            recorder.active = True
            _run_once(contract, call, case, device)
            recorder.active = False
            manifest["launches"] = recorder.launches
            manifest["skipped"] = recorder.skipped
            manifest["code_objects"] = [{"code_object": item["hsaco"], "sha256": item["hsaco_sha256"],
                                         "kernels": [item["kernel_name"]]} for item in recorder.launches]
        elif language == "hip":
            _run_once(contract, call, case, device)
            library = Path(implementation.get("library") or "")
            objects = hip_code_objects(library, output) if library.is_file() else []
            manifest["code_objects"] = [{"code_object": item.name,
                                         "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                                         "kernels": None} for item in objects]
            if not objects:
                manifest["skipped"].append("no device code object could be unbundled from the HIP extension")
        else:
            _run_once(contract, call, case, device)
            manifest["code_objects"] = _flydsl_code_objects(observed, output)
            if not manifest["code_objects"]:
                manifest["skipped"].append("no FlyDSL code object was observed")
        if check is not None:
            check("sanitize")
    finally:
        if original is not None:
            from triton.runtime.jit import JITFunction

            JITFunction.run = original
        if language == "flydsl":
            sys.setprofile(previous_profile)
    (output / "capture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {"code_objects": len(manifest["code_objects"]), "launches": len(manifest["launches"]),
                "skipped": manifest["skipped"][:8]}
    return _report(role, "PASS", [case.identity() | {"status": "PASS"}], "capture", metadata)


_LANES = {"page_fault_guard": page_fault_guard, "csan": csan, "tilelens": tilelens, "fpsan": fpsan,
          "gpu_asan": gpu_asan, "capture": capture}


def run_lane(contract, role: str, lane: str) -> dict:
    # A sidecar's /tmp may be mounted noexec; build HIP extensions in its
    # scratch tree instead, where a built library can be loaded.
    extensions = os.environ.get("TORCH_EXTENSIONS_DIR")
    if extensions and not os.environ.get("AKA_KERNEL_CACHE_DIR"):
        os.environ["AKA_KERNEL_CACHE_DIR"] = str(Path(extensions) / "aka-kernel-cache")
    return _LANES[lane](contract, role)
