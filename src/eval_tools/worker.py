"""Unix-socket worker used by an isolated evaluation-tool sidecar."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import shutil
import socket
import socketserver
import stat
import subprocess
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .execution import (
    DEFAULT_KILL_GRACE_SECONDS,
    DEFAULT_LOG_LIMIT_BYTES,
    DEFAULT_TERM_GRACE_SECONDS,
    LinuxProcessContainment,
    enable_child_subreaper,
    execute_command,
)
from .config import MAX_TOOL_TIMEOUT_SECONDS


MAX_REQUEST_BYTES = 1024 * 1024
MAX_ARGV_ENTRIES = 4096
MAX_ENV_ENTRIES = 4096
MAX_LOG_LIMIT_BYTES = 1024 * 1024 * 1024
TRITON_FPSAN_VERSION = "3.7.0+amd.rocm7.2.0.gitd0d77a509"
TRITON_ASAN_VERSIONS = {
    "3.6.0+git42270451",
    TRITON_FPSAN_VERSION,
}
POSITIVE_CONTROL_TIMEOUT_SECONDS = 120
POSITIVE_CONTROL_LOG_LIMIT_BYTES = 2 * 1024 * 1024
IMAGE_FRAMEWORK_ROOT = Path("/opt/aka-eval-tools")
FRAMEWORK_ROOT_ENV = "AKA_EVAL_TOOL_FRAMEWORK_ROOT"


class RequestValidationError(ValueError):
    pass


class PathValidationError(RequestValidationError):
    pass


def _reject_nonfinite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RequestValidationError("request contains a non-finite JSON number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_json(key)
            _reject_nonfinite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite_json(item)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _framework_provenance(input_root: Path) -> tuple[Path, str, Path]:
    """Resolve the trusted framework tree used by the worker and its probes.

    Production images set ``AKA_EVAL_TOOL_FRAMEWORK_ROOT`` to the immutable
    image-owned tree.  The module-location fallback exists only so the worker
    can still be exercised directly from a developer checkout.
    """

    worker_module = Path(__file__).resolve(strict=True)
    configured = os.environ.get(FRAMEWORK_ROOT_ENV)
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            raise RuntimeError(f"{FRAMEWORK_ROOT_ENV} must be an absolute path")
        framework_root = candidate.resolve(strict=True)
        expected_root = IMAGE_FRAMEWORK_ROOT.resolve(strict=False)
        if framework_root != expected_root:
            raise RuntimeError(
                f"{FRAMEWORK_ROOT_ENV} must resolve to image-owned {expected_root}"
            )
        source = "configured_image"
    elif IMAGE_FRAMEWORK_ROOT.is_dir():
        framework_root = IMAGE_FRAMEWORK_ROOT.resolve(strict=True)
        source = "image_default"
    else:
        framework_root = worker_module.parents[2]
        source = "local_module_fallback"

    package_root = (framework_root / "src" / "eval_tools").resolve(strict=True)
    try:
        worker_module.relative_to(package_root)
    except ValueError as error:
        raise RuntimeError(
            f"worker module {worker_module} is outside trusted framework {package_root}"
        ) from error

    if source != "local_module_fallback":
        resolved_input = input_root.resolve(strict=True)
        try:
            framework_root.relative_to(resolved_input)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"trusted framework root must not come from task input {resolved_input}"
            )
    return framework_root, source, worker_module


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _existing_path(*candidates: str | Path) -> str | None:
    for candidate in candidates:
        if not str(candidate):
            continue
        path = Path(candidate)
        if path.is_file():
            return str(path)
    return None


def _gpu_evidence() -> dict[str, Any]:
    executable = shutil.which("rocminfo") or (
        "/opt/rocm/bin/rocminfo" if Path("/opt/rocm/bin/rocminfo").is_file() else None
    )
    if executable is None:
        return {
            "gpu_arch": None,
            "xnack_supported": False,
            "rocminfo": None,
        }
    environment = os.environ.copy()
    environment["HSA_XNACK"] = "1"
    try:
        completed = subprocess.run(
            [executable],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "gpu_arch": None,
            "xnack_supported": False,
            "rocminfo": executable,
        }
    match = re.search(r"\bgfx[0-9a-z]+\b", completed.stdout, flags=re.IGNORECASE)
    arch = match.group(0).lower() if match else None
    output = completed.stdout.lower()
    # gfx942/gfx950 are XNACK-capable. Running rocminfo itself with HSA_XNACK=1
    # additionally verifies that this container can reach the selected agent.
    xnack = completed.returncode == 0 and (
        "xnack+" in output or arch in {"gfx942", "gfx950"}
    )
    return {
        "gpu_arch": arch,
        "xnack_supported": xnack,
        "rocminfo": executable,
        "rocminfo_returncode": completed.returncode,
    }


def _run_probe_step(
    name: str,
    argv: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / f"{name}.stdout.log"
    stderr_path = artifact_dir / f"{name}.stderr.log"
    try:
        execution = execute_command(
            argv,
            cwd=str(cwd),
            env=environment,
            timeout_seconds=POSITIVE_CONTROL_TIMEOUT_SECONDS,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_limit_bytes=POSITIVE_CONTROL_LOG_LIMIT_BYTES,
            stderr_limit_bytes=POSITIVE_CONTROL_LOG_LIMIT_BYTES,
            term_grace_seconds=5,
            kill_grace_seconds=2,
        )
        returncode = (
            execution.exit_code
            if execution.exit_code is not None
            else (-execution.signal if execution.signal is not None else None)
        )
        timed_out = execution.timed_out
        duration_ms = execution.duration_ms
    except (OSError, RuntimeError, ValueError) as error:
        returncode = None
        timed_out = False
        duration_ms = 0
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return {
        "command": argv,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_excerpt": stdout[-4000:],
        "stderr_excerpt": stderr[-4000:],
        "_stdout": stdout,
        "_stderr": stderr,
    }


def _public_step(step: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in step.items() if not str(key).startswith("_")}


def _fpsan_record(output: str) -> dict[str, Any] | None:
    prefix = "AKA_FPSAN_RESULT "
    for line in output.splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _positive_result(
    *,
    passed: bool,
    kind: str,
    detail: str,
    artifact_dir: Path,
    steps: Mapping[str, Mapping[str, Any]],
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "passed": bool(passed),
        "kind": kind,
        "detail": detail,
        "artifact_dir": str(artifact_dir),
        "steps": {name: _public_step(step) for name, step in steps.items()},
    }
    if controls is not None:
        value["controls"] = dict(controls)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    return value


def _triton_fpsan_positive(
    probe_root: Path, work_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    cache = work_dir / "triton-fpsan-cache"
    step = _run_probe_step(
        "known-mismatch",
        [shutil.which("python") or os.sys.executable, str(probe_root / "triton_fpsan_probe.py"), "wrong"],
        cwd=work_dir,
        environment={
            "TRITON_INSTRUMENTATION_MODE": "fpsan",
            "TRITON_CACHE_DIR": str(cache),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        artifact_dir=artifact_dir,
    )
    record = _fpsan_record(str(step["_stdout"]))
    metadata_files = list(cache.rglob("*.json")) if cache.is_dir() else []
    modes: list[str | None] = []
    for path in metadata_files:
        if path.name.startswith("__grp__"):
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict):
            modes.append(metadata.get("instrumentation_mode"))
    mismatch = bool(
        record
        and record.get("reference_digest")
        and record.get("candidate_digest")
        and record["reference_digest"] != record["candidate_digest"]
    )
    passed = step["returncode"] == 0 and mismatch and len(modes) >= 2 and all(
        mode == "fpsan" for mode in modes
    )
    return _positive_result(
        passed=passed,
        kind="triton_fpsan_known_mismatch",
        detail=(
            "known numerical mismatch produced distinct digests and every compiled kernel metadata record attested instrumentation_mode=fpsan"
            if passed
            else "known mismatch did not produce both distinct digests and compiler metadata attesting fpsan mode"
        ),
        artifact_dir=artifact_dir,
        steps={"known_mismatch": step},
    )


def _asan_environment(evidence: Mapping[str, Any]) -> dict[str, str]:
    host_preload = str(evidence.get("host_asan_preload") or "")
    hip_runtime = str(evidence.get("hip_asan_runtime") or "")
    host_lib_dir = str(evidence.get("host_asan_lib_dir") or "")
    runtime_dir = str(evidence.get("asan_runtime_dir") or "")
    normal_rocm_lib = str(evidence.get("normal_rocm_lib_dir") or "")
    inherited_preload = os.environ.get("LD_PRELOAD", "")
    inherited_libraries = os.environ.get("LD_LIBRARY_PATH", "")
    preload = ":".join(
        value for value in (host_preload, hip_runtime, inherited_preload) if value
    )
    library_path = ":".join(
        value
        for value in (
            host_lib_dir,
            runtime_dir,
            normal_rocm_lib,
            inherited_libraries,
        )
        if value
    )
    return {
        "HSA_XNACK": "1",
        "HSA_DISABLE_FRAGMENT_ALLOCATOR": "1",
        "AMD_PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        "PYTORCH_NO_HIP_MEMORY_CACHING": "1",
        "AMDGCN_USE_BUFFER_OPS": "0",
        "ASAN_OPTIONS": "detect_leaks=0,alloc_dealloc_mismatch=0",
        "LD_PRELOAD": preload,
        "LD_LIBRARY_PATH": library_path,
    }


def _gpu_asan_positive(
    evidence: Mapping[str, Any], probe_root: Path, work_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    binary = work_dir / "gpu_asan_probe"
    arch = str(evidence.get("gpu_arch") or "gfx950").split(":", 1)[0]
    steps: dict[str, dict[str, Any]] = {}
    steps["hip_compile"] = _run_probe_step(
        "hip-compile",
        [
            hipcc,
            "-O2",
            "-fsanitize=address",
            "-shared-libsan",
            f"--offload-arch={arch}:xnack+",
            str(probe_root / "gpu_asan_probe.hip"),
            "-o",
            str(binary),
        ],
        cwd=work_dir,
        environment={},
        artifact_dir=artifact_dir,
    )
    asan_env = _asan_environment(evidence)
    if steps["hip_compile"]["returncode"] == 0:
        steps["hip_safe"] = _run_probe_step(
            "hip-safe", [str(binary)], cwd=work_dir, environment=asan_env, artifact_dir=artifact_dir
        )
        steps["hip_oob"] = _run_probe_step(
            "hip-oob", [str(binary), "oob"], cwd=work_dir, environment=asan_env, artifact_dir=artifact_dir
        )
    hip_safe = steps.get("hip_safe", {})
    hip_oob = steps.get("hip_oob", {})
    hip_report = str(hip_oob.get("_stdout", "")) + str(hip_oob.get("_stderr", ""))
    hip_passed = (
        hip_safe.get("returncode") == 0
        and "SAFE_RUN_COMPLETED" in str(hip_safe.get("_stdout", ""))
        and "AddressSanitizer" in hip_report
        and "buffer-overflow" in hip_report
    )

    triton_env = {
        **asan_env,
        "TRITON_ENABLE_ASAN": "1",
        "TRITON_CACHE_DIR": str(work_dir / "triton-asan-cache"),
    }
    triton_probe = str(probe_root / "triton_asan_probe.py")
    steps["triton_safe"] = _run_probe_step(
        "triton-safe",
        [shutil.which("python") or os.sys.executable, triton_probe],
        cwd=work_dir,
        environment=triton_env,
        artifact_dir=artifact_dir,
    )
    steps["triton_oob"] = _run_probe_step(
        "triton-oob",
        [shutil.which("python") or os.sys.executable, triton_probe, "oob"],
        cwd=work_dir,
        environment=triton_env,
        artifact_dir=artifact_dir,
    )
    triton_report = str(steps["triton_oob"].get("_stdout", "")) + str(
        steps["triton_oob"].get("_stderr", "")
    )
    triton_passed = (
        bool(evidence.get("triton_asan"))
        and steps["triton_safe"].get("returncode") == 0
        and "SAFE_RUN_COMPLETED" in str(steps["triton_safe"].get("_stdout", ""))
        and "AddressSanitizer" in triton_report
        and "buffer-overflow" in triton_report
    )
    controls = {
        "hip": {"passed": hip_passed, "kind": "gpu_asan_hip_oob"},
        "triton": {"passed": triton_passed, "kind": "gpu_asan_triton_oob"},
    }
    return _positive_result(
        passed=hip_passed and triton_passed,
        kind="gpu_asan_known_oob",
        detail=f"HIP OOB detected={hip_passed}; Triton OOB detected={triton_passed}",
        artifact_dir=artifact_dir,
        steps=steps,
        controls=controls,
    )


def _rocjitsu_positive(
    evidence: Mapping[str, Any], probe_root: Path, work_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    binary = work_dir / "rocjitsu_race_probe"
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    steps: dict[str, dict[str, Any]] = {}
    steps["compile"] = _run_probe_step(
        "compile",
        [hipcc, "-O2", "--offload-arch=gfx950", str(probe_root / "rocjitsu_race_probe.hip"), "-o", str(binary)],
        cwd=work_dir,
        environment={},
        artifact_dir=artifact_dir,
    )
    tool_binary = str(evidence.get("rocjitsu_binary") or "")
    config = str(evidence.get("config_path") or "")
    environment = {
        "RJ_RACE": "1",
        "RJ_LOG": "1",
        "RJ_SINKS": "stderr",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }
    if steps["compile"]["returncode"] == 0 and tool_binary and config:
        base = [tool_binary, "--config", config, "--", str(binary)]
        steps["safe"] = _run_probe_step(
            "safe", base, cwd=work_dir, environment=environment, artifact_dir=artifact_dir
        )
        steps["racy"] = _run_probe_step(
            "racy", [*base, "racy"], cwd=work_dir, environment=environment, artifact_dir=artifact_dir
        )
    safe_text = str(steps.get("safe", {}).get("_stdout", "")) + str(steps.get("safe", {}).get("_stderr", ""))
    racy_text = str(steps.get("racy", {}).get("_stdout", "")) + str(steps.get("racy", {}).get("_stderr", ""))
    passed = (
        steps.get("safe", {}).get("returncode") == 0
        and "RACE type=" not in safe_text
        and "RACE type=LDS" in racy_text
    )
    return _positive_result(
        passed=passed,
        kind="rocjitsu_known_lds_race",
        detail="known LDS race was detected and barrier-protected control was clean" if passed else "rocJITsu did not distinguish the known safe/racy pair",
        artifact_dir=artifact_dir,
        steps=steps,
    )


def _hip_fpsan_positive(
    evidence: Mapping[str, Any], probe_root: Path, work_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    binary = work_dir / "hip_fpsan_probe"
    include_dir = str(evidence.get("include_dir") or "")
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    steps: dict[str, dict[str, Any]] = {}
    steps["compile"] = _run_probe_step(
        "compile",
        [hipcc, "-O2", "--offload-arch=gfx950", f"-I{include_dir}", str(probe_root / "hip_fpsan_probe.hip"), "-o", str(binary)],
        cwd=work_dir,
        environment={},
        artifact_dir=artifact_dir,
    )
    if steps["compile"]["returncode"] == 0:
        steps["equivalent"] = _run_probe_step(
            "equivalent", [str(binary)], cwd=work_dir, environment={}, artifact_dir=artifact_dir
        )
        steps["mismatch"] = _run_probe_step(
            "mismatch", [str(binary), "wrong"], cwd=work_dir, environment={}, artifact_dir=artifact_dir
        )
    equivalent = _fpsan_record(str(steps.get("equivalent", {}).get("_stdout", "")))
    mismatch = _fpsan_record(str(steps.get("mismatch", {}).get("_stdout", "")))
    passed = bool(
        steps.get("equivalent", {}).get("returncode") == 0
        and steps.get("mismatch", {}).get("returncode") == 0
        and equivalent
        and mismatch
        and equivalent.get("instrumented") is True
        and mismatch.get("instrumented") is True
        and equivalent.get("reference_digest") == equivalent.get("candidate_digest")
        and mismatch.get("reference_digest") != mismatch.get("candidate_digest")
    )
    return _positive_result(
        passed=passed,
        kind="hip_fpsan_known_mismatch",
        detail="ported HIP-FpSan control distinguished equivalent and known-wrong expressions" if passed else "HIP-FpSan did not distinguish the known equivalent/mismatch pair",
        artifact_dir=artifact_dir,
        steps=steps,
    )


def positive_control_evidence(
    tool: str,
    evidence: Mapping[str, Any],
    *,
    framework_root: Path,
    scratch_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    instance = os.environ.get("AKA_EVAL_TOOL_INSTANCE", "standalone")
    instance = "".join(character if character.isalnum() or character in "_.-" else "_" for character in instance)
    artifact_dir = artifact_root / "_eval_tool_runtime" / instance / tool / "positive-control"
    work_dir = scratch_root / "positive-control" / tool
    work_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("AKA_EVAL_TOOL_SKIP_POSITIVE_CONTROL") == "1":
        return _positive_result(
            passed=False,
            kind="skipped",
            detail="positive control explicitly skipped; it must not satisfy a required policy",
            artifact_dir=artifact_dir,
            steps={},
        )
    probe_root = framework_root / "src" / "eval_tools" / "probes"
    if not probe_root.is_dir():
        return _positive_result(
            passed=False,
            kind="probe_missing",
            detail=f"synthetic probe directory is missing: {probe_root}",
            artifact_dir=artifact_dir,
            steps={},
        )
    try:
        if tool == "triton_fpsan":
            return _triton_fpsan_positive(probe_root, work_dir, artifact_dir)
        if tool == "gpu_asan":
            return _gpu_asan_positive(evidence, probe_root, work_dir, artifact_dir)
        if tool == "rocjitsu":
            return _rocjitsu_positive(evidence, probe_root, work_dir, artifact_dir)
        if tool == "hip_fpsan":
            return _hip_fpsan_positive(evidence, probe_root, work_dir, artifact_dir)
    except Exception as error:
        return _positive_result(
            passed=False,
            kind="probe_error",
            detail=f"{type(error).__name__}: {error}",
            artifact_dir=artifact_dir,
            steps={},
        )
    return _positive_result(
        passed=False,
        kind="not_applicable",
        detail=f"no positive control is defined for tool {tool}",
        artifact_dir=artifact_dir,
        steps={},
    )


def runtime_evidence(
    tool: str,
    *,
    input_root: Path,
    scratch_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Attest tool-image assets and a synthetic known-bug control."""

    framework_root, framework_source, worker_module = _framework_provenance(input_root)
    evidence: dict[str, Any] = {
        "runtime_ref": os.environ.get("AKA_EVAL_TOOL_RUNTIME_REF"),
        "framework_root": str(framework_root),
        "framework_source": framework_source,
        "worker_module": str(worker_module),
        "worker_module_sha256": _sha256_file(worker_module),
        "probe_manifest": {
            path.name: _sha256_file(path)
            for path in sorted(
                (framework_root / "src" / "eval_tools" / "probes").iterdir()
            )
            if path.is_file() and path.name != "__init__.py"
        },
    }
    if tool == "triton_fpsan":
        version = _package_version("triton")
        kernels_version = _package_version("triton-kernels")
        evidence.update({
            "triton_version": version,
            "triton_kernels_version": kernels_version,
            "triton_fpsan": version == TRITON_FPSAN_VERSION,
            "triton_asan": version in TRITON_ASAN_VERSIONS,
            "expected_triton_version": TRITON_FPSAN_VERSION,
        })
    elif tool == "gpu_asan":
        runtime_dir = Path(os.environ.get("AKA_GPU_ASAN_RUNTIME_DIR", "/opt/rocm-7.2.0/lib/asan"))
        hip_runtime = Path(
            os.environ.get(
                "AKA_GPU_ASAN_HIP_RUNTIME", str(runtime_dir / "libamdhip64.so")
            )
        )
        required_libraries = (
            hip_runtime,
            runtime_dir / "libhsa-runtime64.so",
            runtime_dir / "libamd_comgr.so",
        )
        preload_candidates = sorted(
            Path("/opt/rocm-7.2.0/lib/llvm/lib/clang").glob(
                "*/lib/linux/libclang_rt.asan-x86_64.so"
            )
        )
        configured_preload = os.environ.get("AKA_GPU_ASAN_HOST_PRELOAD")
        host_preload = (
            Path(configured_preload)
            if configured_preload
            else (preload_candidates[-1] if preload_candidates else None)
        )
        triton_version = _package_version("triton")
        evidence.update({
            "asan_runtime_dir": str(runtime_dir),
            "hip_asan_runtime": str(hip_runtime) if hip_runtime.is_file() else None,
            "host_asan_preload": str(host_preload) if host_preload and host_preload.is_file() else None,
            "host_asan_lib_dir": str(host_preload.parent) if host_preload and host_preload.is_file() else None,
            "normal_rocm_lib_dir": "/opt/rocm-7.2.0/lib",
            "asan_libraries": {path.name: path.is_file() for path in required_libraries},
            "triton_version": triton_version,
            "triton_asan": triton_version in TRITON_ASAN_VERSIONS,
        })
        evidence.update(_gpu_evidence())
    elif tool == "rocjitsu":
        binary = os.environ.get("AKA_ROCJITSU_BINARY") or shutil.which("rocjitsu")
        config = _existing_path(
            os.environ.get("AKA_ROCJITSU_CONFIG", ""),
            "/opt/rocjitsu/share/rocjitsu/configs/gfx950_cdna4.json",
            "/opt/rocjitsu/share/rocjitsu/configs/gfx950_cdna4_kmd.json",
        )
        evidence.update({
            "rocjitsu_binary": binary if binary and Path(binary).is_file() else None,
            "config_path": config,
            "target_arch": "gfx950",
            "rocjitsu_commit": os.environ.get("AKA_ROCJITSU_COMMIT"),
        })
    elif tool == "hip_fpsan":
        include_dir = Path(os.environ.get("AKA_HIP_FPSAN_INCLUDE_DIR", "/opt/hip-fpsan/include"))
        public_header = include_dir / "fpsan" / "fpsan.hpp"
        evidence.update({
            "include_dir": str(include_dir) if public_header.is_file() else None,
            "public_header": str(public_header) if public_header.is_file() else None,
            "hip_fpsan_headers": public_header.is_file(),
            "hip_fpsan_commit": os.environ.get("AKA_HIP_FPSAN_COMMIT"),
        })
    evidence["positive_control"] = positive_control_evidence(
        tool,
        evidence,
        framework_root=framework_root,
        scratch_root=scratch_root,
        artifact_root=artifact_root,
    )
    return evidence


def _safe_tool_id(tool: str) -> str:
    if not tool or not all(character.isalnum() or character in "_.-" for character in tool):
        raise ValueError(f"invalid tool id: {tool!r}")
    return tool


def _relative_path(value: Any, *, field: str, allow_dot: bool = True) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PathValidationError(f"{field} must be a non-empty relative path")
    if "\x00" in value or "\\" in value:
        raise PathValidationError(f"{field} contains an invalid character")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise PathValidationError(f"{field} escapes its declared root")
    if not allow_dot and path.as_posix() in ("", "."):
        raise PathValidationError(f"{field} must name a child path")
    return path


def _resolve_below(
    root: Path,
    value: Any,
    *,
    field: str,
    must_exist: bool,
    directory: bool,
) -> Path:
    relative = _relative_path(value, field=field)
    root = root.resolve(strict=True)
    candidate = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PathValidationError(f"{field} resolves outside its declared root") from error
    if must_exist and not candidate.exists():
        raise PathValidationError(f"{field} does not exist: {relative.as_posix()}")
    if must_exist and directory and not candidate.is_dir():
        raise PathValidationError(f"{field} is not a directory: {relative.as_posix()}")
    return candidate


def _positive_number(value: Any, *, field: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise RequestValidationError(f"{field} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RequestValidationError(f"{field} must be a positive number") from error
    if not math.isfinite(number) or number <= 0 or number > maximum:
        raise RequestValidationError(f"{field} must be in (0, {maximum}]")
    return number


def _nonnegative_number(value: Any, *, field: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise RequestValidationError(f"{field} must be a non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RequestValidationError(f"{field} must be a non-negative number") from error
    if not math.isfinite(number) or number < 0 or number > maximum:
        raise RequestValidationError(f"{field} must be in [0, {maximum}]")
    return number


def _log_limit(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise RequestValidationError(f"{field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise RequestValidationError(f"{field} must be a non-negative integer") from error
    if number < 0 or number > MAX_LOG_LIMIT_BYTES:
        raise RequestValidationError(f"{field} must be in [0, {MAX_LOG_LIMIT_BYTES}]")
    return number


def _argv(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > MAX_ARGV_ENTRIES:
        raise RequestValidationError(
            f"argv must contain between 1 and {MAX_ARGV_ENTRIES} entries"
        )
    for argument in value:
        if not isinstance(argument, str) or "\x00" in argument:
            raise RequestValidationError("argv entries must be NUL-free strings")
    return list(value)


def _environment(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > MAX_ENV_ENTRIES:
        raise RequestValidationError(f"env must be a mapping with at most {MAX_ENV_ENTRIES} entries")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise RequestValidationError("env keys and values must be strings")
        if not key or "=" in key or "\x00" in key or "\x00" in item:
            raise RequestValidationError("env contains an invalid key or NUL byte")
        result[key] = item
    return result


class EvalToolServer(socketserver.UnixStreamServer):
    """Sequential server: one sidecar never runs two GPU commands concurrently."""

    allow_reuse_address = False

    def __init__(
        self,
        socket_path: Path,
        *,
        tool: str,
        input_root: Path,
        scratch_root: Path,
        artifact_root: Path,
        max_timeout_seconds: float,
    ) -> None:
        self.tool = _safe_tool_id(tool)
        self.input_root = input_root.resolve(strict=True)
        self.scratch_root = scratch_root.resolve(strict=True)
        self.artifact_root = artifact_root.resolve(strict=True)
        self.max_timeout_seconds = max_timeout_seconds
        self.runtime_evidence = runtime_evidence(
            self.tool,
            input_root=self.input_root,
            scratch_root=self.scratch_root,
            artifact_root=self.artifact_root,
        )
        super().__init__(str(socket_path), EvalToolRequestHandler)

    def root(self, name: str) -> Path:
        roots = {
            "input": self.input_root,
            "scratch": self.scratch_root,
            "artifact": self.artifact_root,
        }
        try:
            return roots[name]
        except KeyError as error:
            raise PathValidationError(f"unknown cwd root: {name!r}") from error

    def dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method == "health":
            positive = self.runtime_evidence.get("positive_control")
            control_passed = (
                isinstance(positive, Mapping) and positive.get("passed") is True
            )
            status = (
                "ready"
                if self.tool not in {"triton_fpsan", "gpu_asan", "rocjitsu", "hip_fpsan"}
                or control_passed
                else "degraded"
            )
            return {
                "status": status,
                "tool": self.tool,
                "pid": os.getpid(),
                "protocol_version": 1,
                "evidence": self.runtime_evidence,
            }
        if method == "execute":
            return self.execute(params)
        if method == "shutdown":
            threading.Thread(target=self.shutdown, daemon=True).start()
            return {"status": "stopping", "tool": self.tool}
        raise RequestValidationError(f"unsupported RPC method: {method!r}")

    def execute(self, params: Mapping[str, Any]) -> dict[str, Any]:
        argv = _argv(params.get("argv"))
        cwd_value = params.get("cwd", {"root": "input", "path": "."})
        if not isinstance(cwd_value, Mapping):
            raise RequestValidationError("cwd must contain root and relative path")
        cwd_root_name = cwd_value.get("root", "input")
        cwd_relative = cwd_value.get("path", ".")
        if not isinstance(cwd_root_name, str):
            raise RequestValidationError("cwd.root must be a string")
        cwd_root = self.root(cwd_root_name)
        cwd = _resolve_below(
            cwd_root,
            cwd_relative,
            field="cwd.path",
            must_exist=cwd_root_name == "input",
            directory=True,
        )
        if cwd_root_name in ("scratch", "artifact"):
            cwd.mkdir(parents=True, exist_ok=True)

        artifact_dir = _resolve_below(
            self.artifact_root,
            params.get("artifact_dir"),
            field="artifact_dir",
            must_exist=False,
            directory=True,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Resolve once more after mkdir so a pre-existing/surprising symlink in
        # the created path cannot change containment between validation and use.
        artifact_dir = _resolve_below(
            self.artifact_root,
            str(artifact_dir.relative_to(self.artifact_root)),
            field="artifact_dir",
            must_exist=True,
            directory=True,
        )
        stdout_path = artifact_dir / "stdout.log"
        stderr_path = artifact_dir / "stderr.log"

        timeout_seconds = _positive_number(
            params.get("timeout_s", 300),
            field="timeout_s",
            maximum=self.max_timeout_seconds,
        )
        stdout_limit = _log_limit(
            params.get("stdout_limit_bytes", DEFAULT_LOG_LIMIT_BYTES),
            field="stdout_limit_bytes",
        )
        stderr_limit = _log_limit(
            params.get("stderr_limit_bytes", DEFAULT_LOG_LIMIT_BYTES),
            field="stderr_limit_bytes",
        )
        term_grace = _nonnegative_number(
            params.get("term_grace_s", DEFAULT_TERM_GRACE_SECONDS),
            field="term_grace_s",
            maximum=60,
        )
        kill_grace = _nonnegative_number(
            params.get("kill_grace_s", DEFAULT_KILL_GRACE_SECONDS),
            field="kill_grace_s",
            maximum=60,
        )
        result = execute_command(
            argv,
            cwd=cwd,
            env=_environment(params.get("env")),
            timeout_seconds=timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_limit_bytes=stdout_limit,
            stderr_limit_bytes=stderr_limit,
            term_grace_seconds=term_grace,
            kill_grace_seconds=kill_grace,
            containment=LinuxProcessContainment.capture(fatal_on_failure=True),
        ).to_dict()
        for stream_name in ("stdout", "stderr"):
            stream = result[stream_name]
            assert isinstance(stream, dict)
            stream_path = Path(str(stream["path"])).resolve(strict=False)
            stream["path"] = stream_path.relative_to(self.artifact_root).as_posix()
        return {
            "tool": self.tool,
            "execution": result,
        }


class EvalToolRequestHandler(socketserver.StreamRequestHandler):
    server: EvalToolServer

    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        request_id: Any = None
        if len(raw) > MAX_REQUEST_BYTES:
            self._write_error(request_id, "REQUEST_TOO_LARGE", "request exceeded size limit")
            return
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise RequestValidationError("request must be an object")
            request_id = request.get("id")
            if not isinstance(request_id, str) or not request_id:
                raise RequestValidationError("request id must be a non-empty string")
            _reject_nonfinite_json(request)
            method = request.get("method")
            params = request.get("params", {})
            if not isinstance(method, str) or not method:
                raise RequestValidationError("method must be a non-empty string")
            if not isinstance(params, Mapping):
                raise RequestValidationError("params must be an object")
            result = self.server.dispatch(method, params)
            self._write({"id": request_id, "ok": True, "result": result})
        except json.JSONDecodeError:
            self._write_error(request_id, "INVALID_JSON", "request was not valid JSON")
        except PathValidationError as error:
            self._write_error(request_id, "INVALID_PATH", str(error))
        except RequestValidationError as error:
            self._write_error(request_id, "INVALID_REQUEST", str(error))
        except Exception as error:
            self._write_error(
                request_id,
                "TOOL_ERROR",
                f"worker execution failed: {type(error).__name__}: {error}",
            )

    def _write_error(self, request_id: Any, code: str, message: str) -> None:
        self._write(
            {
                "id": request_id,
                "ok": False,
                "error": {"code": code, "message": message},
            }
        )

    def _write(self, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.wfile.write(payload + b"\n")
        self.wfile.flush()


def _prepare_socket_path(socket_path: Path) -> None:
    if not socket_path.is_absolute():
        raise ValueError("--socket must be an absolute path")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = socket_path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"refusing to replace non-socket path: {socket_path}")
    socket_path.unlink()


def serve(
    *,
    tool: str,
    socket_path: Path,
    input_root: Path,
    scratch_root: Path,
    artifact_root: Path,
    max_timeout_seconds: float,
) -> None:
    enable_child_subreaper()
    _prepare_socket_path(socket_path)
    for root in (input_root, scratch_root, artifact_root):
        root.mkdir(parents=True, exist_ok=True)

    server = EvalToolServer(
        socket_path,
        tool=tool,
        input_root=input_root,
        scratch_root=scratch_root,
        artifact_root=artifact_root,
        max_timeout_seconds=max_timeout_seconds,
    )
    os.chmod(socket_path, 0o600)

    def request_stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        try:
            mode = socket_path.lstat().st_mode
            if stat.S_ISSOCK(mode):
                socket_path.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated evaluation-tool worker")
    parser.add_argument("--tool", required=True)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--max-timeout-s",
        type=float,
        default=float(MAX_TOOL_TIMEOUT_SECONDS),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not math.isfinite(args.max_timeout_s)
        or args.max_timeout_s <= 0
        or args.max_timeout_s > MAX_TOOL_TIMEOUT_SECONDS
    ):
        raise SystemExit(
            f"--max-timeout-s must be in (0, {MAX_TOOL_TIMEOUT_SECONDS}]"
        )
    serve(
        tool=args.tool,
        socket_path=args.socket,
        input_root=args.input_root,
        scratch_root=args.scratch_root,
        artifact_root=args.artifact_root,
        max_timeout_seconds=args.max_timeout_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
