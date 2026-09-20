"""Task-local capture-runtime checks, independent of the arena source tree.

The copy under _support is canonical; scripts/runtime_preflight.py is copied
into each isolated task. Call preflight(config) before correctness/performance.
Compilation may still perform its CPU-only syntax check separately.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re


def runtime_requirements(config: dict, task_dir: Path | None = None) -> dict:
    """Describe the contract from task metadata without importing GPU packages."""
    metadata = config.get("headkernel") or {}
    runtime = metadata.get("runtime") or {}
    image = metadata.get("docker")
    expected_ids = set()
    if runtime.get("expected_image_id"):
        expected_ids.add(str(runtime["expected_image_id"]))
    source_commits = {}
    capture_backend_modules = set()
    task_dir = task_dir or Path(__file__).resolve().parents[1]
    capture_path = task_dir / "ut/meta.json"
    if capture_path.is_file():
        with capture_path.open(encoding="utf-8") as stream:
            capture = json.load(stream)
        capture_backend_modules.update(name for name in capture.get("candidate_backends", [])
                                       if name in ("flydsl", "tilelang", "triton"))
        for section, image_field, id_field in (
            ("source_provenance", "runtime_image", "runtime_image_id"),
            ("baseline_frozen_by", "image", "image_id"),
        ):
            evidence = capture.get(section) or {}
            if evidence.get(id_field):
                if evidence.get(image_field) != image:
                    raise ValueError(f"{section} image evidence disagrees with headkernel.docker")
                expected_ids.add(str(evidence[id_field]))
        provenance = capture.get("source_provenance") or {}
        if provenance.get("repo") and provenance.get("repo_commit"):
            source_commits[provenance["repo"]] = provenance["repo_commit"]
    provenance = metadata.get("runtime_source_provenance") or {}
    if provenance.get("repo") and provenance.get("commit"):
        source_commits[provenance["repo"]] = provenance["commit"]
    if len(expected_ids) > 1:
        raise ValueError("Task metadata contains conflicting captured Docker image IDs")
    expected_image_id = next(iter(expected_ids), None)
    if expected_image_id and re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_id) is None:
        raise ValueError("Captured Docker image ID must be a complete sha256 image/config ID")
    required_modules = {"torch", "sglang", "triton", "aiter"}
    required_modules.update(capture_backend_modules)
    backend = str(metadata.get("backend", "")).lower()
    required_modules.update(name for name in ("flydsl", "tilelang") if name in backend)
    required_modules.update(runtime.get("required_modules") or [])
    versions = {str(name): str(version) for name, version in (runtime.get("package_versions") or {}).items()}
    if runtime.get("torch_version"):
        versions["torch"] = str(runtime["torch_version"])
    sglang_version = runtime.get("sglang_version")
    if sglang_version is None and isinstance(image, str):
        match = re.search(r":v(\d+\.\d+\.\d+)(?:-|$)", image)
        sglang_version = match.group(1) if match else None
    if sglang_version is not None:
        versions["sglang"] = str(sglang_version)
    required_modules.update(versions)
    return {
        "image": image,
        "expected_image_id": expected_image_id,
        "source_commits": source_commits,
        "gpu_arch": "gfx950",
        "hip_version": runtime.get("hip_version", "7.2.x"),
        "required_modules": sorted(required_modules),
        "package_versions": versions,
        "environment": ({"TVM_FFI_DISABLE_TORCH_C_DLPACK": "1"}
                        if sglang_version == "0.5.18" else {}),
        "required_model_types": list(runtime.get("required_model_types") or []),
    }


def _version(module, distribution: str) -> str | None:
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def preflight(config: dict, task_dir: Path | None = None) -> dict:
    """Return a serializable report; mismatches never produce a passing report.

    Optional headkernel.runtime fields tighten the capture contract when its
    provenance supplies those facts: sglang_version, torch_version,
    required_modules, package_versions, hip_version, expected_image_id, and
    required_model_types. The host runner resolves the selected tag and launches
    by Docker's local image/config ID. That ID is not a registry manifest digest.
    """
    metadata = config.get("headkernel") or {}
    runtime = metadata.get("runtime") or {}
    requirements = runtime_requirements(config, task_dir)
    expected_image = metadata.get("docker")
    selected_image = os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE")
    selected_image_id = os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID")
    report = {
        "status": "fail",
        "expected_image": expected_image,
        "selected_image": selected_image,
        "expected_image_id": requirements["expected_image_id"],
        "selected_image_id": selected_image_id,
        "registry_repo_digests": [],
        "captured_source_commits": requirements["source_commits"],
        "required_arch": "gfx950",
        "architecture": None,
        "versions": {},
        "environment": {name: os.environ.get(name) for name in requirements["environment"]},
        "errors": [],
    }
    errors = report["errors"]
    for name, expected in requirements["environment"].items():
        if os.environ.get(name) != expected:
            errors.append(f"Capture runtime requires {name}={expected}; use the cohort launcher")
    if not isinstance(expected_image, str) or not expected_image:
        errors.append("Task config does not declare headkernel.docker")
    if os.environ.get("AGENT_KERNEL_ARENA_DOCKER") != "1":
        errors.append("Run this task through the Docker runner")
    if selected_image != expected_image:
        errors.append(f"Capture image mismatch: expected {expected_image!r}, "
                      f"runner selected {selected_image!r}; use the cohort launcher")
    if not selected_image_id or re.fullmatch(r"sha256:[0-9a-f]{64}", selected_image_id) is None:
        errors.append("The Docker runner did not supply a verified image ID; use the cohort launcher")
    elif requirements["expected_image_id"] and selected_image_id != requirements["expected_image_id"]:
        errors.append(f"Capture image ID mismatch: expected {requirements['expected_image_id']}, "
                      f"runner launched {selected_image_id}")
    try:
        digests = json.loads(os.environ.get("AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS", "[]"))
        if not isinstance(digests, list) or any(not isinstance(value, str) for value in digests):
            raise ValueError("expected a list of registry references")
        report["registry_repo_digests"] = digests
    except (ValueError, TypeError) as exc:
        errors.append(f"Invalid runner registry-digest metadata: {exc}")
    if errors:
        return report

    # Fail before importing source overlays or executing a benchmark. Import
    # paths and versions are recorded even when a required dependency fails.
    modules = {}
    for name in requirements["required_modules"]:
        try:
            module = importlib.import_module(name)
            modules[name] = module
            report["versions"][name] = _version(module, name)
        except Exception as exc:
            errors.append(f"Required runtime module {name!r} failed to import: "
                          f"{type(exc).__name__}: {exc}")

    for name, expected in requirements["package_versions"].items():
        actual = report["versions"].get(name)
        # SGLang local/build metadata is allowed when only a release is pinned.
        observed = actual.split("+", 1)[0] if name == "sglang" and actual and "+" not in expected else actual
        if observed != expected:
            label = "SGLang" if name == "sglang" else name
            errors.append(f"{label} version mismatch: expected {expected}, got {actual}")

    torch = modules.get("torch")
    if torch is not None:
        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        report["versions"]["hip"] = hip_version
        if not hip_version or re.match(r"^7\.2(?:\.|$)", str(hip_version)) is None:
            errors.append(f"The capture requires ROCm/HIP 7.2; got {hip_version!r}")
        if runtime.get("hip_version") and str(hip_version) != str(runtime["hip_version"]):
            errors.append(f"HIP version mismatch: expected {runtime['hip_version']}, got {hip_version}")
        try:
            if not torch.cuda.is_available():
                errors.append("A ROCm GPU is not available to PyTorch")
            else:
                properties = torch.cuda.get_device_properties(torch.cuda.current_device())
                architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
                report["architecture"] = architecture
                if architecture != "gfx950":
                    errors.append(f"GPU architecture mismatch: expected gfx950, got {architecture!r}")
        except Exception as exc:
            errors.append(f"GPU inspection failed: {type(exc).__name__}: {exc}")

    target = metadata.get("target_callable")
    if not isinstance(target, str) or ":" not in target:
        errors.append("Task config does not declare a module:callable target")
    else:
        module_name, symbol = target.split(":", 1)
        try:
            value = importlib.import_module(module_name)
            for part in symbol.split("."):
                value = getattr(value, part)
            if not callable(value):
                errors.append(f"The captured target is not callable: {target}")
        except Exception as exc:
            errors.append(f"Captured runtime target {target!r} is unavailable: "
                          f"{type(exc).__name__}: {exc}")

    for model_type in runtime.get("required_model_types") or []:
        try:
            module = importlib.import_module("transformers.models.auto.configuration_auto")
            module.CONFIG_MAPPING[model_type]
        except Exception as exc:
            errors.append(f"Required model architecture {model_type!r} is unavailable; "
                          f"install the documented architecture patch in the runtime "
                          f"({type(exc).__name__}: {exc})")
    report["status"] = "fail" if errors else "ok"
    return report


def require_runtime(config: dict, task_dir: Path | None = None) -> dict:
    """Persist the host identity and observed runtime, then reject mismatches."""
    task_dir = task_dir or Path(__file__).resolve().parents[1]
    report = preflight(config, task_dir)
    report_path = task_dir / "build/runtime_preflight.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "ok":
        raise RuntimeError("Runtime preflight failed: " + "; ".join(report["errors"]))
    return report
