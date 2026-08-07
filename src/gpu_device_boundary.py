#!/usr/bin/env python3
"""Fail-closed ROCm device binding for formal matched campaigns.

The host resolver joins rocm-smi's physical-card identity with the KFD topology.
This is deliberately not an ordinal-to-render-node calculation: a physical GPU
may own multiple, non-contiguous render nodes when XCP partitioning is enabled.
The container verifier then proves that Docker exposed exactly that set.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


PLAN_SCHEMA = "agentkernelarena.formal-gpu-boundary-plan/v1"
RECEIPT_SCHEMA = "agentkernelarena.formal-gpu-boundary-receipt/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CARD = re.compile(r"card([0-9]+)")
_RENDER = re.compile(r"renderD([0-9]+)")
StatFn = Callable[[Path], os.stat_result]


class GpuBoundaryError(RuntimeError):
    """Raised when a formal GPU boundary cannot be proven."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _without_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "sha256"}


def _normalize_unique_id(raw: Any, label: str) -> str:
    text = str(raw).strip().lower()
    try:
        number = int(text, 16 if text.startswith("0x") else 10)
    except ValueError as error:
        raise GpuBoundaryError(f"{label} is not an integer: {raw!r}") from error
    if number <= 0 or number > 0xFFFFFFFFFFFFFFFF:
        raise GpuBoundaryError(f"{label} is outside the nonzero uint64 range")
    return f"0x{number:016x}"


def _positive_int(raw: Any, label: str) -> int:
    text = str(raw).strip()
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise GpuBoundaryError(f"{label} is not a canonical decimal integer: {raw!r}")
    number = int(text, 10)
    if number <= 0:
        raise GpuBoundaryError(f"{label} must be positive")
    return number


def _nonnegative_int(raw: Any, label: str) -> int:
    text = str(raw).strip()
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise GpuBoundaryError(f"{label} is not a canonical decimal integer: {raw!r}")
    number = int(text, 10)
    if number < 0:
        raise GpuBoundaryError(f"{label} must be nonnegative")
    return number


def _ordered_gpu_ids(values: Sequence[str]) -> list[str]:
    ordered = [str(value) for value in values]
    if not ordered or any(
        not value.isdigit() or str(int(value)) != value for value in ordered
    ):
        raise GpuBoundaryError("GPU IDs must be a nonempty ordered list of decimal IDs")
    if len(ordered) != len(set(ordered)):
        raise GpuBoundaryError("GPU IDs contain duplicates")
    return ordered


def parse_rocm_smi_inventory(raw: str | bytes | Mapping[str, Any]) -> dict[str, dict[str, str]]:
    if isinstance(raw, Mapping):
        decoded: Any = dict(raw)
    else:
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
            raise GpuBoundaryError("rocm-smi inventory is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise GpuBoundaryError("rocm-smi inventory must be a JSON object")

    parsed: dict[str, dict[str, str]] = {}
    for raw_key, raw_value in decoded.items():
        match = _CARD.fullmatch(str(raw_key))
        if match is None or not isinstance(raw_value, dict):
            continue
        host_id = match.group(1)
        required = ("Unique ID", "Serial Number", "Card Series", "GFX Version")
        if any(not str(raw_value.get(key, "")).strip() for key in required):
            raise GpuBoundaryError(f"rocm-smi identity is incomplete for card{host_id}")
        parsed[host_id] = {
            "unique_id": _normalize_unique_id(
                raw_value["Unique ID"], f"rocm-smi card{host_id} Unique ID"
            ),
            "serial_number": str(raw_value["Serial Number"]).strip(),
            "card_series": str(raw_value["Card Series"]).strip(),
            "gfx_version": str(raw_value["GFX Version"]).strip(),
        }
    if not parsed:
        raise GpuBoundaryError("rocm-smi inventory contains no cardN records")
    return parsed


def _parse_properties(path: Path) -> tuple[dict[str, str], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise GpuBoundaryError(f"cannot read KFD properties: {path}: {error}") from error
    properties: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(None, 1)
        if len(fields) != 2:
            raise GpuBoundaryError(f"malformed KFD property at {path}:{line_number}")
        key, value = fields
        if key in properties:
            raise GpuBoundaryError(f"duplicate KFD property {key!r} in {path}")
        properties[key] = value.strip()
    return properties, raw


def _numeric_topology_entries(topology_root: Path) -> list[Path]:
    try:
        entries = list(topology_root.iterdir())
    except OSError as error:
        raise GpuBoundaryError(f"cannot enumerate KFD topology: {topology_root}: {error}") from error
    numeric_entries = sorted(
        (entry for entry in entries if entry.name.isdigit()), key=lambda item: int(item.name)
    )
    if not numeric_entries:
        raise GpuBoundaryError(f"KFD topology has no numeric nodes: {topology_root}")
    return numeric_entries


def _read_topology_generation(topology_root: Path) -> int:
    generation_path = topology_root.parent / "generation_id"
    try:
        raw = generation_path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeError) as error:
        raise GpuBoundaryError(
            f"cannot read KFD topology generation: {generation_path}: {error}"
        ) from error
    generation = _nonnegative_int(text, f"{generation_path}")
    if generation > 0xFFFFFFFFFFFFFFFF:
        raise GpuBoundaryError(f"{generation_path} is outside the uint64 range")
    return generation


def _read_kfd_node(entry: Path) -> dict[str, Any] | None:
    properties_path = entry / "properties"
    gpu_id_path = entry / "gpu_id"
    properties, properties_bytes = _parse_properties(properties_path)
    try:
        gpu_id_bytes = gpu_id_path.read_bytes()
        raw_gpu_id = gpu_id_bytes.decode("ascii")
    except (OSError, UnicodeError) as error:
        raise GpuBoundaryError(
            f"cannot read KFD gpu_id: {gpu_id_path}: {error}"
        ) from error
    gpu_id = _nonnegative_int(raw_gpu_id, f"{gpu_id_path}")
    if gpu_id > 0xFFFFFFFF:
        raise GpuBoundaryError(f"{gpu_id_path} is outside the uint32 range")
    simd_count = _nonnegative_int(
        properties.get("simd_count", ""), f"{properties_path} simd_count"
    )
    if gpu_id == 0:
        zero_only = ("drm_render_minor", "vendor_id", "device_id", "gfx_target_version")
        malformed = simd_count != 0 or "unique_id" in properties
        for key in zero_only:
            malformed = malformed or _nonnegative_int(
                properties.get(key, ""), f"{properties_path} {key}"
            ) != 0
        if malformed:
            raise GpuBoundaryError(
                f"KFD non-GPU node contains GPU identity or resources: {entry}"
            )
        return None
    if simd_count == 0:
        raise GpuBoundaryError(
            f"KFD node has a nonzero gpu_id but no GPU SIMD resources: {entry}"
        )
    if "unique_id" not in properties or "drm_render_minor" not in properties:
        raise GpuBoundaryError(
            f"GPU KFD node lacks unique_id or drm_render_minor: {entry}"
        )
    render_minor = _positive_int(
        properties["drm_render_minor"], f"{properties_path} drm_render_minor"
    )
    return {
        "node_id": int(entry.name),
        "gpu_id": gpu_id,
        "unique_id": _normalize_unique_id(
            properties["unique_id"], f"{properties_path} unique_id"
        ),
        "drm_render_minor": render_minor,
        "gpu_id_sha256": hashlib.sha256(gpu_id_bytes).hexdigest(),
        "properties_sha256": hashlib.sha256(properties_bytes).hexdigest(),
    }


def _read_kfd_topology_snapshot(
    topology_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    generation_before = _read_topology_generation(topology_root)
    entries = _numeric_topology_entries(topology_root)
    nodes: list[dict[str, Any]] = []
    for entry in entries:
        node = _read_kfd_node(entry)
        if node is not None:
            nodes.append(node)
    generation_after = _read_topology_generation(topology_root)
    entries_after = _numeric_topology_entries(topology_root)
    if generation_after != generation_before or [entry.name for entry in entries_after] != [
        entry.name for entry in entries
    ]:
        raise GpuBoundaryError("KFD topology changed during evidence collection")
    if not nodes:
        raise GpuBoundaryError("KFD topology contains no complete GPU nodes")
    gpu_ids = [node["gpu_id"] for node in nodes]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise GpuBoundaryError("KFD topology contains duplicate nonzero gpu_id values")
    return nodes, generation_before


def read_kfd_topology(topology_root: Path) -> list[dict[str, Any]]:
    nodes, _ = _read_kfd_topology_snapshot(topology_root)
    return nodes


def _permission_denied(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, OSError) and current.errno in (errno.EACCES, errno.EPERM):
            return True
        current = current.__cause__ or current.__context__
    return False


def _read_selected_runtime_topology(
    topology_root: Path, selected: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    expected: dict[int, dict[str, Any]] = {}
    for render in selected["render_nodes"]:
        for planned in render["kfd_nodes"]:
            node_id = planned["node_id"]
            if node_id in expected:
                raise GpuBoundaryError("selected GPU plan repeats a KFD node")
            expected[node_id] = {
                **planned,
                "drm_render_minor": render["minor"],
            }

    generation_before = _read_topology_generation(topology_root)
    entries = _numeric_topology_entries(topology_root)
    observed: list[dict[str, Any]] = []
    for entry in entries:
        node_id = int(entry.name)
        try:
            node = _read_kfd_node(entry)
        except GpuBoundaryError as error:
            if node_id not in expected and _permission_denied(error):
                continue
            raise
        if node is None:
            continue
        planned = expected.get(node_id)
        if planned is None:
            raise GpuBoundaryError(
                f"formal worker can read an unexpected GPU KFD node: {entry}"
            )
        for key in (
            "gpu_id",
            "gpu_id_sha256",
            "properties_sha256",
            "drm_render_minor",
        ):
            if node[key] != planned[key]:
                raise GpuBoundaryError(
                    f"live selected KFD node {node_id} differs from planned {key}"
                )
        observed.append(node)
    generation_after = _read_topology_generation(topology_root)
    entries_after = _numeric_topology_entries(topology_root)
    if generation_after != generation_before or [entry.name for entry in entries_after] != [
        entry.name for entry in entries
    ]:
        raise GpuBoundaryError("KFD topology changed during runtime verification")
    if {node["node_id"] for node in observed} != set(expected):
        raise GpuBoundaryError("live KFD topology is missing a planned selected node")
    return sorted(observed, key=lambda node: node["node_id"]), generation_before


def _character_device(path: Path, stat_fn: StatFn) -> dict[str, int]:
    try:
        observed = stat_fn(path)
    except OSError as error:
        raise GpuBoundaryError(f"required device is unavailable: {path}: {error}") from error
    if not stat.S_ISCHR(observed.st_mode):
        raise GpuBoundaryError(f"required path is not a character device: {path}")
    return {"major": os.major(observed.st_rdev), "minor": os.minor(observed.st_rdev)}


def build_plan(
    ordered_gpu_ids: Sequence[str],
    rocm_smi_inventory: str | bytes | Mapping[str, Any],
    *,
    topology_root: Path = Path("/sys/class/kfd/kfd/topology/nodes"),
    dev_root: Path = Path("/dev"),
    stat_fn: StatFn = os.lstat,
) -> dict[str, Any]:
    ordered = _ordered_gpu_ids(ordered_gpu_ids)
    inventory = parse_rocm_smi_inventory(rocm_smi_inventory)
    topology, topology_generation = _read_kfd_topology_snapshot(topology_root)
    missing = [host_id for host_id in ordered if host_id not in inventory]
    if missing:
        raise GpuBoundaryError(f"rocm-smi inventory lacks requested cards: {missing}")

    kfd_identity = _character_device(dev_root / "kfd", stat_fn)
    selected_unique_ids: set[str] = set()
    claimed_render_minors: dict[int, str] = {}
    devices: list[dict[str, Any]] = []
    for host_id in ordered:
        physical = inventory[host_id]
        unique_id = physical["unique_id"]
        if unique_id in selected_unique_ids:
            raise GpuBoundaryError(
                f"multiple requested cards share physical unique ID {unique_id}"
            )
        selected_unique_ids.add(unique_id)
        matching_nodes = [node for node in topology if node["unique_id"] == unique_id]
        if not matching_nodes:
            raise GpuBoundaryError(
                f"card{host_id} unique ID {unique_id} has no matching KFD topology node"
            )

        by_minor: dict[int, list[dict[str, Any]]] = {}
        for node in matching_nodes:
            by_minor.setdefault(node["drm_render_minor"], []).append(
                {
                    "node_id": node["node_id"],
                    "gpu_id": node["gpu_id"],
                    "gpu_id_sha256": node["gpu_id_sha256"],
                    "properties_sha256": node["properties_sha256"],
                }
            )
        render_nodes: list[dict[str, Any]] = []
        for render_minor in sorted(by_minor):
            prior_owner = claimed_render_minors.get(render_minor)
            if prior_owner is not None:
                raise GpuBoundaryError(
                    f"renderD{render_minor} is claimed by card{prior_owner} and card{host_id}"
                )
            claimed_render_minors[render_minor] = host_id
            actual_path = dev_root / "dri" / f"renderD{render_minor}"
            identity = _character_device(actual_path, stat_fn)
            if identity["minor"] != render_minor:
                raise GpuBoundaryError(
                    f"{actual_path} minor {identity['minor']} does not match KFD {render_minor}"
                )
            render_nodes.append(
                {
                    "path": f"/dev/dri/renderD{render_minor}",
                    "major": identity["major"],
                    "minor": identity["minor"],
                    "kfd_nodes": sorted(by_minor[render_minor], key=lambda item: item["node_id"]),
                }
            )
        devices.append(
            {
                "host_gpu_id": host_id,
                **physical,
                "render_nodes": render_nodes,
            }
        )

    body: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "ordered_host_gpu_ids": ordered,
        "kfd_device": {"path": "/dev/kfd", **kfd_identity},
        "devices": devices,
        "rocm_smi_inventory_sha256": canonical_digest(
            {host_id: inventory[host_id] for host_id in ordered}
        ),
        "kfd_topology_sha256": canonical_digest(topology),
        "kfd_topology_generation_id": topology_generation,
    }
    body["sha256"] = canonical_digest(body)
    verify_plan(body, expected_gpu_ids=ordered)
    return body


def verify_plan(
    plan: Mapping[str, Any], *, expected_gpu_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    material = dict(plan)
    if material.get("schema") != PLAN_SCHEMA:
        raise GpuBoundaryError("GPU boundary plan schema is unsupported")
    digest = material.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise GpuBoundaryError("GPU boundary plan has no valid SHA256")
    if canonical_digest(_without_digest(material)) != digest:
        raise GpuBoundaryError("GPU boundary plan digest mismatch")
    ordered = _ordered_gpu_ids(material.get("ordered_host_gpu_ids", []))
    if expected_gpu_ids is not None and ordered != _ordered_gpu_ids(expected_gpu_ids):
        raise GpuBoundaryError("GPU boundary plan pool does not match requested order")

    kfd = material.get("kfd_device")
    if not isinstance(kfd, dict) or kfd.get("path") != "/dev/kfd":
        raise GpuBoundaryError("GPU boundary plan has an invalid KFD device")
    _validate_device_numbers(kfd, "KFD device")
    devices = material.get("devices")
    if not isinstance(devices, list) or len(devices) != len(ordered):
        raise GpuBoundaryError("GPU boundary plan devices do not match its pool")
    if [item.get("host_gpu_id") for item in devices if isinstance(item, dict)] != ordered:
        raise GpuBoundaryError("GPU boundary plan device order is not canonical")

    unique_ids: set[str] = set()
    render_paths: set[str] = set()
    topology_node_ids: set[int] = set()
    for digest_key in ("rocm_smi_inventory_sha256", "kfd_topology_sha256"):
        digest_value = material.get(digest_key)
        if not isinstance(digest_value, str) or not _SHA256.fullmatch(digest_value):
            raise GpuBoundaryError(f"GPU boundary plan has an invalid {digest_key}")
    topology_generation = material.get("kfd_topology_generation_id")
    if (
        not isinstance(topology_generation, int)
        or isinstance(topology_generation, bool)
        or topology_generation < 0
        or topology_generation > 0xFFFFFFFFFFFFFFFF
    ):
        raise GpuBoundaryError("GPU boundary plan has an invalid topology generation")
    for device in devices:
        if not isinstance(device, dict):
            raise GpuBoundaryError("GPU boundary device is not an object")
        unique_id = _normalize_unique_id(device.get("unique_id"), "plan unique_id")
        if unique_id != device.get("unique_id") or unique_id in unique_ids:
            raise GpuBoundaryError("GPU boundary plan has a duplicate/noncanonical unique ID")
        unique_ids.add(unique_id)
        for identity_key in ("serial_number", "card_series", "gfx_version"):
            if not isinstance(device.get(identity_key), str) or not device[identity_key]:
                raise GpuBoundaryError(
                    f"GPU boundary device has an invalid {identity_key}"
                )
        render_nodes = device.get("render_nodes")
        if not isinstance(render_nodes, list) or not render_nodes:
            raise GpuBoundaryError("GPU boundary device has no render nodes")
        minors: list[int] = []
        for render in render_nodes:
            if not isinstance(render, dict):
                raise GpuBoundaryError("render node is not an object")
            _validate_device_numbers(render, "render node")
            minor = render["minor"]
            path = render.get("path")
            if path != f"/dev/dri/renderD{minor}" or path in render_paths:
                raise GpuBoundaryError("render node path is invalid or duplicated")
            render_paths.add(path)
            minors.append(minor)
            kfd_nodes = render.get("kfd_nodes")
            if not isinstance(kfd_nodes, list) or not kfd_nodes:
                raise GpuBoundaryError("render node lacks KFD topology evidence")
            node_ids = [node.get("node_id") for node in kfd_nodes if isinstance(node, dict)]
            if (
                len(node_ids) != len(kfd_nodes)
                or node_ids != sorted(node_ids)
                or len(node_ids) != len(set(node_ids))
            ):
                raise GpuBoundaryError("KFD topology evidence is malformed or unsorted")
            for node in kfd_nodes:
                node_id = node["node_id"]
                gpu_id = node.get("gpu_id")
                gpu_id_sha256 = node.get("gpu_id_sha256")
                properties_sha256 = node.get("properties_sha256")
                if (
                    not isinstance(node_id, int)
                    or isinstance(node_id, bool)
                    or node_id < 0
                    or node_id in topology_node_ids
                    or not isinstance(gpu_id, int)
                    or isinstance(gpu_id, bool)
                    or gpu_id <= 0
                    or not isinstance(gpu_id_sha256, str)
                    or not _SHA256.fullmatch(gpu_id_sha256)
                    or not isinstance(properties_sha256, str)
                    or not _SHA256.fullmatch(properties_sha256)
                ):
                    raise GpuBoundaryError("KFD topology evidence is invalid or duplicated")
                topology_node_ids.add(node_id)
        if minors != sorted(minors) or len(minors) != len(set(minors)):
            raise GpuBoundaryError("render nodes are not unique and numerically sorted")
    return material


def _validate_device_numbers(value: Mapping[str, Any], label: str) -> None:
    for key in ("major", "minor"):
        number = value.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise GpuBoundaryError(f"{label} has an invalid {key}")


def load_plan(path: Path, *, expected_gpu_ids: Sequence[str] | None = None) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GpuBoundaryError(f"cannot load GPU boundary plan {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise GpuBoundaryError("GPU boundary plan must be a JSON object")
    return verify_plan(decoded, expected_gpu_ids=expected_gpu_ids)


def selected_device(plan: Mapping[str, Any], host_gpu_id: str) -> dict[str, Any]:
    verified = verify_plan(plan)
    matches = [
        item for item in verified["devices"] if item["host_gpu_id"] == str(host_gpu_id)
    ]
    if len(matches) != 1:
        raise GpuBoundaryError(f"GPU boundary plan does not bind card{host_gpu_id}")
    return matches[0]


def docker_device_arguments(
    plan: Mapping[str, Any],
    host_gpu_id: str,
    *,
    dev_root: Path = Path("/dev"),
    stat_fn: StatFn = os.lstat,
) -> list[str]:
    verified = verify_plan(plan)
    device = selected_device(verified, host_gpu_id)
    mappings: list[str] = []
    for planned in [verified["kfd_device"], *device["render_nodes"]]:
        relative = Path(planned["path"]).relative_to("/dev")
        identity = _character_device(dev_root / relative, stat_fn)
        if identity != {"major": planned["major"], "minor": planned["minor"]}:
            raise GpuBoundaryError(f"device identity changed after plan creation: {planned['path']}")
        mappings.append(f"--device={planned['path']}:{planned['path']}:rw")
    return mappings


def verify_visible_devices(
    plan: Mapping[str, Any],
    host_gpu_id: str,
    *,
    dev_root: Path = Path("/dev"),
    stat_fn: StatFn = os.lstat,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verified = verify_plan(plan)
    device = selected_device(verified, host_gpu_id)
    expected_paths = [render["path"] for render in device["render_nodes"]]
    dri_root = dev_root / "dri"
    try:
        names = sorted(path.name for path in dri_root.iterdir())
    except OSError as error:
        raise GpuBoundaryError(f"cannot enumerate visible DRI devices: {error}") from error
    observed_render = [name for name in names if name.startswith("renderD")]
    if any(_RENDER.fullmatch(name) is None for name in observed_render):
        raise GpuBoundaryError("visible DRI directory has a malformed render node")
    observed_render.sort(key=lambda name: int(_RENDER.fullmatch(name).group(1)))
    expected_names = [Path(path).name for path in expected_paths]
    if observed_render != expected_names:
        raise GpuBoundaryError(
            f"visible render nodes differ: expected={expected_names}, observed={observed_render}"
        )
    if any(_CARD.fullmatch(name) for name in names):
        raise GpuBoundaryError("formal container exposes a DRI card node")
    if os.path.lexists(dev_root / "mem"):
        raise GpuBoundaryError("formal container exposes /dev/mem")

    expected_devices = [verified["kfd_device"], *device["render_nodes"]]
    observed: list[dict[str, Any]] = []
    for planned in expected_devices:
        relative = Path(planned["path"]).relative_to("/dev")
        identity = _character_device(dev_root / relative, stat_fn)
        if identity != {"major": planned["major"], "minor": planned["minor"]}:
            raise GpuBoundaryError(f"visible device identity mismatch: {planned['path']}")
        observed.append({"path": planned["path"], **identity})

    environment = dict(os.environ if environ is None else environ)
    required_environment = {
        "AGENT_KERNEL_ARENA_HOST_GPU_ID": str(host_gpu_id),
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": verified["sha256"],
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }
    for key, expected in required_environment.items():
        if environment.get(key) != expected:
            raise GpuBoundaryError(
                f"formal GPU environment mismatch for {key}: {environment.get(key)!r}"
            )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "plan_sha256": verified["sha256"],
        "host_gpu_id": str(host_gpu_id),
        "unique_id": device["unique_id"],
        "expected_render_nodes": expected_paths,
        "observed_devices": observed,
        "environment": required_environment,
        "card_nodes": [],
        "dev_mem_present": False,
        "verified": True,
    }
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


def verify_runtime_identity(
    plan: Mapping[str, Any],
    host_gpu_id: str,
    *,
    structural_receipt: Mapping[str, Any],
    rocm_smi_inventory: str | bytes | Mapping[str, Any],
    torch_observation: Mapping[str, Any],
    topology_root: Path = Path("/sys/class/kfd/kfd/topology/nodes"),
) -> dict[str, Any]:
    """Bind the structural boundary to live ROCm, Torch, and KFD identity."""
    verified = verify_plan(plan)
    selected = selected_device(verified, host_gpu_id)
    structural = dict(structural_receipt)
    structural_digest = structural.pop("sha256", None)
    if (
        structural.get("schema") != RECEIPT_SCHEMA
        or structural_digest != canonical_digest(structural)
        or structural.get("plan_sha256") != verified["sha256"]
        or structural.get("host_gpu_id") != str(host_gpu_id)
        or structural.get("unique_id") != selected["unique_id"]
        or structural.get("verified") is not True
    ):
        raise GpuBoundaryError("structural GPU boundary receipt is invalid")

    inventory = parse_rocm_smi_inventory(rocm_smi_inventory)
    if len(inventory) != 1:
        raise GpuBoundaryError("formal worker runtime must enumerate exactly one physical GPU")
    observed_card, observed_identity = next(iter(inventory.items()))
    if observed_identity["unique_id"] != selected["unique_id"]:
        raise GpuBoundaryError("runtime physical GPU unique ID differs from the plan")
    count = torch_observation.get("device_count")
    arch = str(torch_observation.get("gcn_arch_name", "")).split(":", 1)[0]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != 1
        or arch != selected["gfx_version"]
        or not str(torch_observation.get("device_name", "")).strip()
    ):
        raise GpuBoundaryError("Torch runtime does not expose exactly the selected GPU")

    expected_minors = sorted(
        int(Path(render["path"]).name.removeprefix("renderD"))
        for render in selected["render_nodes"]
    )
    matching_nodes, topology_generation = _read_selected_runtime_topology(
        topology_root, selected
    )
    observed_minors = sorted({node["drm_render_minor"] for node in matching_nodes})
    if (
        not matching_nodes
        or topology_generation != verified["kfd_topology_generation_id"]
        or any(node["unique_id"] != selected["unique_id"] for node in matching_nodes)
        or observed_minors != expected_minors
    ):
        raise GpuBoundaryError("live KFD topology does not match selected render nodes")

    runtime = {
        "visible_physical_gpu_count": 1,
        "rocm_smi_card": observed_card,
        "rocm_smi_identity": observed_identity,
        "torch": {
            "device_count": count,
            "device_name": str(torch_observation["device_name"]),
            "gcn_arch_name": str(torch_observation["gcn_arch_name"]),
        },
        "kfd_nodes": matching_nodes,
        "kfd_topology_generation_id": topology_generation,
        "observed_render_minors": observed_minors,
    }
    receipt = {
        **structural,
        "runtime_identity": runtime,
        "runtime_verified": True,
    }
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


def _write_json(value: Mapping[str, Any], output: str) -> None:
    rendered = canonical_json_bytes(value).decode("utf-8") + "\n"
    if output == "-":
        sys.stdout.write(rendered)
        return
    path = Path(output)
    path.write_text(rendered, encoding="utf-8")


def _load_inventory_argument(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    try:
        return Path(value).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise GpuBoundaryError(f"cannot read rocm-smi JSON: {value}: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--gpu-ids", required=True)
    resolve.add_argument("--rocm-smi-json", required=True)
    resolve.add_argument(
        "--topology-root", default="/sys/class/kfd/kfd/topology/nodes"
    )
    resolve.add_argument("--dev-root", default="/dev")
    resolve.add_argument("--output", default="-")

    docker_args = subparsers.add_parser("docker-args")
    docker_args.add_argument("--plan", required=True)
    docker_args.add_argument("--host-gpu-id", required=True)
    docker_args.add_argument("--dev-root", default="/dev")

    digest = subparsers.add_parser("plan-digest")
    digest.add_argument("--plan", required=True)

    visible = subparsers.add_parser("verify-visible")
    visible.add_argument("--plan", required=True)
    visible.add_argument("--host-gpu-id", required=True)
    visible.add_argument("--dev-root", default="/dev")
    visible.add_argument("--output", default="-")
    runtime = subparsers.add_parser("verify-runtime")
    runtime.add_argument("--plan", required=True)
    runtime.add_argument("--host-gpu-id", required=True)
    runtime.add_argument("--dev-root", default="/dev")
    runtime.add_argument("--topology-root", default="/sys/class/kfd/kfd/topology/nodes")
    runtime.add_argument("--rocm-smi-json", required=True)
    runtime.add_argument("--torch-json", required=True)
    runtime.add_argument("--output", default="-")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "resolve":
            plan = build_plan(
                args.gpu_ids.split(","),
                _load_inventory_argument(args.rocm_smi_json),
                topology_root=Path(args.topology_root),
                dev_root=Path(args.dev_root),
            )
            _write_json(plan, args.output)
        elif args.command == "docker-args":
            plan = load_plan(Path(args.plan))
            for item in docker_device_arguments(
                plan, args.host_gpu_id, dev_root=Path(args.dev_root)
            ):
                print(item)
        elif args.command == "plan-digest":
            print(load_plan(Path(args.plan))["sha256"])
        elif args.command == "verify-visible":
            receipt = verify_visible_devices(
                load_plan(Path(args.plan)),
                args.host_gpu_id,
                dev_root=Path(args.dev_root),
            )
            _write_json(receipt, args.output)
        elif args.command == "verify-runtime":
            plan = load_plan(Path(args.plan))
            structural = verify_visible_devices(
                plan,
                args.host_gpu_id,
                dev_root=Path(args.dev_root),
            )
            try:
                torch_observation = json.loads(
                    Path(args.torch_json).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise GpuBoundaryError(f"cannot load Torch GPU observation: {error}") from error
            if not isinstance(torch_observation, dict):
                raise GpuBoundaryError("Torch GPU observation must be a JSON object")
            receipt = verify_runtime_identity(
                plan,
                args.host_gpu_id,
                structural_receipt=structural,
                rocm_smi_inventory=_load_inventory_argument(args.rocm_smi_json),
                torch_observation=torch_observation,
                topology_root=Path(args.topology_root),
            )
            _write_json(receipt, args.output)
        else:  # pragma: no cover - argparse enforces the command set.
            raise GpuBoundaryError(f"unsupported command: {args.command}")
    except GpuBoundaryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GpuBoundaryError",
    "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
    "build_plan",
    "canonical_digest",
    "docker_device_arguments",
    "load_plan",
    "parse_rocm_smi_inventory",
    "read_kfd_topology",
    "selected_device",
    "verify_plan",
    "verify_runtime_identity",
    "verify_visible_devices",
]
