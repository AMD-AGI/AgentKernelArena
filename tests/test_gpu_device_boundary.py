from __future__ import annotations

import copy
import errno
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import gpu_device_boundary as boundary


def _write_node(root: Path, node_id: int, **properties: object) -> None:
    node = root / str(node_id)
    node.mkdir(parents=True)
    gpu_id = properties.pop("gpu_id")
    (node / "gpu_id").write_text(f"{gpu_id}\n", encoding="ascii")
    rendered = "".join(f"{key} {value}\n" for key, value in properties.items())
    (node / "properties").write_text(rendered, encoding="utf-8")


def _inventory() -> dict:
    return {
        "card0": {
            "Unique ID": "0x0000000000000abc",
            "Serial Number": "SERIAL-0",
            "Card Series": "AMD Instinct MI355X",
            "GFX Version": "gfx950",
        },
        "card7": {
            "Unique ID": "0x0000000000000def",
            "Serial Number": "SERIAL-7",
            "Card Series": "AMD Instinct MI355X",
            "GFX Version": "gfx950",
        },
    }


def _fake_devices(dev_root: Path, minors: list[int]):
    (dev_root / "dri").mkdir(parents=True)
    (dev_root / "kfd").touch()
    identities = {dev_root / "kfd": (235, 0)}
    for minor in minors:
        path = dev_root / "dri" / f"renderD{minor}"
        path.touch()
        identities[path] = (226, minor)

    def fake_lstat(path: Path):
        major, minor = identities[Path(path)]
        return SimpleNamespace(
            st_mode=0o660 | 0o020000,
            st_rdev=os.makedev(major, minor),
        )

    return fake_lstat, identities


def _build_xcp_plan(tmp_path: Path, gpu_ids: tuple[str, ...] = ("0",)):
    topology = tmp_path / "topology"
    topology.mkdir()
    (tmp_path / "generation_id").write_text("1\n", encoding="ascii")
    _write_node(
        topology,
        0,
        gpu_id=0,
        simd_count=0,
        drm_render_minor=0,
        vendor_id=0,
        device_id=0,
        gfx_target_version=0,
    )
    _write_node(
        topology,
        1,
        gpu_id=101,
        unique_id=int("abc", 16),
        drm_render_minor=128,
        simd_count=304,
    )
    _write_node(
        topology,
        2,
        gpu_id=102,
        unique_id=int("abc", 16),
        drm_render_minor=137,
        simd_count=304,
    )
    # Multiple KFD agents may describe one render partition. The plan retains
    # both topology receipts but maps the character device only once.
    _write_node(
        topology,
        3,
        gpu_id=103,
        unique_id=int("abc", 16),
        drm_render_minor=137,
        simd_count=304,
    )
    _write_node(
        topology,
        9,
        gpu_id=701,
        unique_id=int("def", 16),
        drm_render_minor=149,
        simd_count=304,
    )
    dev_root = tmp_path / "dev"
    visible_minors = [128, 137, *([149] if "7" in gpu_ids else [])]
    fake_lstat, identities = _fake_devices(dev_root, visible_minors)
    plan = boundary.build_plan(
        gpu_ids,
        _inventory(),
        topology_root=topology,
        dev_root=dev_root,
        stat_fn=fake_lstat,
    )
    return plan, topology, dev_root, fake_lstat, identities


def _deny_node_reads(monkeypatch, *node_ids: int) -> None:
    denied = {str(node_id) for node_id in node_ids}
    original = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.parent.name in denied and path.name in {"gpu_id", "properties"}:
            raise PermissionError(errno.EPERM, "device cgroup denied KFD node", path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)


def test_resolver_joins_rocm_unique_id_to_all_noncontiguous_xcp_nodes(tmp_path) -> None:
    plan, _, _, _, _ = _build_xcp_plan(tmp_path, ("0", "7"))

    assert plan["ordered_host_gpu_ids"] == ["0", "7"]
    assert [item["path"] for item in plan["devices"][0]["render_nodes"]] == [
        "/dev/dri/renderD128",
        "/dev/dri/renderD137",
    ]
    assert [
        item["node_id"] for item in plan["devices"][0]["render_nodes"][1]["kfd_nodes"]
    ] == [2, 3]
    assert plan["devices"][1]["render_nodes"][0]["path"] == "/dev/dri/renderD149"
    assert plan["sha256"] == boundary.canonical_digest(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    assert boundary.verify_plan(plan, expected_gpu_ids=("0", "7")) == plan


def test_resolver_is_deterministic_for_identical_evidence(tmp_path) -> None:
    first, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    second = boundary.build_plan(
        ("0",),
        _inventory(),
        topology_root=topology,
        dev_root=dev_root,
        stat_fn=fake_lstat,
    )

    assert first == second
    assert first["kfd_topology_generation_id"] == 1


def test_resolver_rejects_a_generation_change_during_scan(tmp_path, monkeypatch) -> None:
    _, topology, _, _, _ = _build_xcp_plan(tmp_path)
    generation_path = topology.parent / "generation_id"
    original = boundary._read_kfd_node
    changed = False

    def change_generation(entry):
        nonlocal changed
        node = original(entry)
        if not changed:
            generation_path.write_text("2\n", encoding="ascii")
            changed = True
        return node

    monkeypatch.setattr(boundary, "_read_kfd_node", change_generation)
    with pytest.raises(boundary.GpuBoundaryError, match="changed during"):
        boundary.read_kfd_topology(topology)


def test_resolver_reads_gpu_id_from_the_kfd_node_file(tmp_path) -> None:
    plan, topology, _, _, _ = _build_xcp_plan(tmp_path)

    first_node = plan["devices"][0]["render_nodes"][0]["kfd_nodes"][0]
    assert first_node["gpu_id"] == 101
    assert first_node["gpu_id_sha256"] == hashlib.sha256(
        (topology / "1" / "gpu_id").read_bytes()
    ).hexdigest()
    assert first_node["properties_sha256"] == hashlib.sha256(
        (topology / "1" / "properties").read_bytes()
    ).hexdigest()
    assert "gpu_id " not in (topology / "1" / "properties").read_text(
        encoding="utf-8"
    )


def test_resolver_fails_closed_on_missing_or_contradictory_gpu_id_file(tmp_path) -> None:
    _, topology, _, _, _ = _build_xcp_plan(tmp_path)
    (topology / "1" / "gpu_id").unlink()
    with pytest.raises(boundary.GpuBoundaryError, match="cannot read KFD gpu_id"):
        boundary.read_kfd_topology(topology)

    (topology / "1" / "gpu_id").write_text("0\n", encoding="ascii")
    with pytest.raises(boundary.GpuBoundaryError, match="non-GPU node"):
        boundary.read_kfd_topology(topology)


@pytest.mark.parametrize("raw_gpu_id", ["+101\n", "00101\n", "1_01\n", "4294967296\n"])
def test_resolver_rejects_noncanonical_or_out_of_range_gpu_ids(
    tmp_path, raw_gpu_id
) -> None:
    _, topology, _, _, _ = _build_xcp_plan(tmp_path)
    (topology / "1" / "gpu_id").write_text(raw_gpu_id, encoding="ascii")

    with pytest.raises(boundary.GpuBoundaryError):
        boundary.read_kfd_topology(topology)


def test_resolver_rejects_duplicate_gpu_ids(tmp_path) -> None:
    _, topology, _, _, _ = _build_xcp_plan(tmp_path)
    (topology / "2" / "gpu_id").write_text("101\n", encoding="ascii")

    with pytest.raises(boundary.GpuBoundaryError, match="duplicate"):
        boundary.read_kfd_topology(topology)


@pytest.mark.parametrize(
    ("property_name", "property_value"),
    [
        ("unique_id", "2748"),
        ("drm_render_minor", "128"),
        ("vendor_id", "4098"),
        ("device_id", "30115"),
        ("gfx_target_version", "90500"),
    ],
)
def test_resolver_rejects_zero_gpu_id_with_gpu_identity(
    tmp_path, property_name, property_value
) -> None:
    _, topology, _, _, _ = _build_xcp_plan(tmp_path)
    cpu_properties = topology / "0" / "properties"
    lines = cpu_properties.read_text(encoding="utf-8").splitlines()
    replacement = f"{property_name} {property_value}"
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(f"{property_name} "):
            lines[index] = replacement
            replaced = True
    if not replaced:
        lines.append(replacement)
    cpu_properties.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(boundary.GpuBoundaryError, match="non-GPU node"):
        boundary.read_kfd_topology(topology)


@pytest.mark.parametrize("mutation", ["zero_simd", "missing_unique_id", "missing_render"])
def test_resolver_rejects_incomplete_nonzero_gpu_nodes(tmp_path, mutation) -> None:
    _, topology, _, _, _ = _build_xcp_plan(tmp_path)
    properties_path = topology / "1" / "properties"
    lines = properties_path.read_text(encoding="utf-8").splitlines()
    if mutation == "zero_simd":
        lines = ["simd_count 0" if line.startswith("simd_count ") else line for line in lines]
    elif mutation == "missing_unique_id":
        lines = [line for line in lines if not line.startswith("unique_id ")]
    else:
        lines = [line for line in lines if not line.startswith("drm_render_minor ")]
    properties_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(boundary.GpuBoundaryError):
        boundary.read_kfd_topology(topology)


def test_resolver_fails_when_rocm_identity_has_no_kfd_match(tmp_path) -> None:
    plan, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    assert plan
    inventory = _inventory()
    inventory["card0"]["Unique ID"] = "0x1234"

    with pytest.raises(boundary.GpuBoundaryError, match="no matching KFD"):
        boundary.build_plan(
            ("0",),
            inventory,
            topology_root=topology,
            dev_root=dev_root,
            stat_fn=fake_lstat,
        )


def test_resolver_rejects_duplicate_physical_identity(tmp_path) -> None:
    _, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    inventory = _inventory()
    inventory["card7"]["Unique ID"] = inventory["card0"]["Unique ID"]

    with pytest.raises(boundary.GpuBoundaryError, match="share physical unique ID"):
        boundary.build_plan(
            ("0", "7"),
            inventory,
            topology_root=topology,
            dev_root=dev_root,
            stat_fn=fake_lstat,
        )


def test_plan_digest_tampering_is_rejected(tmp_path) -> None:
    plan, _, _, _, _ = _build_xcp_plan(tmp_path)
    tampered = copy.deepcopy(plan)
    tampered["devices"][0]["serial_number"] = "CHANGED"

    with pytest.raises(boundary.GpuBoundaryError, match="digest mismatch"):
        boundary.verify_plan(tampered)


def test_docker_arguments_recheck_device_identity(tmp_path) -> None:
    plan, _, dev_root, fake_lstat, identities = _build_xcp_plan(tmp_path)

    assert boundary.docker_device_arguments(
        plan, "0", dev_root=dev_root, stat_fn=fake_lstat
    ) == [
        "--device=/dev/kfd:/dev/kfd:rw",
        "--device=/dev/dri/renderD128:/dev/dri/renderD128:rw",
        "--device=/dev/dri/renderD137:/dev/dri/renderD137:rw",
    ]
    identities[dev_root / "dri" / "renderD137"] = (226, 138)
    with pytest.raises(boundary.GpuBoundaryError, match="identity changed"):
        boundary.docker_device_arguments(
            plan, "0", dev_root=dev_root, stat_fn=fake_lstat
        )


def test_container_verifier_accepts_exact_boundary_and_emits_receipt(tmp_path) -> None:
    plan, _, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    environment = {
        "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }

    receipt = boundary.verify_visible_devices(
        plan, "0", dev_root=dev_root, stat_fn=fake_lstat, environ=environment
    )

    assert receipt["verified"] is True
    assert receipt["plan_sha256"] == plan["sha256"]
    assert receipt["expected_render_nodes"] == [
        "/dev/dri/renderD128",
        "/dev/dri/renderD137",
    ]
    assert receipt["sha256"] == boundary.canonical_digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )


def test_runtime_verifier_proves_one_matching_rocm_torch_and_kfd_identity(
    tmp_path, monkeypatch
) -> None:
    plan, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    environment = {
        "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }
    structural = boundary.verify_visible_devices(
        plan, "0", dev_root=dev_root, stat_fn=fake_lstat, environ=environment
    )
    _deny_node_reads(monkeypatch, 9)

    receipt = boundary.verify_runtime_identity(
        plan,
        "0",
        structural_receipt=structural,
        rocm_smi_inventory={"card0": _inventory()["card0"]},
        torch_observation={
            "device_count": 1,
            "device_name": "AMD Instinct MI355X",
            "gcn_arch_name": "gfx950:sramecc+:xnack-",
        },
        topology_root=topology,
    )

    assert receipt["runtime_verified"] is True
    assert receipt["runtime_identity"]["visible_physical_gpu_count"] == 1
    assert receipt["runtime_identity"]["observed_render_minors"] == [128, 137]
    assert receipt["runtime_identity"]["rocm_smi_identity"]["unique_id"] == (
        "0x0000000000000abc"
    )


def test_runtime_verifier_rejects_a_readable_unselected_gpu(tmp_path) -> None:
    plan, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    structural = boundary.verify_visible_devices(
        plan,
        "0",
        dev_root=dev_root,
        stat_fn=fake_lstat,
        environ={
            "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
            "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
        },
    )

    with pytest.raises(boundary.GpuBoundaryError, match="unexpected GPU KFD node"):
        boundary.verify_runtime_identity(
            plan,
            "0",
            structural_receipt=structural,
            rocm_smi_inventory={"card0": _inventory()["card0"]},
            torch_observation={
                "device_count": 1,
                "device_name": "AMD Instinct MI355X",
                "gcn_arch_name": "gfx950",
            },
            topology_root=topology,
        )


def test_runtime_verifier_rejects_permission_denial_for_selected_node(
    tmp_path, monkeypatch
) -> None:
    plan, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    structural = boundary.verify_visible_devices(
        plan,
        "0",
        dev_root=dev_root,
        stat_fn=fake_lstat,
        environ={
            "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
            "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
        },
    )
    _deny_node_reads(monkeypatch, 1, 9)

    with pytest.raises(boundary.GpuBoundaryError, match="cannot read KFD"):
        boundary.verify_runtime_identity(
            plan,
            "0",
            structural_receipt=structural,
            rocm_smi_inventory={"card0": _inventory()["card0"]},
            torch_observation={
                "device_count": 1,
                "device_name": "AMD Instinct MI355X",
                "gcn_arch_name": "gfx950",
            },
            topology_root=topology,
        )


@pytest.mark.parametrize("mutation", ["gpu_id", "properties", "missing_node"])
def test_runtime_verifier_rejects_selected_topology_drift(
    tmp_path, monkeypatch, mutation
) -> None:
    plan, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    structural = boundary.verify_visible_devices(
        plan,
        "0",
        dev_root=dev_root,
        stat_fn=fake_lstat,
        environ={
            "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
            "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
        },
    )
    if mutation == "gpu_id":
        (topology / "1" / "gpu_id").write_text("999\n", encoding="ascii")
    elif mutation == "properties":
        properties = topology / "1" / "properties"
        properties.write_text(
            properties.read_text(encoding="utf-8") + "num_cp_queues 1\n",
            encoding="utf-8",
        )
    else:
        for path in (topology / "1").iterdir():
            path.unlink()
        (topology / "1").rmdir()
    _deny_node_reads(monkeypatch, 9)

    with pytest.raises(boundary.GpuBoundaryError):
        boundary.verify_runtime_identity(
            plan,
            "0",
            structural_receipt=structural,
            rocm_smi_inventory={"card0": _inventory()["card0"]},
            torch_observation={
                "device_count": 1,
                "device_name": "AMD Instinct MI355X",
                "gcn_arch_name": "gfx950",
            },
            topology_root=topology,
        )


def test_runtime_verifier_rejects_a_different_topology_generation(
    tmp_path, monkeypatch
) -> None:
    plan, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    structural = boundary.verify_visible_devices(
        plan,
        "0",
        dev_root=dev_root,
        stat_fn=fake_lstat,
        environ={
            "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
            "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
        },
    )
    (topology.parent / "generation_id").write_text("2\n", encoding="ascii")
    _deny_node_reads(monkeypatch, 9)

    with pytest.raises(boundary.GpuBoundaryError, match="topology"):
        boundary.verify_runtime_identity(
            plan,
            "0",
            structural_receipt=structural,
            rocm_smi_inventory={"card0": _inventory()["card0"]},
            torch_observation={
                "device_count": 1,
                "device_name": "AMD Instinct MI355X",
                "gcn_arch_name": "gfx950",
            },
            topology_root=topology,
        )


def test_runtime_verifier_rejects_more_than_one_torch_device(tmp_path) -> None:
    plan, topology, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    environment = {
        "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }
    structural = boundary.verify_visible_devices(
        plan, "0", dev_root=dev_root, stat_fn=fake_lstat, environ=environment
    )

    with pytest.raises(boundary.GpuBoundaryError, match="Torch runtime"):
        boundary.verify_runtime_identity(
            plan,
            "0",
            structural_receipt=structural,
            rocm_smi_inventory={"card0": _inventory()["card0"]},
            torch_observation={
                "device_count": 2,
                "device_name": "AMD Instinct MI355X",
                "gcn_arch_name": "gfx950",
            },
            topology_root=topology,
        )


@pytest.mark.parametrize("extra_name", ["renderD149", "card0"])
def test_container_verifier_rejects_extra_dri_access(tmp_path, extra_name) -> None:
    plan, _, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    (dev_root / "dri" / extra_name).touch()
    environment = {
        "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }

    with pytest.raises(boundary.GpuBoundaryError):
        boundary.verify_visible_devices(
            plan, "0", dev_root=dev_root, stat_fn=fake_lstat, environ=environment
        )


def test_container_verifier_rejects_dev_mem_and_nonzero_mask(tmp_path) -> None:
    plan, _, dev_root, fake_lstat, _ = _build_xcp_plan(tmp_path)
    (dev_root / "mem").touch()
    environment = {
        "AGENT_KERNEL_ARENA_HOST_GPU_ID": "0",
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256": plan["sha256"],
        "ROCR_VISIBLE_DEVICES": "7",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }

    with pytest.raises(boundary.GpuBoundaryError, match="/dev/mem"):
        boundary.verify_visible_devices(
            plan, "0", dev_root=dev_root, stat_fn=fake_lstat, environ=environment
        )
    (dev_root / "mem").unlink()
    with pytest.raises(boundary.GpuBoundaryError, match="ROCR_VISIBLE_DEVICES"):
        boundary.verify_visible_devices(
            plan, "0", dev_root=dev_root, stat_fn=fake_lstat, environ=environment
        )
