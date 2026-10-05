"""Select physical GPUs independently of render/KFD ordinal ordering."""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools import gpu_binding as binding


def topology(tmp_path, entries):
    root = tmp_path / "sys"
    for node, minor, unique, pci in entries:
        device = root / "devices" / pci
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text("0x1002\n")
        render = root / "class/drm" / f"renderD{minor}"
        render.mkdir(parents=True, exist_ok=True)
        if not (render / "device").exists():
            (render / "device").symlink_to(device, target_is_directory=True)
        properties = root / "class/kfd/kfd/topology/nodes" / str(node)
        properties.mkdir(parents=True)
        (properties / "properties").write_text(f"drm_render_minor {minor}\nunique_id {unique}\nsimd_count 64\n")
    return root


def device_stat(path):
    minor = int(Path(path).name.removeprefix("renderD"))
    return SimpleNamespace(st_mode=stat.S_IFCHR | 0o660, st_rdev=os.makedev(226, minor))


def test_nonzero_render_selection_uses_matching_kfd_uuid_not_ordinal_zero(tmp_path):
    root = topology(tmp_path, [(7, 128, 0xAAAA, "0000:01:00.0"), (2, 129, 0xBBBB, "0000:83:00.0")])
    selected = binding.select_gpu("/dev/dri/renderD129", sys_root=root, stat_device=device_stat)
    assert selected == {"render_device": "/dev/dri/renderD129", "drm_minor": 129,
                        "kfd_node": "2", "rocr_uuid": "GPU-000000000000bbbb", "pci_bus_id": "0000:83:00.0"}


def test_eight_render_nodes_have_independent_uuid_bindings(tmp_path):
    root = topology(tmp_path, [(20 - index, 128 + index, 1000 + index, f"0000:{index + 1:02x}:00.0")
                              for index in range(8)])
    selections = [binding.select_gpu(f"/dev/dri/renderD{128 + index}", sys_root=root, stat_device=device_stat)
                  for index in range(8)]
    assert len({selected["rocr_uuid"] for selected in selections}) == 8
    assert [selected["rocr_uuid"] for selected in selections] == [f"GPU-{1000 + index:016x}" for index in range(8)]


@pytest.mark.parametrize("entries", [[], [(1, 129, 0, "0000:83:00.0")],
                                     [(1, 129, 1, "0000:83:00.0"), (2, 129, 2, "0000:83:00.0")]])
def test_missing_uuid_or_ambiguous_kfd_mapping_fails_closed(tmp_path, entries):
    root = topology(tmp_path, entries or [(1, 128, 1, "0000:83:00.0")])
    if not entries:
        (root / "class/drm/renderD129").mkdir()
        (root / "class/drm/renderD129/device").symlink_to(root / "devices/0000:83:00.0", target_is_directory=True)
    with pytest.raises(ValueError):
        binding.select_gpu("/dev/dri/renderD129", sys_root=root, stat_device=device_stat)


@pytest.mark.parametrize("attack", ["wrong_uuid", "wrong_pci", "two_hip", "two_hsa", "wrong_env"])
def test_preflight_requires_exactly_one_expected_gpu(attack):
    expected = {"rocr_uuid": "GPU-000000000000bbbb", "pci_bus_id": "0000:83:00.0"}
    observed = {"hsa_gpu_uuids": [expected["rocr_uuid"]], "hip_device_count": 1,
                "hip_pci_bus_ids": [expected["pci_bus_id"]], "rocr_visible_devices": expected["rocr_uuid"]}
    assert binding.validate_preflight(observed, expected) is observed
    if attack == "wrong_uuid":
        observed["hsa_gpu_uuids"] = ["GPU-000000000000aaaa"]
    elif attack == "wrong_pci":
        observed["hip_pci_bus_ids"] = ["0000:01:00.0"]
    elif attack == "two_hip":
        observed["hip_device_count"] = 2
    elif attack == "two_hsa":
        observed["hsa_gpu_uuids"].append("GPU-000000000000aaaa")
    else:
        observed["rocr_visible_devices"] = "0"
    with pytest.raises(ValueError, match="exactly the requested GPU"):
        binding.validate_preflight(observed, expected)


def test_docker_command_uses_uuid_and_runs_preflight_before_task(tmp_path):
    expected = {"rocr_uuid": "GPU-000000000000bbbb", "pci_bus_id": "0000:83:00.0"}
    original = ["docker", "run", "--device", "/dev/dri/renderD129", "--env", "ROCR_VISIBLE_DEVICES=0",
                "--env", "HIP_VISIBLE_DEVICES=0", "image", "-I", "-B", "/task/scripts/task_runner.py"]
    command = binding.command_with_binding(original, "image", expected, tmp_path / "helper.py", tmp_path / "gpu.json")
    assert "ROCR_VISIBLE_DEVICES=0" not in command
    assert "ROCR_VISIBLE_DEVICES=GPU-000000000000bbbb" in command
    assert command[-10:] == ["image", "-I", "-B", "/gpu_binding.py", "--expected", "/gpu_expectation.json",
                             "--proof", "/task/build/gpu_preflight.json", "--", "/task/scripts/task_runner.py"]
    assert f"type=bind,src={tmp_path / 'gpu.json'},dst=/gpu_expectation.json,readonly" in command
    assert original[original.index("--env") + 1] == "ROCR_VISIBLE_DEVICES=0"


def test_runtime_probe_queries_hsa_uuid_and_hip_pci_without_counting_cpu_agents(monkeypatch):
    calls = []

    def hsa_init():
        calls.append("hsa_init")
        return 0

    def agent_info(agent, attribute, value):
        if attribute == 17:
            value._obj.value = 0 if agent.handle == 10 else 1
        else:
            assert attribute == 0xA011
            value.value = b"GPU-000000000000bbbb"
        return 0

    def iterate(callback, _data):
        agent = type(callback)._argtypes_[0]
        assert callback(agent(10), None) == 0
        assert callback(agent(20), None) == 0
        return 0

    def hip_init(_flags):
        calls.append("hip_init")
        return 0

    def count(value):
        value._obj.value = 1
        return 0

    def pci(value, _size, ordinal):
        assert ordinal == 0
        value.value = b"0000:83:00.0"
        return 0

    libraries = {"libhsa-runtime64.so.1": SimpleNamespace(hsa_init=hsa_init, hsa_agent_get_info=agent_info,
                                                         hsa_iterate_agents=iterate),
                 "libamdhip64.so": SimpleNamespace(hipInit=hip_init, hipGetDeviceCount=count, hipDeviceGetPCIBusId=pci)}
    monkeypatch.setattr(binding.ctypes, "CDLL", lambda name: libraries[name])
    assert binding.runtime_identity() == {"hsa_gpu_uuids": ["GPU-000000000000bbbb"],
                                          "hip_device_count": 1, "hip_pci_bus_ids": ["0000:83:00.0"]}
    assert calls == ["hsa_init", "hip_init"]
