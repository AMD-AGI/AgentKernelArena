"""Bind a ROCm render node by KFD UUID and verify the selected runtime GPU.

This selects the benchmark device; /dev/kfd is not a complete GPU security
sandbox. UUID formatting follows ROCr's HSA_AMD_AGENT_INFO_UUID implementation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import runpy
import stat
import sys
from pathlib import Path


def pci_address(value):
    match = re.fullmatch(r"(?:(?P<domain>[0-9a-fA-F]{4,8}):)?(?P<bus>[0-9a-fA-F]{2}):(?P<slot>[0-9a-fA-F]{2})\.(?P<function>[0-7])", value.strip())
    if match is None:
        raise ValueError("invalid PCI address: " + value)
    return (f"{int(match['domain'] or '0', 16):04x}:{int(match['bus'], 16):02x}:"
            f"{int(match['slot'], 16):02x}.{match['function']}")


def select_gpu(render_device, *, sys_root=Path("/sys"), stat_device=os.stat):
    """Read stable UUID/PCI identity from the requested node, not its ordinal."""
    render_device = str(render_device)
    match = re.fullmatch(r"/dev/dri/renderD([0-9]+)", render_device)
    if match is None:
        raise ValueError("an explicit /dev/dri/renderD<number> node is required")
    info = stat_device(render_device)
    minor = int(match[1])
    if not stat.S_ISCHR(info.st_mode) or os.minor(info.st_rdev) != minor:
        raise ValueError("requested render node has an unexpected device identity")
    sys_root = Path(sys_root)
    device = (sys_root / "class/drm" / Path(render_device).name / "device").resolve(strict=True)
    if int((device / "vendor").read_text().strip(), 16) != 0x1002:
        raise ValueError("requested render node is not an AMD GPU")
    pci = pci_address(device.name)
    matches = []
    for path in sorted((sys_root / "class/kfd/kfd/topology/nodes").glob("*/properties")):
        properties = {}
        for line in path.read_text().splitlines():
            key, value = line.split(maxsplit=1)
            properties[key] = int(value, 16 if value.lower().startswith("0x") else 10)
        if properties.get("drm_render_minor") == minor:
            matches.append((path.parent.name, properties))
    if len(matches) != 1:
        raise ValueError("render node must match exactly one KFD GPU; partitioned/ambiguous mapping is unsupported")
    node, properties = matches[0]
    unique_id = properties.get("unique_id", 0)
    if not 0 < unique_id < 2**64:
        raise ValueError("requested GPU has no usable KFD UUID; ordinal fallback is forbidden")
    return {"render_device": render_device, "drm_minor": minor, "kfd_node": node,
            "rocr_uuid": f"GPU-{unique_id:016x}", "pci_bus_id": pci}


def _checked(status, operation):
    if status != 0:
        raise RuntimeError(f"GPU preflight {operation} failed with runtime status {status}")


def runtime_identity():
    """Query HSA UUIDs and HIP PCI identity before importing the task runner."""
    class Agent(ctypes.Structure):
        _fields_ = [("handle", ctypes.c_uint64)]

    hsa = ctypes.CDLL("libhsa-runtime64.so.1")
    hsa.hsa_init.restype = ctypes.c_uint32
    hsa.hsa_agent_get_info.argtypes = [Agent, ctypes.c_int, ctypes.c_void_p]
    hsa.hsa_agent_get_info.restype = ctypes.c_uint32
    callback_type = ctypes.CFUNCTYPE(ctypes.c_uint32, Agent, ctypes.c_void_p)
    hsa.hsa_iterate_agents.argtypes = [callback_type, ctypes.c_void_p]
    hsa.hsa_iterate_agents.restype = ctypes.c_uint32
    _checked(hsa.hsa_init(), "hsa_init")
    uuids, failures = [], []

    @callback_type
    def visit(agent, _data):
        device_type = ctypes.c_int()
        status = hsa.hsa_agent_get_info(agent, 17, ctypes.byref(device_type))  # HSA_AGENT_INFO_DEVICE
        if status:
            failures.append(status)
            return status
        if device_type.value == 1:  # HSA_DEVICE_TYPE_GPU
            value = ctypes.create_string_buffer(21)
            status = hsa.hsa_agent_get_info(agent, 0xA011, value)  # HSA_AMD_AGENT_INFO_UUID
            if status:
                failures.append(status)
                return status
            uuids.append(value.value.decode("ascii", errors="replace"))
        return 0

    _checked(hsa.hsa_iterate_agents(visit, None), "hsa_iterate_agents")
    if failures:
        raise RuntimeError("GPU preflight HSA identity query failed")
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipInit.argtypes = [ctypes.c_uint]
    hip.hipInit.restype = ctypes.c_int
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    _checked(hip.hipInit(0), "hipInit")
    count = ctypes.c_int()
    _checked(hip.hipGetDeviceCount(ctypes.byref(count)), "hipGetDeviceCount")
    buses = []
    for ordinal in range(count.value):
        bus = ctypes.create_string_buffer(32)
        _checked(hip.hipDeviceGetPCIBusId(bus, len(bus), ordinal), "hipDeviceGetPCIBusId")
        buses.append(pci_address(bus.value.decode("ascii")))
    # Keep the runtime initialized for the task in this same process. Shutting
    # down HSA while HIP retains handles would invalidate the subsequent run.
    return {"hsa_gpu_uuids": uuids, "hip_device_count": count.value, "hip_pci_bus_ids": buses}


def validate_preflight(proof, expected):
    if (not isinstance(proof, dict) or proof.get("hsa_gpu_uuids") != [expected["rocr_uuid"]]
            or type(proof.get("hip_device_count")) is not int or proof["hip_device_count"] != 1
            or proof.get("hip_pci_bus_ids") != [expected["pci_bus_id"]]
            or proof.get("rocr_visible_devices") != expected["rocr_uuid"]):
        raise ValueError(f"GPU preflight did not select exactly the requested GPU: expected={expected}; observed={proof}")
    return proof


def command_with_binding(command, image, expected, helper_path, expected_path):
    """Override physical ROCr selection and prepend a read-only preflight."""
    command = list(command)
    for index, value in enumerate(command):
        if value.startswith("ROCR_VISIBLE_DEVICES="):
            command[index] = "ROCR_VISIBLE_DEVICES=" + expected["rocr_uuid"]
    for path in (helper_path, expected_path):
        if "," in str(path):
            raise ValueError("Docker binding paths cannot contain commas")
    image_index = command.index(image)
    command[image_index:image_index] = [
        "--mount", f"type=bind,src={helper_path},dst=/gpu_binding.py,readonly",
        "--mount", f"type=bind,src={expected_path},dst=/gpu_expectation.json,readonly",
        "--env", "CUDA_VISIBLE_DEVICES=0",
    ]
    runner = command[-1]
    command[-1] = "/gpu_binding.py"
    return command + ["--expected", "/gpu_expectation.json", "--proof", "/task/build/gpu_preflight.json", "--", runner]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-device")
    parser.add_argument("--expected")
    parser.add_argument("--proof")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.render_device:
        print(json.dumps(select_gpu(args.render_device), indent=2))
        return
    if not args.expected or not args.proof:
        parser.error("container preflight requires --expected and --proof")
    expected = json.loads(Path(args.expected).read_text())
    observed = {**runtime_identity(), "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES")}
    validate_preflight(observed, expected)
    Path(args.proof).write_text(json.dumps(observed, indent=2) + "\n")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command:
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")


if __name__ == "__main__":
    main()
