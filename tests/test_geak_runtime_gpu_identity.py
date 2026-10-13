"""Runtime hardware identity is measured, not inferred from task names."""
import sys
from types import SimpleNamespace

import pytest

from agents.geak.compatibility import runtime_gpu_identity


def install_torch(monkeypatch, *, available=True, arch="gfx950:sramecc+:xnack-", units=256,
                  name="AMD Instinct MI355X"):
    properties = SimpleNamespace(gcnArchName=arch, name=name, multi_processor_count=units)
    cuda = SimpleNamespace(is_available=lambda: available,
                           get_device_properties=lambda index: properties)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))


def test_runtime_device_properties_supply_the_lane_identity(monkeypatch):
    install_torch(monkeypatch)
    assert runtime_gpu_identity() == dict(device_gfx="gfx950", device_target="unknown",
                                         device_name="AMD Instinct MI355X", physical_cu_count=256)


@pytest.mark.parametrize("overrides", [{"available": False}, {"arch": ""},
                                       {"arch": "unverified"}, {"units": 0}])
def test_missing_runtime_hardware_fails_preparation(monkeypatch, overrides):
    install_torch(monkeypatch, **overrides)
    with pytest.raises(ValueError):
        runtime_gpu_identity()


@pytest.mark.parametrize("name,target", [("AMD Radeon AI PRO R9700", "r9700"),
                                         ("Generic gfx1201", "unknown")])
def test_gfx1201_alone_does_not_claim_r9700_qualification(monkeypatch, name, target):
    install_torch(monkeypatch, arch="gfx1201", units=64, name=name)
    assert runtime_gpu_identity()["device_target"] == target
