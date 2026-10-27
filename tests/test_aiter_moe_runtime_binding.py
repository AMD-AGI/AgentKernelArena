"""CPU checks of pre-import dispatch binding; no native qualification."""
import importlib.util
import os
from pathlib import Path

import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/aiter_workloads/deepseek_v4_pro_tp8_ep1_c64_8k1k_fused_moe"


@pytest.fixture
def runtime_binding(monkeypatch):
    spec = importlib.util.spec_from_file_location("_moe_dispatch_binding", TASK / "scripts/runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    for name in runtime.DISPATCH_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    declared = {name: os.environ.get(name) for name in runtime.DISPATCH_ENVIRONMENT}
    return runtime, {"dispatch_environment": declared}, {"AITER_BF16_FP8_MOE_BOUND": "0"}


def test_matching_environment_is_checked_without_overwriting_it(runtime_binding, monkeypatch):
    runtime, binding, required = runtime_binding
    monkeypatch.setenv("UNRELATED_TASK_SETTING", "keep")
    before = dict(os.environ)
    assert runtime.verify_dispatch_environment(binding, required) == binding["dispatch_environment"]
    assert dict(os.environ) == before


@pytest.mark.parametrize("name,value", [
    ("AITER_MOE_A8W4_BYPASS_QUANT", "1"),
    ("AITER_MOE_SORT_BACKEND", "opus"),
    ("AITER_JIT_DIR", "/different/native/cache"),
    ("AITER_BF16_FP8_MOE_BOUND", "256"),
])
def test_inherited_override_cannot_change_captured_dispatch(runtime_binding, monkeypatch, name, value):
    runtime, binding, required = runtime_binding
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="Effective runtime dispatch setting differs"):
        runtime.verify_dispatch_environment(binding, required)
    assert os.environ[name] == value


def test_unrecorded_absence_is_not_assumed(runtime_binding):
    runtime, binding, required = runtime_binding
    del binding["dispatch_environment"]["AITER_USE_NT"]
    with pytest.raises(RuntimeError, match="Every MoE dispatch setting"):
        runtime.verify_dispatch_environment(binding, required)


def test_capture_cannot_silently_change_selected_workload_setting(runtime_binding):
    runtime, binding, required = runtime_binding
    binding["dispatch_environment"]["AITER_BF16_FP8_MOE_BOUND"] = "256"
    with pytest.raises(RuntimeError, match="selected workload environment"):
        runtime.verify_dispatch_environment(binding, required)


def test_external_table_needs_content_binding(runtime_binding, monkeypatch):
    runtime, binding, required = runtime_binding
    binding["dispatch_environment"]["AITER_CONFIG_FMOE"] = "unbound.csv"
    monkeypatch.setenv("AITER_CONFIG_FMOE", "unbound.csv")
    with pytest.raises(RuntimeError, match="protected content binding"):
        runtime.verify_dispatch_environment(binding, required)


def test_explicit_absence_differs_from_empty_string(runtime_binding, monkeypatch):
    runtime, binding, required = runtime_binding
    monkeypatch.setenv("AITER_USE_NT", "")
    with pytest.raises(RuntimeError, match="Effective runtime dispatch setting differs"):
        runtime.verify_dispatch_environment(binding, required)


def test_native_binding_hashes_opaque_cache_without_deserializing(tmp_path):
    from importlib.machinery import ExtensionFileLoader
    from types import ModuleType

    spec = importlib.util.spec_from_file_location("_moe_native_identity", TASK / "scripts/native_identity.py")
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    extension = tmp_path / "baseline.so"
    extension.write_bytes(b"\x7fELFfake-test-artifact")
    module = ModuleType("baseline")
    module.__file__ = str(extension)
    module.__loader__ = ExtensionFileLoader("baseline", str(extension))
    cache = tmp_path / "launch.pkl"
    # Deliberately not a valid pickle: byte identity must not deserialize it.
    cache.write_bytes(b"opaque runtime bytes")
    required = {"baseline": "python_extension", "launch": "runtime_cache"}

    def capture():
        return identity.capture_native_identity(
            required, python_modules={"baseline": module}, runtime_cache_paths={"launch": str(cache)},
            max_artifact_bytes=1024, max_total_bytes=2048,
        )

    expected = capture()
    assert expected["declared_artifact_set_complete"] is True
    assert expected["artifacts"]["launch"]["format_check"] == "not_checked"
    identity.verify_native_identity(expected, capture())
    cache.write_bytes(b"changed runtime bytes")
    with pytest.raises(identity.NativeIdentityError, match="identity differs"):
        identity.verify_native_identity(expected, capture())
    cache.unlink()
    assert capture()["declared_artifact_set_complete"] is False
    with pytest.raises(ValueError, match="alone cannot establish"):
        identity.capture_native_identity({"launch": "runtime_cache"}, max_artifact_bytes=1024, max_total_bytes=2048)
