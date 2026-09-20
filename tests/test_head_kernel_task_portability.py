"""CPU checks for portable captured contracts; no GPU execution is claimed."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml


from head_kernel_test_utils import task_directory

SUITE = Path(__file__).resolve().parents[1] / "tasks" / "head_kernels"
TASKS = sorted(p.parent for p in SUITE.rglob("config.yaml"))


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def contract():
    return load_file(TASKS[0] / "ut" / "task_contract.py", "captured_contract_test")


class DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


F32 = DType("float32")
U8 = DType("uint8")


class Tensor:
    def __init__(self, shape=(2, 3), stride=(3, 1), dtype=F32, device="cpu"):
        self.shape = shape
        self._stride = stride
        self.dtype = dtype
        self.device = device
        self.copied_from = None

    def to(self, device):
        return Tensor(self.shape, self._stride, self.dtype, device)

    def stride(self):
        return self._stride

    def view(self, dtype):
        return Tensor(self.shape, self._stride, dtype, self.device)

    def reshape(self, shape):
        if __import__("math").prod(shape) != __import__("math").prod(self.shape):
            raise ValueError("incorrect number of captured elements")
        return Tensor(shape, self._stride, self.dtype, self.device)

    def copy_(self, other):
        self.copied_from = other
        return self

    def contiguous(self):
        return self


TORCH = types.SimpleNamespace(
    dtype=DType,
    float32=F32,
    uint8=U8,
    empty_strided=lambda shape, stride, dtype, device: Tensor(shape, stride, dtype, device),
)


def test_capture_restores_stride_dtype_device_and_tensor_attrs(contract):
    result = contract.restore_tensor(
        TORCH,
        {"data": Tensor(dtype=U8), "dtype": "torch.float32", "bitcast": True,
         "shape": [2, 3], "stride": [1, 2], "tensor_attrs": {"is_shuffled": True}},
        "cuda:1",
    )
    assert result.shape == (2, 3)
    assert result.stride() == (1, 2)
    assert result.dtype is F32
    assert result.device == "cuda:1"
    assert result.is_shuffled is True
    assert result.copied_from is not None


@pytest.mark.parametrize("extra, message", [
    ({"dtype": "torch.uint8"}, "payload dtype"),
    ({"dtype": "torch.unavailable"}, "unsupported captured dtype"),
    ({"stride": [-1, 2]}, "invalid captured stride"),
    ({"stride": [1]}, "invalid captured stride"),
    ({"shape": [3, 3]}, "incorrect number of captured elements"),
    ({"bitcast": True}, "missing its recorded dtype"),
])
def test_capture_rejects_invalid_contract(contract, extra, message):
    with pytest.raises((ValueError, TypeError), match=message):
        contract.restore_tensor(TORCH, {"data": Tensor(), **extra}, "cpu")


def test_task_paths_reject_absolute_parent_and_symlink_escapes(contract, tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "inside").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (task / "outside-link").symlink_to(outside, target_is_directory=True)
    assert contract.task_path(task, "inside/data.pt") == task / "inside/data.pt"
    for relative in [str(outside), "../outside/data.pt", "outside-link/data.pt"]:
        with pytest.raises(ValueError):
            contract.task_path(task, relative)


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
def test_tasks_declare_portable_command_and_architecture_contract(task):
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    assert cfg["platform_support"] == {"required_arch": "gfx950", "status": "active"}
    for mode in ("compile", "correctness", "performance"):
        commands = cfg[f"{mode}_command"]
        assert isinstance(commands, list) and len(commands) == 1
        assert f"--timeout {cfg[f'{mode}_timeout']}" in commands[0]
        assert "/shared_nfs/" not in commands[0]
    assert "/shared_nfs/" not in cfg["prompt"]["instructions"]
    assert "not measurements of this run" in cfg["prompt"]["instructions"]
    assert cfg["headkernel"]["source_package"]


@pytest.mark.parametrize("task_name", [
    "qwen3.8-2.4t__fused_moe_2stage_mxfp4",
    "qwen3.8-2.4t__paged_attention_decode",
])
def test_qwen_baseline_is_captured_before_editable_candidate_import(task_name, tmp_path, monkeypatch):
    task = task_directory(task_name)
    overlay = load_file(task / "ut" / "overlay_setup.py", "test_overlay_setup")
    cases = load_file(task / "ut" / "cases.py", "test_qwen_cases")
    binding = cases.META["candidate_bind"]
    target_module, attr = binding["target"].split(":")
    baseline = lambda: "frozen"
    candidate = lambda: "candidate"
    owner = types.ModuleType(target_module)
    setattr(owner, attr, baseline)
    implementation = types.ModuleType(binding["impl_module"])
    setattr(implementation, binding["impl_attr"], candidate)
    # A candidate-controlled helper must not choose the correctness oracle.
    implementation.baseline_callable = lambda: candidate
    monkeypatch.setitem(sys.modules, target_module, owner)
    monkeypatch.setitem(sys.modules, binding["impl_module"], implementation)
    monkeypatch.delitem(sys.modules, "_aka_frozen_baseline_bindings", raising=False)
    (tmp_path / "_overlay_manifest.json").write_text(json.dumps({"rebinds": [binding]}))
    try:
        exec(compile(overlay.SITECUSTOMIZE, "sitecustomize.py", "exec"),
             {"__file__": str(tmp_path / "sitecustomize.py")})
        assert getattr(owner, attr) is candidate
        assert cases.baseline_callable() is baseline
    finally:
        sys.modules.pop("_aka_frozen_baseline_bindings", None)


def test_kimi_launcher_binds_frozen_reference_without_replacing_it(tmp_path):
    bindings = load_file(
        task_directory("kimi-k3__fwd_grouped_kernel_stage1") / "ut" / "bindings.py", "kimi_bindings_test")
    (tmp_path / "baseline_ref").mkdir()
    (tmp_path / "kernel_src").mkdir()
    (tmp_path / "baseline_ref" / "decode_attention.py.orig").write_text(
        "def _decode_grouped_att_m_fwd(): return 'frozen'\n")
    source = tmp_path / "kernel_src" / "geak_mla_stage1.py"
    source.write_text("def make_launcher(base):\n    def candidate(): return base._decode_grouped_att_m_fwd()\n    return candidate\n")
    baseline, candidate = bindings.resolve_pair(str(tmp_path))
    assert baseline is not candidate
    assert baseline() == candidate() == "frozen"
    source.write_text("def make_launcher(base): return base._decode_grouped_att_m_fwd\n")
    with pytest.raises(RuntimeError, match="independent launcher"):
        bindings.resolve_pair(str(tmp_path))


# GLM now uses the official model-free context API; its positive and negative
# initialization controls live in test_top5_glm_device_source.py.


@pytest.mark.parametrize("filename, hash_field", [
    ("reference_io.pt", "reference_io_sha256"),
    ("timing_geometry.pt", "timing_geometry_sha256"),
])
def test_frozen_capture_digest_is_checked_before_each_deserialization(tmp_path, filename, hash_field):
    import hashlib
    helper = tmp_path / "task_contract.py"
    helper.write_text((TASKS[0] / "ut" / "task_contract.py").read_text())
    local_contract = load_file(helper, "local_capture_test")
    payload = tmp_path / filename
    payload.write_bytes(b"authoritative capture")
    (tmp_path / "meta.json").write_text(json.dumps({
        hash_field: hashlib.sha256(payload.read_bytes()).hexdigest(),
    }))
    loads = []
    torch = types.SimpleNamespace(load=lambda path, **kw: loads.append((path, kw)) or "loaded")
    assert local_contract.verified_torch_load(torch, payload, map_location="cpu") == "loaded"
    assert len(loads) == 1
    payload.write_bytes(b"modified after provisioning")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        local_contract.verified_torch_load(torch, payload, map_location="cpu")
    assert len(loads) == 1
    (tmp_path / "meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="missing frozen SHA-256"):
        local_contract.verified_torch_load(torch, payload)
    assert len(loads) == 1


@pytest.mark.parametrize("task_name", [
    "deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl",
    "deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4",
])
def test_deepseek_rejects_persistent_output_object_and_missing_timing_case(task_name, monkeypatch):
    cases = load_file(task_directory(task_name) / "ut" / "cases.py", "deepseek_cases_test")

    class Output:
        def untyped_storage(self):
            return self

        def data_ptr(self):
            return 42

    output = Output()
    monkeypatch.setattr(cases, "_torch", lambda: types.SimpleNamespace(
        is_tensor=lambda obj: isinstance(obj, Output)))
    monkeypatch.setattr(cases, "_invoke", lambda _: output)
    assert cases.call({}) is output
    with pytest.raises(RuntimeError, match="shared_output_buffer"):
        cases.call({})
    monkeypatch.setattr(cases, "_oracle", lambda: [{"sig": "present"}])
    monkeypatch.setattr(cases, "_bucket_sigs", lambda: ["present", "missing"])
    with pytest.raises(RuntimeError, match="benchmark cases missing"):
        cases.timing_cases(None, {})


def test_tensor_attribute_restoration_cannot_silently_fail(contract):
    class RejectAttributes:
        __slots__ = ()

    with pytest.raises(AttributeError):
        contract.tensor_attrs(RejectAttributes(), {"is_shuffled": True})


def test_qwen_rmsnorm_retains_real_live_shape_evidence():
    path = task_directory("qwen3.8-2.4t__gemma_fused_add_rmsnorm") / "ut" / "unittest.py"
    unittest = load_file(path, "rmsnorm_shape_evidence_test")
    result = unittest._verify_provenance()
    assert result["live_shapes"] == [64, 8192]
    assert result["layout_processes"] == 8


def test_flydsl_moe_namespaces_remain_independent_and_baseline_is_hashed(tmp_path, monkeypatch):
    import hashlib
    helper = load_file(
        task_directory("kimi-k3__moe_gemm1_stage1") / "ut" / "flydsl_package.py",
        "flydsl_package_test",
    )
    monkeypatch.setitem(sys.modules, "flydsl", types.SimpleNamespace(__version__="0.2.4+captured"))
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    # Unrelated package-wide API imports must not become task dependencies.
    for folder, answer in [(baseline, "baseline"), (candidate, "candidate")]:
        (folder / "__init__.py").write_text("raise RuntimeError('unrelated API import')\n")
        (folder / "detail.py").write_text(f"VALUE = {answer!r}\n")
        (folder / "moe_kernels.py").write_text(
            "from .detail import VALUE\ndef flydsl_moe_stage1(): return VALUE\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"baseline_files": {
        name: hashlib.sha256((baseline / name).read_bytes()).hexdigest()
        for name in ("moe_kernels.py", "detail.py")
    }}))
    try:
        base = helper.load_moe_package(baseline, "test_frozen_moe", frozen_manifest=manifest)
        cand = helper.load_moe_package(candidate, "test_candidate_moe")
        assert base.flydsl_moe_stage1() == "baseline"
        assert cand.flydsl_moe_stage1() == "candidate"
        assert base.flydsl_moe_stage1 is not cand.flydsl_moe_stage1
        (baseline / "detail.py").write_text("VALUE = 'tampered'\n")
        with pytest.raises(RuntimeError, match="frozen FlyDSL dependency changed"):
            helper.load_moe_package(baseline, "test_changed_moe", frozen_manifest=manifest)
    finally:
        for name in list(sys.modules):
            if name.startswith(("test_frozen_moe", "test_candidate_moe", "test_changed_moe")):
                sys.modules.pop(name, None)
