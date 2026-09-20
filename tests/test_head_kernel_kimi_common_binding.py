"""CPU integration checks for the common selector and real Kimi UT adapters.

Image packages, capture serialization, and numerical operand construction are
mocked. The protected unit-test import, adapter loaders, baseline/source paths,
and common selected_cases dispatch are the shipped implementations.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

from head_kernel_test_utils import task_directory

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tasks/head_kernels"
TASK_NAMES = ["kimi-k3__fwd_grouped_kernel_stage1",
              "kimi-k3__moe_gemm1_stage1", "kimi-k3__moe_gemm2_stage2"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MarkerTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.marker = None


class Generator:
    def __init__(self, **kwargs):
        pass

    def manual_seed(self, value):
        return self


@pytest.fixture
def mocked_image(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.Generator = Generator
    torch.bfloat16 = "bf16"
    torch.zeros = lambda shape, dtype, device: MarkerTensor(shape, dtype)
    torch.serialization = SimpleNamespace(clear_safe_globals=lambda: None)
    def load_capture(*args, **kwargs):
        assert kwargs["weights_only"] is True
        return {"cases": []}
    torch.load = load_capture
    flydsl = ModuleType("flydsl")
    flydsl.__version__ = "0.2.4+test-image"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "flydsl", flydsl)
    return torch


def clear_kimi_modules():
    for name in list(sys.modules):
        if name.startswith(("geak_frozen_flydsl", "geak_candidate_flydsl",
                            "_aka_kimi_mla_", "_headkernel_cases_test")):
            sys.modules.pop(name, None)


def make_task_fixture(tmp_path, task_name):
    original = task_directory(task_name)
    task = tmp_path / task_name
    ut = task / "ut"
    ut.mkdir(parents=True)
    for filename in ["unittest.py", "harness_lib.py", "task_contract.py", "meta.json"]:
        shutil.copyfile(original / "ut" / filename, ut / filename)
    metadata = json.loads((ut / "meta.json").read_text())
    if task_name == TASK_NAMES[0]:
        shutil.copyfile(original / "ut/bindings.py", ut / "bindings.py")
        baseline = ut / "baseline_ref/decode_attention.py.orig"
        baseline.parent.mkdir()
        baseline.write_text(
            "def _decode_grouped_att_m_fwd(*args, **kwargs):\n"
            "    args[3].marker = 'frozen-baseline'\n"
            "    args[4].marker = 'frozen-baseline-lse'\n")
        candidate = task / "source/geak_mla_stage1.py"
        candidate.parent.mkdir()
        (ut / "kernel_src").symlink_to("../source", target_is_directory=True)
    else:
        shutil.copyfile(original / "ut/flydsl_package.py", ut / "flydsl_package.py")
        baseline = ut / "baseline_src/flydsl/moe_kernels.py"
        baseline.parent.mkdir(parents=True)
        (baseline.parent / "__init__.py").write_text("raise RuntimeError('wrong generic loader')\n")
        entry = metadata["entry_attr"]
        baseline.write_text(f"def {entry}(*args, **kwargs): return 'frozen-baseline'\n")
        (ut / "dependency_manifest.json").write_text(json.dumps({"baseline_files": {
            "moe_kernels.py": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        }}))
        candidate = task / "source/flydsl/moe_kernels.py"
        candidate.parent.mkdir(parents=True)
        (candidate.parent / "__init__.py").write_text("raise RuntimeError('wrong generic loader')\n")
        (ut / "kernel_src").symlink_to("../source", target_is_directory=True)
        payload = b"mock serialized capture, not GPU data"
        (ut / "reference_io.pt").write_bytes(payload)
        metadata["reference_io_sha256"] = hashlib.sha256(payload).hexdigest()
        (ut / "meta.json").write_text(json.dumps(metadata))
    return task, ut, candidate, metadata


def write_candidate(path, task_name, marker, meta):
    if task_name == TASK_NAMES[0]:
        path.write_text(
            "def make_launcher(base):\n"
            "    def launch(*args, **kwargs):\n"
            f"        args[3].marker = {marker!r}\n"
            f"        args[4].marker = {marker + '-lse'!r}\n"
            "    return launch\n")
    else:
        path.write_text(f"def {meta['entry_attr']}(*args, **kwargs): return {marker!r}\n")


def attach_numerical_stubs(module, task_name, torch):
    if task_name == TASK_NAMES[0]:
        module._synth = lambda bs, ctx, rng: {
            "pos": [None] * 8, "kw": {},
            "out_slots": {3: ((bs,), torch.bfloat16), 4: ((bs,), torch.bfloat16)},
        }
    else:
        module.build_inputs = lambda spec: {"token_num": int(spec["token_num"])}
        module._invoke = lambda fn, inputs, output, zero=False: fn()


def marker(value):
    return value[0].marker if isinstance(value, tuple) else value


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_actual_common_selector_engages_independent_kimi_source(tmp_path, mocked_image, task_name):
    bench = load(SUITE / "_support/_bench.py", "kimi_common_bench_test")
    task, ut, candidate, meta = make_task_fixture(tmp_path, task_name)
    try:
        for version in ["candidate-v1", "candidate-v2-source-edit"]:
            clear_kimi_modules()
            write_candidate(candidate, task_name, version, meta)
            # Use the common worker's actual module loader on the shipped UT.
            module = bench.load_module("_headkernel_cases_test", ut / "unittest.py")
            assert module.BASELINE_FN is not module.CANDIDATE_FN
            assert Path(module.BASELINE_FN.__code__.co_filename).is_relative_to(ut)
            assert Path(module.CANDIDATE_FN.__code__.co_filename).resolve().is_relative_to(task / "source")
            attach_numerical_stubs(module, task_name, mocked_image)
            reference_rows, reference_call = bench.selected_cases(
                module, module.h, meta, mocked_image, reference=True)
            candidate_rows, candidate_call = bench.selected_cases(
                module, module.h, meta, mocked_image, reference=False)
            assert bench.validate_cases(reference_rows) == bench.validate_cases(candidate_rows)
            assert reference_rows
            assert all(marker(reference_call(row["args"])) == "frozen-baseline"
                       for row in reference_rows)
            assert all(marker(candidate_call(row["args"])) == version
                       for row in candidate_rows)
    finally:
        clear_kimi_modules()


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_declared_standalone_adapter_validates_real_task_layout(task_name):
    ut = task_directory(task_name) / "ut"
    meta = json.loads((ut / "meta.json").read_text())
    path, entry = meta["standalone_binding"].split(":")
    adapter = load(ut / path, "kimi_layout_test")
    assert callable(getattr(adapter, entry))
    assert adapter.validate_layout(ut) is True
