"""CPU loader and negative controls; these are not GPU correctness claims."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from head_kernel_test_utils import task_directory

ROOT = Path(__file__).resolve().parents[1]
TASK = task_directory("glm-5.3-flash__fused_moe_kernel")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def contract():
    return load(TASK / "ut/device_source_contract.py", "glm_device_contract_test")


@pytest.fixture
def small_task(tmp_path):
    task = tmp_path / "task"
    ut = task / "ut"
    (task / "source").mkdir(parents=True)
    (ut / "baseline_ref").mkdir(parents=True)
    overlay = ut / "baseline_overlay"
    (overlay / "_patched").mkdir(parents=True)
    device = "def fused_moe_kernel(x):\n    return x * 2\n\ndef launch(x):\n    return fused_moe_kernel(x)\n"
    dispatcher = "from fake_runtime.device import launch\ndef fused_experts_impl(x):\n    return launch(x)\n"
    (task / "source/device.py").write_text(device)
    (ut / "baseline_ref/device.py.orig").write_text(device)
    (overlay / "_patched/device.py").write_text(device)
    (overlay / "_patched/dispatcher.py").write_text(dispatcher)
    shutil.copyfile(TASK / "ut/baseline_overlay/sitecustomize.py", overlay / "sitecustomize.py")
    manifest = {"modules": [{"module": "fake_runtime.device", "file": "_patched/device.py"},
                            {"module": "fake_runtime.dispatcher", "file": "_patched/dispatcher.py"}],
                "rebinds": [], "captures": [], "markers": []}
    (overlay / "_overlay_manifest.json").write_text(json.dumps(manifest))
    frozen = ["baseline_ref/device.py.orig", "baseline_overlay/_patched/device.py",
              "baseline_overlay/_patched/dispatcher.py"]
    metadata = {"candidate_source": "source/device.py", "device_reference": "baseline_ref/device.py.orig",
                "device_module": "fake_runtime.device", "dispatcher_module": "fake_runtime.dispatcher",
                "editable_functions": ["fused_moe_kernel"],
                "frozen_files": [{"file": f, "sha256": hashlib.sha256((ut / f).read_bytes()).hexdigest()}
                                 for f in frozen]}
    (ut / "device_source_contract.json").write_text(json.dumps(metadata))
    package = tmp_path / "installed/fake_runtime"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    # A fallback to the installed module must be observable as wrong output.
    (package / "device.py").write_text("def launch(x): return -999\n")
    (package / "dispatcher.py").write_text("def fused_experts_impl(x): return -888\n")
    return task, package.parent


def run_worker(installed, overlay):
    program = """
import hashlib, importlib, json, pathlib, runpy, sys
sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[2] + '/sitecustomize.py')
dispatcher = importlib.import_module('fake_runtime.dispatcher')
device = importlib.import_module('fake_runtime.device')
path = pathlib.Path(device.__file__).resolve()
print(json.dumps({'value': dispatcher.fused_experts_impl(7),
                  'file': str(pathlib.Path(dispatcher.__file__).resolve()),
                  'device_source': {'file': str(path),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}))
"""
    result = subprocess.run([sys.executable, "-S", "-c", program, str(installed), str(overlay)],
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_device_is_bound_before_dispatcher_and_baseline_stays_independent(contract, small_task):
    task, installed = small_task
    source = task / "source/device.py"
    source.write_text(source.read_text().replace("return x * 2", "return x * 3"))
    baseline, candidate = contract.build_candidate_overlay(task / "ut")
    base = run_worker(installed, baseline)
    changed = run_worker(installed, candidate)
    assert base["value"] == 14
    assert changed["value"] == 21
    assert run_worker(installed, baseline)["value"] == 14
    contract.verify_leg_identities(task / "ut", baseline, candidate, base, changed)
    with pytest.raises(RuntimeError, match="independent"):
        contract.verify_leg_identities(task / "ut", baseline, candidate, base, base)


@pytest.mark.parametrize("change", [
    lambda s: s.replace("return fused_moe_kernel(x)", "return x * 0"),
    lambda s: "import os\n" + s,
    lambda s: s.replace("def fused_moe_kernel(x)", "def fused_moe_kernel(x, extra=0)"),
    lambda s: s.replace("def fused_moe_kernel(x)", "@print\ndef fused_moe_kernel(x)"),
])
def test_host_launcher_import_signature_and_decorator_changes_are_rejected(contract, small_task, change):
    task, _ = small_task
    path = task / "source/device.py"
    path.write_text(change(path.read_text()))
    with pytest.raises(RuntimeError, match="host contract"):
        contract.build_candidate_overlay(task / "ut")


def test_frozen_source_corruption_is_rejected(contract, small_task):
    task, _ = small_task
    (task / "ut/baseline_ref/device.py.orig").write_text("corrupted reference")
    with pytest.raises(RuntimeError, match="frozen GLM source changed"):
        contract.build_candidate_overlay(task / "ut")


def test_real_source_pin_and_device_only_edit_boundary(contract):
    cfg = yaml.safe_load((TASK / "config.yaml").read_text())
    assert cfg["source_file_path"] == ["source/fused_moe_triton_kernels.py"]
    assert cfg["target_kernel_functions"] == ["fused_moe_kernel"]
    source = contract.validate_candidate(TASK / "ut")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "9c3342d3147e7d60a78a2c934111f0fc1becbb8df1d2d32aafc82e6c8a0b2e70"
    assert "pre_run_patch" not in cfg["headkernel"]
    assert "runtime_inputs" not in cfg["headkernel"]


def fake_runtime(monkeypatch, *, wrong_config=False):
    state = {"installed": 0, "restored": 0}
    context = SimpleNamespace(server_args=None)
    config = SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=False),
                             moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=wrong_config,
                                                 moe_runner_backend="triton"))
    class Override:
        def __init__(self, fields):
            state["fields"] = fields
        def install(self):
            state["installed"] += 1
            context.server_args = SimpleNamespace(**state["fields"])
        def restore(self):
            state["restored"] += 1
    context.override_server_args = lambda **fields: Override(fields)
    rc = ModuleType("sglang.srt.runtime_context")
    rc.get_context = lambda: context
    rc.get_exec = lambda: config
    srt = ModuleType("sglang.srt")
    srt.runtime_context = rc
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", rc)
    return state, config


def test_model_free_context_keeps_settings_and_parallel_initialization(monkeypatch):
    bootstrap = load(TASK / "ut/sglang_bootstrap.py", "glm_bootstrap_cpu")
    state, config = fake_runtime(monkeypatch)
    calls = []
    monkeypatch.setattr(bootstrap, "_init_parallel", lambda: calls.append("parallel"))
    bootstrap.ensure()
    bootstrap.ensure()
    assert state["installed"] == 1 and calls == ["parallel"]
    assert "model_path" not in state["fields"] and "tokenizer_path" not in state["fields"]
    assert state["fields"]["tp_size"] == 8
    config.moe.enable_fused_moe_sum_all_reduce = True
    with pytest.raises(RuntimeError, match="runtime config"):
        bootstrap.ensure()
    bootstrap.cleanup()
    assert state["restored"] == 1


def test_context_mismatch_restores_before_any_parallel_init(monkeypatch):
    bootstrap = load(TASK / "ut/sglang_bootstrap.py", "glm_bootstrap_bad_cpu")
    state, _ = fake_runtime(monkeypatch, wrong_config=True)
    monkeypatch.setattr(bootstrap, "_init_parallel", lambda: pytest.fail("must not initialize"))
    with pytest.raises(RuntimeError, match="runtime config"):
        bootstrap.ensure()
    assert state["restored"] == 1 and not bootstrap._done


def test_parallel_workers_use_unique_task_local_file_rendezvous(tmp_path, monkeypatch):
    calls = []
    ps = SimpleNamespace(
        model_parallel_is_initialized=lambda: False,
        init_distributed_environment=lambda **kw: calls.append(("world", kw)),
        initialize_model_parallel=lambda **kw: calls.append(("tp", kw)),
        destroy_model_parallel=lambda: calls.append(("destroy_tp", {})),
        destroy_distributed_environment=lambda: calls.append(("destroy_world", {})),
    )
    distributed = ModuleType("sglang.srt.distributed")
    distributed.parallel_state = ps
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda n: calls.append(("device", n)))))
    workers = [load(TASK / "ut/sglang_bootstrap.py", f"glm_rendezvous_{i}") for i in range(2)]
    for worker in workers:
        monkeypatch.setattr(worker, "__file__", str(tmp_path / "task/ut/sglang_bootstrap.py"))
        worker._init_parallel()
    worlds = [kw for name, kw in calls if name == "world"]
    assert len(worlds) == 2
    stores = [kw["distributed_init_method"] for kw in worlds]
    assert stores[0] != stores[1]
    assert all(uri.startswith((tmp_path / "task/build").as_uri() + "/") for uri in stores)
    assert all(kw["world_size"] == 1 and kw["rank"] == 0 and kw["backend"] == "nccl" for kw in worlds)
    assert [kw for name, kw in calls if name == "tp"] == [
        {"tensor_model_parallel_size": 1}, {"tensor_model_parallel_size": 1}]
    for worker in workers:
        worker.cleanup()
    assert not list((tmp_path / "task/build").glob("glm-rendezvous-*"))
