"""CPU regressions for protected harness module/compiled-owner lifetimes.

The complete task harness is imported. GPU-only dependencies are fail-on-use
stubs, and candidate modules contain synthetic Python owners, not GPU kernels.
"""
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = (("sglang/gdn_l2norm_fwd", "gdn_l2norm_src", "load_module"),
         ("sglang/fused_norm_gate", "fused_norm_gate_src", "load_module"),
         ("aiter/rope_fwd", "rope_fwd_src", "_load_source"),
         ("sglang/fused_gdn_gating", "fused_gdn_gating_src", "load_module"))


def _unexpected_gpu_use(*args, **kwargs):
    raise AssertionError("Module-lifetime checks must not execute a GPU harness action")


def _candidate_source(version):
    return (
        f"VERSION = {version!r}\n"
        "class CompiledOwner:\n    pass\n"
        "owner = CompiledOwner()\n"
        "def entry():\n    return owner\n"
    )


@pytest.fixture(params=TASKS)
def harness(request, tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(_candidate_source("initial"))
    runtime = ModuleType("task_runtime")
    runtime.candidate_relative_path = lambda: str(candidate)
    perf = ModuleType("_aka_benchmark")
    perf.TimedRun = _unexpected_gpu_use
    perf.benchmark_cuda_graph_or_events = _unexpected_gpu_use
    scripts = ModuleType("scripts")
    scripts.__path__ = []
    replay = ModuleType("scripts.replay_checks")
    for name in ("require_unchanged", "require_tensor_contract", "verify_timed_run", "verify_timed_pair"):
        setattr(replay, name, _unexpected_gpu_use)
    for name, module in (("task_runtime", runtime), ("_aka_benchmark", perf),
                         ("scripts", scripts), ("scripts.replay_checks", replay)):
        monkeypatch.setitem(sys.modules, name, module)
    # The actual loader owns this alias; restore any previous test's module.
    task, alias, loader = request.param
    monkeypatch.setitem(sys.modules, alias, ModuleType(alias))
    monkeypatch.chdir(tmp_path)
    path = ROOT / "tasks" / "triton2flydsl" / task / "test_kernel_harness.py"
    spec = importlib.util.spec_from_file_location("isolated_module_lifetime_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, candidate=candidate, alias=alias, load=getattr(module, loader))


def test_same_source_reuses_module_and_retains_owner_without_import_alias(harness):
    first = harness.load()
    assert harness.load() is first
    module_ref = weakref.ref(first)
    owner_ref = weakref.ref(first.owner)
    # Import-alias ownership alone is insufficient: the harness must retain the
    # compiled module throughout correctness plus performance in this action.
    sys.modules[harness.alias] = ModuleType(harness.alias)
    del first
    for _ in range(3):
        gc.collect()
        assert module_ref() is not None
        assert owner_ref() is not None
        assert harness.load() is module_ref()


def test_changed_source_is_separate_and_old_owner_remains_alive(harness):
    first = harness.load()
    first_ref = weakref.ref(first)
    old_owner = weakref.ref(first.owner)
    harness.candidate.write_text(_candidate_source("different-source-and-size"))
    second = harness.load()
    assert second is not first
    assert second.VERSION == "different-source-and-size"
    assert first.VERSION == "initial"
    assert harness.load() is second
    del first
    gc.collect()
    assert first_ref() is not None
    assert old_owner() is not None
    assert second.owner is not old_owner()


def test_failed_import_is_not_reused_as_a_complete_module(harness):
    harness.candidate.write_text('raise RuntimeError("import is incomplete")\n')
    for _ in range(2):
        with pytest.raises(RuntimeError, match="import is incomplete"):
            harness.load()
    harness.candidate.write_text(_candidate_source("recovered-valid-module"))
    recovered = harness.load()
    assert recovered.VERSION == "recovered-valid-module"
    assert harness.load() is recovered


def test_identical_bytes_at_a_different_source_path_use_a_distinct_module(harness, tmp_path):
    first = harness.load()
    alternative = tmp_path / "other_candidate.py"
    alternative.write_bytes(harness.candidate.read_bytes())
    harness.module.SOURCE_FILE = str(alternative)
    second = harness.load()
    assert second is not first
    assert second.VERSION == first.VERSION
    assert harness.load() is second
