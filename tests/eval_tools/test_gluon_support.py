"""Gluon and unmapped languages fail closed; schema-v3 workspaces profile solution/."""
from __future__ import annotations

import pytest

from src.eval_tools.contracts import ArtifactKind, CapabilityCheck, CapabilityState, KernelLanguage
from src.eval_tools.plugins import get_plugin, plugin_ids
from src.eval_tools.task_declarations import candidate_paths, candidate_symbols, protected_paths
from src.eval_tools.task_profile import resolve_builtin_capability, resolve_task_profile
from tests.eval_tools.test_plugins import context, profile, runtime


def _v3_config(language: str, entry: str = "solution/kernel.py") -> dict:
    return {
        "schema_version": 2,
        "candidate": {"language": language, "initial_state": "unimplemented",
                      "editable": [{"path": "solution", "scope": "tree"}],
                      "entrypoints": [{"file": entry, "kind": "function", "symbol": "run"}]},
        "baseline": {"kind": "provided", "language": "triton",
                     "source_files": ["baseline/kernel.py", "reference.py", "workloads.json"]},
        "evaluation": {"workloads": "workloads.json"},
        "contract": {"schema_version": 3, "task_file": "task.yaml", "target_language": language,
                     "baseline_language": "triton", "operator": {"name": "rms_norm", "category": "normalization"}},
    }


@pytest.mark.parametrize("language,expected", [
    ("triton", KernelLanguage.TRITON), ("gluon", KernelLanguage.GLUON), ("hip", KernelLanguage.HIP),
    ("flydsl", KernelLanguage.FLYDSL), ("gpu", KernelLanguage.UNKNOWN),
    ("tilelang", KernelLanguage.UNKNOWN), ("other", KernelLanguage.UNKNOWN),
])
def test_v3_language_mapping_is_exact(language, expected):
    assert resolve_task_profile(_v3_config(language)).language == expected


def test_v3_profile_describes_the_solution_tree():
    config = _v3_config("gluon")
    profile_ = resolve_task_profile(config)
    assert profile_.source_files == ("solution", "solution/kernel.py")
    assert profile_.target_functions == ()  # run() is a host function, not a dispatch symbol
    assert profile_.artifact_kind == ArtifactKind.PYTHON_JIT
    assert profile_.framework == "standalone"
    assert profile_.evidence["target_language"] == "gluon"
    assert candidate_paths(config) == ("solution", "solution/kernel.py")
    assert candidate_symbols(_v3_config("hip", "solution/kernel.hip")) == ()
    assert {"_aka_eval", "task.yaml", "baseline/kernel.py", "reference.py", "workloads.json"} <= set(
        protected_paths(config))
    assert "solution" not in protected_paths(config)


def test_v2_symbols_and_protection_are_unchanged_without_a_contract():
    config = _v3_config("triton")
    del config["contract"]
    assert candidate_symbols(config) == ("run",)
    assert "_aka_eval" not in protected_paths(config)


@pytest.mark.parametrize("language", ["gluon", "gpu", "tilelang", "other"])
def test_builtin_matrix_reports_unsupported_not_not_applicable(language):
    profile_ = resolve_task_profile(_v3_config(language))
    for tool in ("triton_fpsan", "gpu_asan", "rocjitsu", "rocjitsu_waitcheck", "rocjitsu_consan", "hip_fpsan"):
        capability = resolve_builtin_capability(tool, profile_, CapabilityCheck.ready())
        assert capability.effective.state == CapabilityState.UNSUPPORTED, tool
        assert not capability.ready
        assert capability.engine.reason_code.endswith("_GLUON_UNSUPPORTED" if language == "gluon" else "_LANGUAGE_UNSUPPORTED")


# page_fault_guard, csan and tilelens exist only through the schema-v3 runner;
# test_v3_sanitizers covers their language handling.
V3_ONLY_TOOLS = {"page_fault_guard", "csan", "tilelens"}


@pytest.mark.parametrize("language", [KernelLanguage.GLUON, KernelLanguage.UNKNOWN])
@pytest.mark.parametrize("tool", sorted(set(plugin_ids()) - V3_ONLY_TOOLS))
def test_plugins_fail_closed_for_gluon_and_unknown(tmp_path, tool, language):
    # Even a profile that claims framework=triton cannot revive a Gluon candidate.
    capability = get_plugin(tool).assess(
        context(tmp_path, profile(language, framework="triton"), {"command": ["true"]}),
        runtime(triton_fpsan=True, gpu_asan=True, xnack_supported=True),
    )
    assert capability.effective.state == CapabilityState.UNSUPPORTED
    assert capability.effective.reason_code.endswith(
        "gluon_unsupported" if language == KernelLanguage.GLUON else "language_unsupported")
    assert not capability.ready


def test_supported_languages_are_not_blocked_by_the_language_gate(tmp_path):
    capability = get_plugin("triton_fpsan").assess(
        context(tmp_path, profile(KernelLanguage.TRITON, framework="triton"), {"command": ["true"]}),
        runtime(triton_fpsan=True),
    )
    assert capability.ready
    hip_on_triton_tool = get_plugin("triton_fpsan").assess(
        context(tmp_path, profile(KernelLanguage.HIP, framework="standalone"), {}), runtime())
    assert hip_on_triton_tool.effective.state == CapabilityState.NOT_APPLICABLE
