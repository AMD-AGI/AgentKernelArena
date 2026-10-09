from types import SimpleNamespace

import pytest

from src.scripts import rocm_sdk_runtime as wrapper


def test_profiler_uses_core_tree_without_losing_workload_arguments(tmp_path, monkeypatch):
    core = tmp_path / "_rocm_sdk_core"
    executable = core / "bin" / "rocprofv3"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(wrapper.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(core / "__init__.py")))
    devel = tmp_path / "_rocm_sdk_devel" / "lib"
    env = {"LD_LIBRARY_PATH": f"{devel}:/custom/lib:{core / 'lib'}", "ROCR_VISIBLE_DEVICES": "2"}
    arguments = ["--kernel-trace", "--", "python", "workload with spaces.py"]

    command, actual_env = wrapper.profiler_command(arguments, env)

    assert command == [str(executable), *arguments]
    assert actual_env == {**env, "LD_LIBRARY_PATH": f"{core / 'lib'}:/custom/lib"}
    assert env["LD_LIBRARY_PATH"].startswith(str(devel))


def test_missing_core_profiler_does_not_fall_back_to_conflicting_devel(tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py")))
    with pytest.raises(RuntimeError, match="profiler is missing"):
        wrapper.profiler_command([], {})


def test_workload_environment_does_not_require_a_profiler(tmp_path, monkeypatch):
    core = tmp_path / "_rocm_sdk_core"
    monkeypatch.setattr(wrapper.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(core / "__init__.py")))
    original = {"LD_LIBRARY_PATH": "/custom", "ROCR_VISIBLE_DEVICES": "3"}
    assert wrapper.runtime_environment(original) == {
        **original, "LD_LIBRARY_PATH": f"{core / 'lib'}:/custom",
    }


def test_missing_core_fails_closed(monkeypatch):
    monkeypatch.setattr(wrapper.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="no ROCm SDK core"):
        wrapper.runtime_environment({})
