"""Framework diagnostics must not invalidate immutable task input checks."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fp4_runner_hash", ROOT / "scripts/task_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for name, data in (("source/kernel.py", b"device body"), ("ut/reference.py", b"independent oracle"),
                       ("fixtures/blob.bin", b"actual captured storage"), ("cases.json", b"case controls")):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return module


def test_live_validator_logging_and_private_extension_cache_preserve_hash(runner):
    audit = runner.ROOT / ".validator_audit"
    audit.mkdir()
    stderr = audit / "compile_1.stderr.log"
    stderr.write_text("")
    before = runner.package_hash()
    stderr.write_text("[aiter] import [module_aiter_core]\n")
    (audit / "attempt.json").write_text('{"running": true}')
    cache = runner.ROOT / ".validator_torch_extensions"
    cache.mkdir()
    (cache / "extension.so").write_bytes(b"runtime compiler output")
    assert runner.package_hash() == before


@pytest.mark.parametrize("relative", ["source/kernel.py", "ut/reference.py", "fixtures/blob.bin", "cases.json",
                                       "ut/.validator_audit/hidden.py", "fixtures/.validator_torch_extensions/hidden.bin"])
def test_protected_changes_and_nested_diagnostic_names_change_hash(runner, relative):
    before = runner.package_hash()
    path = runner.ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"changed protected input")
    assert runner.package_hash() != before


def test_directory_symlink_cannot_remove_protected_input_coverage(runner):
    original = runner.ROOT / "ut"
    original.rename(runner.ROOT / "real-ut")
    original.symlink_to("real-ut", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner.package_hash()
