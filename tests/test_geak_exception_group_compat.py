"""Exception diagnostics must also work on Python 3.10, without eager SDK imports."""
import ast
import builtins
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from agents.geak_v4 import workflow_runner as runner


@pytest.mark.parametrize("backport_state", ["missing", "invalid_type"])
def test_plain_failure_without_builtin_or_usable_backport(monkeypatch, backport_state):
    unavailable = None
    if backport_state == "invalid_type":
        unavailable = ModuleType("exceptiongroup")
        unavailable.BaseExceptionGroup = object()  # Never pass a non-type to isinstance.
    original = RuntimeError("GEAK Workflow tool returned an error")
    # Ordinary exceptions must not be mistaken for groups by duck typing.
    original.exceptions = [ValueError("FAKE_SECRET")]
    identity = {}
    with monkeypatch.context() as patch:
        patch.delattr(builtins, "BaseExceptionGroup", raising=False)
        patch.delattr(builtins, "ExceptionGroup", raising=False)
        patch.setitem(sys.modules, "exceptiongroup", unavailable)
        runner._record_sdk_failure(identity, original)
    assert identity["sdk_diagnostics"]["failure"] == {
        "leaves": [{"exception_class": "RuntimeError", "reason": "workflow_tool_error"}],
        "truncated": False,
    }
    assert "FAKE_SECRET" not in json.dumps(identity)


def test_missing_builtin_traverses_real_backport_groups(monkeypatch):
    backport = pytest.importorskip("exceptiongroup")
    # On 3.11+ the public backport exports builtin aliases. Use its actual
    # Python implementation and expose precisely the public binding used on 3.10.
    from exceptiongroup._exceptions import BaseExceptionGroup, ExceptionGroup

    original = RuntimeError("GEAK Workflow tool returned an error")
    native_failure = OSError("FAKE_SECRET")
    native_failure.exit_code = 23
    nested = ExceptionGroup("FAKE_SECRET", [ExceptionGroup("FAKE_SECRET", [original]), native_failure])
    identity = {}
    with monkeypatch.context() as patch:
        patch.delattr(builtins, "BaseExceptionGroup", raising=False)
        patch.delattr(builtins, "ExceptionGroup", raising=False)
        patch.setattr(backport, "BaseExceptionGroup", BaseExceptionGroup)
        runner._record_sdk_failure(identity, nested)
    assert identity["sdk_diagnostics"]["failure"] == {"leaves": [
        {"exception_class": "RuntimeError", "reason": "workflow_tool_error"},
        {"exception_class": "OSError", "reason": "unclassified", "exit_code": 23},
    ], "truncated": False}
    assert "FAKE_SECRET" not in json.dumps(identity)


def test_dry_run_imports_no_sdk_or_backport_with_missing_builtin(tmp_path):
    kernel = tmp_path / "kernel"
    workflow = tmp_path / "workflow"
    kernel.mkdir()
    workflow.mkdir()
    (workflow / "kernel_workflow.js").write_text("// No execution in this test.\n")
    handoff = tmp_path / "handoff.json"
    output = tmp_path / "result.json"
    handoff.write_text(json.dumps({"schema_version": 1, "kernel_path": str(kernel), "workflow_dir": str(workflow),
                                  "eval_dir": str(tmp_path / "eval")}))
    script = """
import builtins, runpy, sys
original_import = builtins.__import__
attempts = []
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'anyio', 'claude_agent_sdk', 'exceptiongroup'}:
        attempts.append(name)
        raise ImportError('optional dependency blocked in offline fixture')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
for name in ('BaseExceptionGroup', 'ExceptionGroup'):
    if hasattr(builtins, name):
        delattr(builtins, name)
namespace = runpy.run_path(sys.argv[1])
assert namespace['main']([sys.argv[2], sys.argv[3], '--dry-run']) == 0
assert attempts == [], attempts
identity = {}
namespace['_record_sdk_failure'](identity, RuntimeError('opaque'))
assert identity['sdk_diagnostics']['failure']['leaves'][0]['exception_class'] == 'RuntimeError'
assert attempts == ['exceptiongroup'], attempts
"""
    result = subprocess.run([sys.executable, "-B", "-S", "-c", script,
                             runner.__file__, str(handoff), str(output)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "dry_run"


def test_diagnostic_sources_parse_with_python310_grammar():
    root = Path(runner.__file__).resolve().parents[2]
    for relative in ("agents/geak_v4/workflow_runner.py", "agents/geak/engine_worker.py",
                     "tests/test_geak_sdk_diagnostics.py", "tests/test_geak_exception_group_compat.py"):
        ast.parse((root / relative).read_text(), filename=relative, feature_version=(3, 10))
