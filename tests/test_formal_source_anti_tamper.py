from __future__ import annotations

import os
from pathlib import Path

import pytest

from src import evaluator
from src.evaluator_utils import (
    FORMAL_SOURCE_ANTI_TAMPER_POLICY,
    FORMAL_SOURCE_ANTI_TAMPER_RULES_SHA256,
    FORMAL_SOURCE_ANTI_TAMPER_SCHEMA,
    inspect_formal_source_anti_tamper,
)


def _workspace(tmp_path: Path, source: str) -> tuple[Path, dict]:
    workspace = tmp_path / "workspace"
    source_directory = workspace / "source"
    source_directory.mkdir(parents=True)
    (source_directory / "candidate.py").write_text(source, encoding="utf-8")
    return workspace, {"source_file_path": ["source/candidate.py"]}


def _rules(report: dict) -> set[str]:
    return {
        violation["rule"]
        for file_report in report["files"]
        for violation in file_report["violations"]
    } | {violation["rule"] for violation in report["violations"]}


def test_clean_triton_source_has_stable_anchored_report(tmp_path: Path) -> None:
    workspace, config = _workspace(
        tmp_path,
        "import torch\nimport triton\nimport triton.language as tl\n"
        "from math import ceil\n\n"
        "@triton.jit\ndef kernel(x):\n    return tl.load(x) + ceil(0.1)\n",
    )

    initial = inspect_formal_source_anti_tamper(workspace, config)
    anchored = inspect_formal_source_anti_tamper(
        workspace,
        config,
        expected_source_manifest_sha256=initial["source_manifest_sha256"],
    )

    assert anchored["schema"] == FORMAL_SOURCE_ANTI_TAMPER_SCHEMA
    assert anchored["policy"] == FORMAL_SOURCE_ANTI_TAMPER_POLICY
    assert anchored["rules_sha256"] == FORMAL_SOURCE_ANTI_TAMPER_RULES_SHA256
    assert anchored["verdict"] == "PASS"
    assert anchored["expected_source_manifest_sha256"] == anchored[
        "source_manifest_sha256"
    ]
    assert anchored["files"] == [
        {
            "path": "source/candidate.py",
            "sha256": anchored["files"][0]["sha256"],
            "size_bytes": anchored["files"][0]["size_bytes"],
            "language": "python",
            "status": "PASS",
            "violations": [],
        }
    ]


@pytest.mark.parametrize(
    ("source", "expected_rule"),
    [
        ("import torch\ntorch.allclose = lambda *a: True\n", "protected_attribute_mutation"),
        ("import torch\ntorch.equal = lambda *a: True\n", "protected_attribute_mutation"),
        ("import torch\ntorch.isfinite = lambda *a: True\n", "protected_attribute_mutation"),
        ("import torch\ntorch.cuda.Event = object\n", "protected_attribute_mutation"),
        ("import torch\ntorch.set_default_dtype(torch.float64)\n", "protected_state_mutating_call"),
        ("import torch\nchange = torch.set_default_dtype\nchange(torch.float64)\n", "protected_state_mutating_call"),
        ("import torch\ngetattr(torch, 'equal')\n", "dynamic_protected_namespace_access"),
        ("import torch\ntorch.serialization.os.system('true')\n", "forbidden_namespace_bridge"),
        ("import torch as t\nT = t.Tensor\nT.data_ptr = lambda self: 0\n", "protected_attribute_mutation"),
        ("import __main__\n__main__.checker = lambda *a: True\n", "non_allowlisted_import"),
        ("def f():\n    return 1\nf.__globals__['checker'] = None\n", "dunder_namespace_access"),
        ("exec('import sys; sys.modules.clear()')\n", "forbidden_dynamic_namespace_call"),
        ("eval('1 + 1')\n", "forbidden_dynamic_namespace_call"),
        ("compile('x = 1', '<x>', 'exec')\n", "forbidden_dynamic_namespace_call"),
        ("m = __import__('sys')\n", "forbidden_dynamic_namespace_call"),
        (
            "import subprocess\n"
            "subprocess.run(['python3', '-c', "
            "'p=\"source/candidate.py\"; b=open(p,\"rb\").read(); "
            "open(p,\"wb\").write(b+bytes([10])); open(p,\"wb\").write(b)'])\n",
            "non_allowlisted_import",
        ),
        ("import socket\n", "non_allowlisted_import"),
        ("import multiprocessing\n", "non_allowlisted_import"),
        ("import inspect\n", "non_allowlisted_import"),
        ("setattr(torch, 'equal', lambda *a: True)\n", "forbidden_dynamic_namespace_call"),
        ("delattr(torch, 'equal')\n", "forbidden_dynamic_namespace_call"),
        ("globals()['checker'] = None\n", "forbidden_dynamic_namespace_call"),
        ("getattr(lambda: None, '__globals__')\n", "dynamic_dunder_access"),
    ],
)
def test_static_guard_rejects_evaluator_mutation_paths(
    tmp_path: Path,
    source: str,
    expected_rule: str,
) -> None:
    workspace, config = _workspace(tmp_path, source)
    report = inspect_formal_source_anti_tamper(workspace, config)

    assert report["verdict"] == "FAIL"
    assert expected_rule in _rules(report)


def test_source_change_invalidates_anchored_report(tmp_path: Path) -> None:
    workspace, config = _workspace(tmp_path, "import torch\n")
    initial = inspect_formal_source_anti_tamper(workspace, config)
    (workspace / "source/candidate.py").write_text(
        "import torch\nVALUE = 2\n", encoding="utf-8"
    )

    report = inspect_formal_source_anti_tamper(
        workspace,
        config,
        expected_source_manifest_sha256=initial["source_manifest_sha256"],
    )

    assert report["verdict"] == "FAIL"
    assert "source_manifest_mismatch" in _rules(report)


def test_symlink_and_hardlink_sources_fail_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("import torch\n", encoding="utf-8")

    symlink_workspace = tmp_path / "symlink-workspace"
    (symlink_workspace / "source").mkdir(parents=True)
    (symlink_workspace / "source/candidate.py").symlink_to(outside)
    symlink_report = inspect_formal_source_anti_tamper(
        symlink_workspace, {"source_file_path": ["source/candidate.py"]}
    )

    hardlink_workspace = tmp_path / "hardlink-workspace"
    (hardlink_workspace / "source").mkdir(parents=True)
    os.link(outside, hardlink_workspace / "source/candidate.py")
    hardlink_report = inspect_formal_source_anti_tamper(
        hardlink_workspace, {"source_file_path": ["source/candidate.py"]}
    )

    assert symlink_report["verdict"] == "FAIL"
    assert hardlink_report["verdict"] == "FAIL"
    assert _rules(symlink_report) & {"unsafe_source_file", "unsafe_source_path"}
    assert "unsafe_source_file" in _rules(hardlink_report)


def test_formal_evaluator_persists_final_anchored_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, config = _workspace(tmp_path, "import torch\n")
    config.update(
        {
            "compile_command": ["compile"],
            "correctness_command": ["correctness"],
        }
    )
    monkeypatch.setattr(evaluator, "force_jit_rebuild", lambda *args: {})
    monkeypatch.setattr(
        evaluator, "run_command", lambda *args, **kwargs: (True, "PASS", "")
    )
    monkeypatch.setattr(evaluator, "measure_performance", lambda *args, **kwargs: [])

    result = evaluator.evaluate_kernel(
        workspace,
        config,
        [],
        source_anti_tamper_required=True,
    )

    report = result["source_anti_tamper"]
    assert report["verdict"] == "PASS"
    assert report["expected_source_manifest_sha256"] == report[
        "source_manifest_sha256"
    ]


def test_nonformal_evaluator_result_shape_remains_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, config = _workspace(tmp_path, "import torch\n")
    config.update(
        {
            "compile_command": ["compile"],
            "correctness_command": ["correctness"],
        }
    )
    monkeypatch.setattr(evaluator, "force_jit_rebuild", lambda *args: {})
    monkeypatch.setattr(
        evaluator, "run_command", lambda *args, **kwargs: (True, "PASS", "")
    )
    monkeypatch.setattr(evaluator, "measure_performance", lambda *args, **kwargs: [])

    result = evaluator.evaluate_kernel(workspace, config, [])

    assert "source_anti_tamper" not in result


def test_formal_evaluator_detects_child_rewrite_and_restore_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, config = _workspace(tmp_path, "import torch\n")
    config.update(
        {
            "compile_command": ["compile"],
            "correctness_command": ["correctness"],
        }
    )
    source_path = workspace / "source/candidate.py"
    monkeypatch.setattr(evaluator, "force_jit_rebuild", lambda *args: {})

    def mutate_source(*args, **kwargs):
        source_path.write_text("import torch\nVALUE = 9\n", encoding="utf-8")
        return True, "PASS", ""

    monkeypatch.setattr(evaluator, "run_command", mutate_source)

    result = evaluator.evaluate_kernel(
        workspace,
        config,
        [],
        source_anti_tamper_required=True,
    )

    assert result["pass_compilation"] is False
    assert result["source_anti_tamper"]["verdict"] == "FAIL"
    assert "source_manifest_mismatch" in _rules(result["source_anti_tamper"])
