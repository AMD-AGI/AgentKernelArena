"""CPU dependency-contract regressions; no candidate execution or GPU qualification.

The three fixture files are exact archived candidates. They are intentionally
read as data, never imported: using an AITER FlyDSL operator was the violation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_protocol import parse_command_result

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
FIXTURES = Path(__file__).parent / "fixtures/flydsl_operator_dependencies"
AFFECTED = [TASKS / "flydsl2flydsl" / name for name in (
    "blockscale_preshuffle_gemm_kernel", "flash_attn_func_kernel",
    "fp8_gemm_4wave_kernel", "fp8_gemm_8wave_kernel", "fused_rope_cache_kernel",
    "hgemm_splitk_kernel", "layernorm_kernel", "moe_sorting_kernel",
    "pa_decode_fp8_kernel", "pa_decode_swa_kernel", "preshuffle_gemm_v2_kernel",
    "rmsnorm_kernel", "silu_and_mul_fq_kernel", "softmax_kernel",
    "topk_gating_softmax_kernel",
)] + [TASKS / "torch2flydsl" / name for name in (
    "batched_gemm_bf16_kernel", "hgemm_kernel",
)] + [TASKS / "triton2flydsl" / name for name in (
    "aiter/gemm_a16w16", "aiter/layernorm", "aiter/softmax",
    "sglang/decode_attention", "sglang/sglang_fused_moe",
)]
ALL_RUNTIMES = sorted(p for family in ("flydsl2flydsl", "torch2flydsl", "triton2flydsl")
                      for p in (TASKS / family).rglob("task_runtime.py"))
ARCHIVED = (
    ("aiter_gemm_delegate.py.txt", "triton2flydsl/aiter/gemm_a16w16",
     "2dbee73e7817f09dae8035bd75d7c3d9ceba7152781f3d0876bff4abb1d6ec9c"),
    ("batched_gemm_delegate.py.txt", "torch2flydsl/batched_gemm_bf16_kernel",
     "e434c2346ece4660aaa2afa022cf110e4a1158ba33c26196d9f8dfb02472ad13"),
    ("sglang_moe_delegate.py.txt", "triton2flydsl/sglang/sglang_fused_moe",
     "716cfdd84076dda7ca1d40ef8cbea5dbb2ddd1a6c527cc93b2abaead8ae01da7"),
)
PA = TASKS / "flydsl2flydsl/pa_decode_fp8_kernel"
DTYPE = TASKS / "torch2flydsl/qk_norm_rope_quant_kernel/task_runtime.py"


def runtime(path):
    spec = importlib.util.spec_from_file_location("dependency_test_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(task, role, action, *, initial=False):
    env = os.environ.copy()
    env.pop("ARENA_EVAL_PHASE", None)
    if initial:
        env["ARENA_EVAL_PHASE"] = "task_validation"
    process = subprocess.run(
        [sys.executable, "scripts/evaluate.py", role, action], cwd=task,
        capture_output=True, text=True, env=env, timeout=30,
    )
    return parse_command_result(process.stdout, role=role, action=action,
                                returncode=process.returncode)


@pytest.mark.parametrize("fixture,task_id,digest", ARCHIVED)
@pytest.mark.parametrize("action", ["compile", "correctness", "performance"])
def test_exact_archived_delegations_fail_before_import(fixture, task_id, digest, action, tmp_path):
    raw = (FIXTURES / fixture).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    task = tmp_path / "task"
    shutil.copytree(TASKS / task_id, task)
    runner = runtime(task / "task_runtime.py")
    (task / runner.candidate_relative_path()).write_bytes(raw)
    result = invoke(task, "candidate", action)
    assert not result.passed
    assert "Final operator must execute FlyDSL, not aiter" in result.reason
    assert "numerical_mismatch" not in result.reason
    assert all(row.get("failure_kind") != "numerical_mismatch" for row in result.cases)


@pytest.mark.parametrize("path", ALL_RUNTIMES, ids=lambda p: str(p.parent.relative_to(TASKS)))
def test_all_dependency_variants_reject_three_real_delegations(path):
    # Includes the 89 already-strict runtimes, which this change does not edit.
    checker = runtime(path)
    for fixture, _, _ in ARCHIVED:
        with pytest.raises(ValueError, match="Final operator must execute FlyDSL, not aiter"):
            checker.check_dependencies([FIXTURES / fixture], final_language=True)


@pytest.mark.parametrize("source", [
    "import aiter", "import aiter as provider",
    "import aiter.ops.flydsl.gemm_kernels as kernels",
    "from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm",
    "from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm as launch",
    "from aiter.ops.flydsl import gemm_kernels as kernels",
    "from aiter import ops as operators", "from aiter import *",
    "from . import aiter as provider", "from aiter import fused_moe as launch",
    "import ctypes as native", "from subprocess import run as launch",
    "from importlib import import_module as load; op = load('aiter')",
    "import builtins; load = builtins.__import__; op = load('aiter')",
    "load = __import__; op = load('aiter')",
    "import torch; launch = torch.ops.aiter.some_operator",
    "from torch import ops as operators; launch = operators.aiter.some_operator",
    "from torch.utils import cpp_extension as compiler",
    "from scripts import task_reference", "from scripts import task_baseline",
])
def test_equivalent_alias_and_native_routes_rejected(source, tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("import flydsl\n" + source + "\n")
    with pytest.raises(ValueError):
        runtime(AFFECTED[0] / "task_runtime.py").check_dependencies([path])


def test_multifile_candidate_cannot_hide_delegation_in_helper(tmp_path):
    main = tmp_path / "kernel.py"
    helper = tmp_path / "launch_helpers.py"
    main.write_text("import flydsl\nfrom launch_helpers import launch\n")
    helper.write_text("from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm as launch\n")
    with pytest.raises(ValueError, match="not aiter"):
        runtime(AFFECTED[0] / "task_runtime.py").check_dependencies([main, helper])


@pytest.mark.parametrize("task", AFFECTED, ids=lambda p: str(p.relative_to(TASKS)))
def test_initial_implementation_or_provided_baseline_preserved(task):
    checker = runtime(task / "task_runtime.py")
    cfg = checker.config()
    result = invoke(task, "baseline", "compile")
    assert result.passed, result.reason
    if cfg["candidate"]["initial_state"] == "unimplemented":
        result = invoke(task, "candidate", "compile", initial=True)
        assert not result.passed and "no baseline fallback" in result.reason
    else:
        checker.check_dependencies(checker.candidate_files(),
                                   final_language=cfg["candidate"]["initial_language"] == "flydsl")
        result = invoke(task, "candidate", "compile", initial=True)
        assert result.passed, result.reason
        if cfg["candidate"]["initial_language"] == "triton":
            result = invoke(task, "candidate", "compile")
            assert not result.passed and "not triton" in result.reason


def test_allocations_layout_and_bundled_helpers_stay_available(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text("""import math
import functools
import torch
import flydsl
from flydsl import expr as fx
from kernels import dpp_utils
def launch(x):
    out = torch.empty_like(x)
    temporary = torch.empty((32,), device=x.device, dtype=x.dtype)
    return out.view(x.shape), temporary, math.ceil(x.numel() / 32)
""")
    for task in AFFECTED:
        runtime(task / "task_runtime.py").check_dependencies([path])


@pytest.fixture
def pa_checker(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(PA / "scripts/dependency_policy.json", tmp_path / "scripts/dependency_policy.json")
    checker = runtime(PA / "task_runtime.py")
    checker.ROOT = tmp_path
    return checker


@pytest.mark.parametrize("helper,owner", [
    ("get_pa_metadata_info_v1", "get_pa_metadata"),
    ("get_pa_metadata_v1", "get_pa_metadata"),
    ("pa_reduce_v1", "pa_decode_ps_launch"),
])
@pytest.mark.parametrize("alias", [False, True])
def test_precise_pa_support_calls_allow_import_aliases(pa_checker, helper, owner, alias):
    binding = "support" if alias else helper
    imported = f"{helper} as support" if alias else helper
    path = pa_checker.ROOT / "kernel.py"
    path.write_text(f"import flydsl\ndef {owner}(*args):\n"
                    f"    from aiter.ops.attention import {imported}\n"
                    f"    return {binding}(*args)\n")
    pa_checker.check_dependencies([path])


@pytest.mark.parametrize("body", [
    "import aiter.ops.attention as support\nreturn support.pa_reduce_v1(*args)",
    "from aiter.ops.attention import *\nreturn pa_reduce_v1(*args)",
    "from .aiter.ops.attention import pa_reduce_v1\nreturn pa_reduce_v1(*args)",
    "from aiter.ops.attention import pa_reduce_v1, other_operator\nreturn pa_reduce_v1(*args)",
    "from aiter.ops.attention import pa_reduce_v1\nreturn pa_reduce_v1",
    "from aiter.ops.attention import pa_reduce_v1\nreturn getattr(pa_reduce_v1, '__globals__')",
    "from aiter.ops.attention import pa_reduce_v1\nother = pa_reduce_v1\nreturn other(*args)",
    "from aiter.ops.attention import pa_reduce_v1\nreturn wrapper(pa_reduce_v1, *args)",
    "global pa_reduce_v1\nfrom aiter.ops.attention import pa_reduce_v1\nreturn pa_reduce_v1(*args)",
    "from aiter.ops.attention import pa_reduce_v1\nreturn lambda: pa_reduce_v1(*args)",
    "from aiter.ops.attention import pa_reduce_v1\ndef escape():\n    return pa_reduce_v1(*args)\nreturn escape()",
])
def test_pa_exception_cannot_expose_namespace_or_callable(pa_checker, body):
    path = pa_checker.ROOT / "kernel.py"
    path.write_text("import flydsl\ndef pa_decode_ps_launch(*args):\n    " + body.replace("\n", "\n    ") + "\n")
    with pytest.raises(ValueError):
        pa_checker.check_dependencies([path])


@pytest.mark.parametrize("file,owner", [("other.py", "pa_decode_ps_launch"), ("kernel.py", "other_launch")])
def test_pa_exception_does_not_transfer_to_other_file_or_function(pa_checker, file, owner):
    path = pa_checker.ROOT / file
    path.write_text(f"import flydsl\ndef {owner}(*args):\n"
                    "    from aiter.ops.attention import pa_reduce_v1\n"
                    "    return pa_reduce_v1(*args)\n")
    with pytest.raises(ValueError, match="not aiter"):
        pa_checker.check_dependencies([path])


@pytest.mark.parametrize("mutation", ["edit", "delete"])
def test_task_helper_policy_is_protected_from_candidate_edit(tmp_path, mutation):
    task = tmp_path / "task"
    shutil.copytree(PA, task)
    snapshot = snapshot_workspace_harness(task, task_root=PA)
    policy = task / "scripts/dependency_policy.json"
    assert "scripts/dependency_policy.json" in snapshot.digests
    if mutation == "delete":
        policy.unlink()
    else:
        value = json.loads(policy.read_text())
        value["helpers"].append({"symbol": "aiter.ops.flydsl.gemm_kernels.flydsl_hgemm",
                                 "file": "kernel.py", "function": "pa_decode_ps_launch"})
        policy.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("policy", [
    {"version": 2, "helpers": []}, {"version": True, "helpers": []},
    {"version": 1, "helpers": "aiter.*"},
    {"version": 1, "helpers": [{"symbol": "aiter.*", "file": "kernel.py", "function": "launch"}]},
    {"version": 1, "helpers": [{"symbol": "aiter.helper", "file": "../kernel.py", "function": "launch"}]},
])
def test_malformed_or_escaping_policy_fails_closed(pa_checker, policy):
    (pa_checker.ROOT / "scripts/dependency_policy.json").write_text(json.dumps(policy))
    path = pa_checker.ROOT / "kernel.py"
    path.write_text("import flydsl\n")
    with pytest.raises(ValueError):
        pa_checker.check_dependencies([path])


def test_policy_symlink_cannot_escape_task_workspace(pa_checker, tmp_path):
    outside = tmp_path.parent / "outside-policy.json"
    outside.write_text('{"version":1,"helpers":[]}')
    policy = pa_checker.ROOT / "scripts/dependency_policy.json"
    policy.unlink()
    policy.symlink_to(outside)
    path = pa_checker.ROOT / "kernel.py"
    path.write_text("import flydsl\n")
    with pytest.raises(ValueError, match="escapes task workspace"):
        pa_checker.check_dependencies([path])


@pytest.mark.parametrize("statement,accepted", [
    ("from aiter.utility import dtypes; dtype = dtypes.fp8", True),
    ("from aiter.utility import dtypes as constants; dtype = constants.fp8", True),
    ("from aiter.utility import dtypes; leaked = dtypes", False),
    ("from aiter.utility import dtypes; dtype = dtypes.other_operator", False),
    ("from aiter.utility import dtypes; dtype = getattr(dtypes, 'fp8')", False),
    ("import aiter.utility.dtypes as dtypes; dtype = dtypes.fp8", False),
])
def test_existing_separate_dtype_constant_exception_unchanged(tmp_path, statement, accepted):
    path = tmp_path / "kernel.py"
    path.write_text("import flydsl\n" + statement + "\n")
    checker = runtime(DTYPE)
    if accepted:
        checker.check_dependencies([path])
    else:
        with pytest.raises(ValueError):
            checker.check_dependencies([path])
