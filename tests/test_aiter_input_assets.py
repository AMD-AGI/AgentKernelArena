"""Path/binding checks; real read-only mount coverage needs the operator fixture.

Unit acceptance tests explicitly simulate statvfs's mount flag. The integration test
requires AKA_TEST_READONLY_INPUT_ASSET to name a real externally bind-mounted
read-only file; file permissions alone do not satisfy that check. No Torch or
GPU is used. The task tensor loaders separately hash the opened file contents.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TASKS = Path(__file__).resolve().parents[1] / "tasks/aiter_workloads"


@pytest.fixture(params=["fused_moe", "mhc"], ids=["moe", "mhc"])
def resolver(request, monkeypatch):
    # MoE imports only contract validation at module load; tensor/AITER imports
    # stay inside its loader functions and are not part of these path checks.
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(validate_contract=lambda _: None))
    path = TASKS / f"deepseek_v4_pro_tp8_ep1_c64_8k1k_{request.param}/scripts/cases.py"
    spec = importlib.util.spec_from_file_location(f"_input_assets_{request.param}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(task, asset, *, logical="inputs/call.pt"):
    descriptor = {"path": logical, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
    entry = {"path": str(asset), "sha256": descriptor["sha256"], "size_bytes": asset.stat().st_size}
    _write_assets(task, {logical: entry})
    return descriptor, entry


def _write_assets(task, assets):
    (task / "workloads.json").write_text(json.dumps({"runtime_binding": {"input_assets": assets}}))


@pytest.fixture
def bound_asset(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    asset = tmp_path / "captured.pt"
    asset.write_bytes(b"captured input bytes for path checks\n")
    descriptor, entry = _binding(task, asset)
    return task, asset, descriptor, entry


@pytest.mark.parametrize("workload", [None, {"runtime_binding": None}, {"runtime_binding": {"input_assets": {}}}])
def test_local_inputs_keep_the_existing_path_and_bytes(tmp_path, resolver, workload):
    task = tmp_path / "task"
    task.mkdir()
    local = task / "local.pt"
    local.write_bytes(b"local captured input")
    if workload is not None:
        (task / "workloads.json").write_text(json.dumps(workload))
    assert resolver.resolve_input_path(task, {"path": "local.pt"}) == local
    assert local.read_bytes() == b"local captured input"


@pytest.mark.parametrize("logical", ["", ".", "./call.pt", "nested//call.pt", "nested/../call.pt", "../call.pt", "/call.pt", "inputs/"])
def test_logical_paths_cannot_escape_or_use_noncanonical_spelling(tmp_path, resolver, logical):
    with pytest.raises(ValueError, match="canonical task-relative"):
        resolver.resolve_input_path(tmp_path, {"path": logical})


def test_local_symlink_may_resolve_inside_task_but_cannot_escape(tmp_path, resolver):
    task = tmp_path / "task"
    task.mkdir()
    local = task / "data.pt"
    local.write_bytes(b"inside")
    alias = task / "alias.pt"
    alias.symlink_to(local)
    assert resolver.resolve_input_path(task, {"path": "alias.pt"}) == local
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    alias.unlink()
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the task"):
        resolver.resolve_input_path(task, {"path": "alias.pt"})


def test_declared_external_asset_with_simulated_readonly_mount_flag(bound_asset, resolver, monkeypatch):
    task, asset, descriptor, _ = bound_asset
    before = asset.read_bytes()
    monkeypatch.setattr(resolver.os, "statvfs", lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    assert resolver.resolve_input_path(task, descriptor) == asset
    assert asset.read_bytes() == before
    assert not (task / descriptor["path"]).exists()


@pytest.mark.parametrize("fault", ["different_digest", "malformed_digest", "wrong_size", "negative_size", "boolean_size", "extra_field"])
def test_external_asset_binding_must_match_case_and_file(bound_asset, resolver, monkeypatch, fault):
    task, _, descriptor, entry = bound_asset
    if fault == "different_digest":
        entry["sha256"] = "0" * 64
    elif fault == "malformed_digest":
        entry["sha256"] = descriptor["sha256"] = "X" * 64
    elif fault == "wrong_size":
        entry["size_bytes"] += 1
    elif fault == "negative_size":
        entry["size_bytes"] = -1
    elif fault == "boolean_size":
        entry["size_bytes"] = True
    else:
        entry["fallback"] = True
    _write_assets(task, {descriptor["path"]: entry})
    monkeypatch.setattr(resolver.os, "statvfs", lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    with pytest.raises(ValueError, match="digest|size|requires path"):
        resolver.resolve_input_path(task, descriptor)


@pytest.mark.parametrize("fault", ["relative", "noncanonical", "inside_task", "symlink", "directory", "missing"])
def test_external_asset_must_be_an_existing_canonical_outside_file(bound_asset, resolver, monkeypatch, fault):
    task, asset, descriptor, entry = bound_asset
    if fault == "relative":
        entry["path"] = "captured.pt"
    elif fault == "noncanonical":
        entry["path"] = str(asset.parent) + "/./" + asset.name
    elif fault == "inside_task":
        inside = task / "other.pt"
        inside.write_bytes(asset.read_bytes())
        entry["path"] = str(inside)
    elif fault == "symlink":
        alias = asset.parent / "alias.pt"
        alias.symlink_to(asset)
        entry["path"] = str(alias)
    elif fault == "directory":
        directory = asset.parent / "directory"
        directory.mkdir()
        entry.update(path=str(directory), size_bytes=directory.stat().st_size)
    else:
        asset.unlink()
    _write_assets(task, {descriptor["path"]: entry})
    monkeypatch.setattr(resolver.os, "statvfs", lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolver.resolve_input_path(task, descriptor)


def test_file_permissions_do_not_replace_readonly_mount_requirement(bound_asset, resolver):
    task, asset, descriptor, _ = bound_asset
    asset.chmod(0o444)
    assert not os.statvfs(asset).f_flag & os.ST_RDONLY
    with pytest.raises(ValueError, match="read-only mount"):
        resolver.resolve_input_path(task, descriptor)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_declared_external_asset_rejects_any_local_shadow(bound_asset, resolver, monkeypatch, kind):
    task, _, descriptor, _ = bound_asset
    local = task / descriptor["path"]
    local.parent.mkdir()
    if kind == "file":
        local.write_bytes(b"shadow")
    elif kind == "directory":
        local.mkdir()
    else:
        local.symlink_to(task / "missing.pt")
    monkeypatch.setattr(resolver.os, "statvfs", lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    with pytest.raises(ValueError, match="local shadow"):
        resolver.resolve_input_path(task, descriptor)


def test_missing_undeclared_asset_has_no_implicit_search(bound_asset, resolver):
    task, asset, _, _ = bound_asset
    # A matching basename outside the task is irrelevant without its own key.
    with pytest.raises(FileNotFoundError):
        resolver.resolve_input_path(task, {"path": asset.name})


def test_runtime_input_assets_must_be_a_mapping(bound_asset, resolver):
    task, _, descriptor, _ = bound_asset
    _write_assets(task, [])
    with pytest.raises(ValueError, match="input_assets must be a mapping"):
        resolver.resolve_input_path(task, descriptor)


def test_operator_readonly_bind_mount_accepts_declared_asset(tmp_path, resolver):
    configured = os.environ.get("AKA_TEST_READONLY_INPUT_ASSET")
    if not configured:
        pytest.skip("operator integration: supply an actual read-only bind-mounted file")
    asset = Path(configured)
    assert asset.is_absolute() and asset.resolve(strict=True) == asset
    assert os.statvfs(asset).f_flag & os.ST_RDONLY, "integration fixture must use a real read-only mount"
    task = tmp_path / "task"
    task.mkdir()
    descriptor, _ = _binding(task, asset)
    assert resolver.resolve_input_path(task, descriptor) == asset
