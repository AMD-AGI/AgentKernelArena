"""Protected GLM device-source boundary and independent process overlays."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import shutil


def _contract(ut_dir):
    return json.loads((Path(ut_dir) / "device_source_contract.json").read_text())


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _host_contract(source, editable):
    tree = ast.parse(source)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in editable:
            found.add(node.name)
            # Freeze imports, decorators, signatures and launchers. Only the
            # declared Triton function bodies can change.
            node.body = [ast.Pass()]
    if found != set(editable):
        raise RuntimeError("declared GLM device functions are missing")
    return ast.dump(tree, include_attributes=False)


def validate_candidate(ut_dir):
    ut = Path(ut_dir).resolve()
    contract = _contract(ut)
    for entry in contract["frozen_files"]:
        path = ut / entry["file"]
        if not path.is_file() or _digest(path) != entry["sha256"]:
            raise RuntimeError(f"frozen GLM source changed: {entry['file']}")
    source = ut.parent / contract["candidate_source"]
    reference = ut / contract["device_reference"]
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("GLM candidate must be a regular task source file")
    expected = _host_contract(reference.read_text(), contract["editable_functions"])
    if _host_contract(source.read_text(), contract["editable_functions"]) != expected:
        raise RuntimeError("GLM host contract changed; edit only the declared device function bodies")
    return source


def build_candidate_overlay(ut_dir):
    ut = Path(ut_dir).resolve()
    source = validate_candidate(ut)
    contract = _contract(ut)
    baseline, candidate = ut / "baseline_overlay", ut / "_cand_overlay"
    manifest = json.loads((baseline / "_overlay_manifest.json").read_text())
    expected = {contract["device_module"], contract["dispatcher_module"]}
    if {entry["module"] for entry in manifest["modules"]} != expected:
        raise RuntimeError("GLM baseline must bind both frozen device and dispatcher modules")
    shutil.rmtree(candidate, ignore_errors=True)
    shutil.copytree(baseline, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    device_entry = next(e for e in manifest["modules"] if e["module"] == contract["device_module"])
    shutil.copyfile(source, candidate / device_entry["file"])
    return str(baseline), str(candidate)


def device_identity(ut_dir):
    contract = _contract(ut_dir)
    module = importlib.import_module(contract["device_module"])
    path = Path(module.__file__).resolve()
    kernel = getattr(module, "fused_moe_kernel")
    function = getattr(kernel, "fn", kernel)
    if getattr(function, "__module__", None) != contract["device_module"]:
        raise RuntimeError("GLM device function resolves outside its selected module")
    return {"module": contract["device_module"], "file": str(path), "sha256": _digest(path)}


def verify_leg_identities(ut_dir, baseline, candidate, baseline_identity, candidate_identity):
    ut = Path(ut_dir).resolve()
    contract = _contract(ut)
    source = validate_candidate(ut)
    for overlay, identity, expected_hash in (
        (baseline, baseline_identity, _digest(ut / contract["device_reference"])),
        (candidate, candidate_identity, _digest(source)),
    ):
        manifest = json.loads((Path(overlay) / "_overlay_manifest.json").read_text())
        modules = {entry["module"]: entry["file"] for entry in manifest["modules"]}
        expected_device = (Path(overlay) / modules[contract["device_module"]]).resolve()
        expected_dispatcher = (Path(overlay) / modules[contract["dispatcher_module"]]).resolve()
        device = identity.get("device_source", {})
        if (device.get("file") != str(expected_device)
                or device.get("sha256") != expected_hash
                or identity.get("file") != str(expected_dispatcher)):
            raise RuntimeError("GLM worker did not bind its independent frozen/candidate device source")
    if baseline_identity["device_source"]["file"] == candidate_identity["device_source"]["file"]:
        raise RuntimeError("GLM baseline and candidate share a device module")
