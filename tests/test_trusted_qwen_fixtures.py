"""Original fixture namespaces and byte verification survive Qwen staging."""
import hashlib
import json
from pathlib import Path

import pytest

from src.tools import trusted_qwen_eval as adapter


@pytest.mark.parametrize("fault", [None, "corrupt", "unapproved_remote"])
def test_original_registry_is_taken_from_git_and_only_verified_body_is_installed(tmp_path, monkeypatch, fault):
    name = "qwen3.8-2.4t__fused_moe_2stage_mxfp4"
    relative = name + "/ut/reference_io.pt"
    body = b"original captured fixture bytes"
    digest = hashlib.sha256(body).hexdigest()
    registry = {"schema_version": 1, "artifacts": [{"task": name, "path": relative,
                 "size_bytes": len(body), "sha256": digest}],
                "oci_storage": {"remote_root": "oci:bucket/approved-original"}}
    calls = []
    def git(command):
        calls.append(command)
        return json.dumps(registry).encode()
    monkeypatch.setattr(adapter.subprocess, "check_output", git)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: pytest.fail("unapproved network access"))
    monkeypatch.setattr(adapter, "materialize_perf_helpers_in_workspace", lambda *a: [])
    task, output, mirror = tmp_path / "task", tmp_path / "output", tmp_path / "mirror"
    (task / "ut").mkdir(parents=True); output.mkdir()
    (task / "ut/meta.json").write_text(json.dumps({"reference_io_sha256": digest}))
    original = mirror / relative
    original.parent.mkdir(parents=True); original.write_bytes(body if fault != "corrupt" else b"wrong")
    contract = {"config": {"benchmark_quality": {"adapter": adapter.PROFILES[name]}}}
    kwargs = dict(local_mirror=None if fault == "unapproved_remote" else mirror,
                  oci_prefix="oci:bucket/unapproved", max_files=1, max_bytes=1024, timeout=10)
    arguments = (tmp_path, "a" * 40, "tasks/headkernel/" + name, task, output, contract)
    if fault:
        with pytest.raises((ValueError, OSError)):
            adapter.prepare_fixtures(*arguments, **kwargs)
        assert not (task / "ut/reference_io.pt").exists()
    else:
        receipt = adapter.prepare_fixtures(*arguments, **kwargs)
        assert (task / "ut/reference_io.pt").read_bytes() == body
        assert (task / "ut/reference_io.pt").stat().st_ino != original.stat().st_ino
        proof = json.loads((output / receipt["file"]).read_text())
        assert proof["artifacts"] == [{"path": "ut/reference_io.pt", "bytes": len(body), "sha256": digest}]
        assert proof["generated_replacement_inputs"] is False
    assert calls == [["git", "-C", str(tmp_path), "show", "a" * 40 + ":tools/headkernel-artifacts.json"]]
