import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.eval_tools.adapters import (
    CapsuleValidationError,
    NativeLauncherContract,
    ReplayCapsule,
    build_triton_abi,
    pack_dynamic_layout,
    parse_flydsl_static_launch,
)
from src.eval_tools.adapters import rocjitsu_replay


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capsule_fixture(tmp_path: Path):
    hsaco = b"\x7fELF" + b"\x00" * 60
    before = bytes(range(16))
    expected = before
    (tmp_path / "kernel.hsaco").write_bytes(hsaco)
    (tmp_path / "input.bin").write_bytes(before)
    (tmp_path / "expected.bin").write_bytes(expected)
    raw = {
        "schema_version": 1,
        "producer": {
            "adapter": "test",
            "adapter_version": "1",
            "framework_version": "1",
            "image_digest": "sha256:image",
            "rocm_version": "7.2",
        },
        "target": {"gpu_arch": "gfx950", "xnack": False, "code_object_version": 5},
        "case": {"task_id": "task", "case_id": "case0", "seed": 1, "candidate_scope": "module.kernel"},
        "code_object": {"path": "kernel.hsaco", "sha256": sha(hsaco), "kernel_name": "kernel"},
        "launch": {"grid": [1, 1, 1], "block": [64, 1, 1], "dynamic_smem_bytes": 0},
        "abi": [
            {"index": 0, "name": "out", "kind": "pointer", "c_type": "pointer", "size": 8, "ref": "buf"},
            {"index": 1, "name": "n", "kind": "scalar", "c_type": "i32", "size": 4, "value": 4},
            {"index": 2, "name": "global_scratch", "kind": "implicit", "c_type": "pointer", "size": 8, "ref": "scratch:global"},
            {"index": 3, "name": "profile_scratch", "kind": "implicit", "c_type": "pointer", "size": 8, "ref": "scratch:profile"},
        ],
        "allocations": [
            {
                "id": "buf",
                "byte_size": len(before),
                "before_blob": "input.bin",
                "before_sha256": sha(before),
                "expected_blob": "expected.bin",
                "expected_sha256": sha(expected),
                "alignment": 16,
            }
        ],
        "views": [{"arg_index": 0, "allocation_id": "buf", "byte_offset": 0, "dtype": "int32", "shape": [4], "stride": [1]}],
        "relocations": [],
        "scratch": {"global_bytes": 0, "profile_bytes": 0, "profile_alignment": 1},
        "dispatch_count": 1,
    }
    path = tmp_path / "capsule.json"
    path.write_text(json.dumps(raw))
    return path, raw


def test_capsule_validates_files_alias_view_and_full_abi(tmp_path):
    path, _ = capsule_fixture(tmp_path)
    capsule = ReplayCapsule.load(path)
    assert capsule.code_object.kernel_name == "kernel"
    assert capsule.views[0].allocation_id == "buf"
    assert capsule.abi[-2].name == "global_scratch"


def test_capsule_rejects_hash_mismatch(tmp_path):
    path, _ = capsule_fixture(tmp_path)
    (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELFbroken")
    with pytest.raises(CapsuleValidationError, match="SHA-256"):
        ReplayCapsule.load(path)


def test_capsule_rejects_path_escape_and_descriptor(tmp_path):
    _, raw = capsule_fixture(tmp_path)
    raw["code_object"]["path"] = "../kernel.hsaco"
    with pytest.raises(CapsuleValidationError, match="relative"):
        ReplayCapsule.from_dict(raw, base_dir=tmp_path)
    _, raw = capsule_fixture(tmp_path)
    raw["abi"][0]["kind"] = "descriptor"
    with pytest.raises(CapsuleValidationError, match="descriptor"):
        ReplayCapsule.from_dict(raw, base_dir=tmp_path)


def test_native_launcher_contract_contains_exact_launch_and_hidden_args(tmp_path):
    path, _ = capsule_fixture(tmp_path)
    source = NativeLauncherContract().render(ReplayCapsule.load(path))
    assert "hipModuleLoad" in source
    assert "hipModuleGetFunction" in source
    assert "function, 1, 1, 1," in source
    assert "64, 1, 1," in source
    assert "void* arg_2 = global_scratch" in source
    assert "void* arg_3 = profile_scratch" in source
    assert "AKA_REPLAY_RESULT pass" in source
    assert "hipGetDeviceProperties" in source
    assert "device_properties.maxGridSize" in source
    assert "device_properties.maxThreadsDim" in source
    assert "device_properties.sharedMemPerBlock" in source
    assert "int main(int argc, char** argv)" in source
    assert str(tmp_path) not in source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grid", [1.9, 1, 1]),
        ("grid", [True, 1, 1]),
        ("grid", [1 << 32, 1, 1]),
        ("block", ["64", 1, 1]),
        ("dynamic_smem_bytes", 1.5),
        ("dynamic_smem_bytes", 1 << 32),
    ],
)
def test_capsule_rejects_lossy_or_out_of_range_launch_geometry(
    tmp_path, field, value
):
    _path, raw = capsule_fixture(tmp_path)
    raw["launch"][field] = value
    with pytest.raises(CapsuleValidationError, match="exact JSON integer|must be in"):
        ReplayCapsule.from_dict(raw, base_dir=tmp_path)


def test_capsule_requires_at_least_one_golden_output(tmp_path):
    _path, raw = capsule_fixture(tmp_path)
    raw["allocations"][0].pop("expected_blob")
    raw["allocations"][0].pop("expected_sha256")
    with pytest.raises(CapsuleValidationError, match="golden expected output"):
        ReplayCapsule.from_dict(raw, base_dir=tmp_path)


def test_native_launcher_plan_passes_capsule_root_at_runtime(tmp_path):
    path, _ = capsule_fixture(tmp_path)
    plan = NativeLauncherContract().materialize(path, tmp_path / "build")

    assert plan.run_command == (str(plan.binary_path), str(tmp_path))
    assert "kernel.hsaco" in plan.source_path.read_text(encoding="utf-8")


def test_trusted_rocjitsu_replay_compiles_and_runs_generated_launcher(
    tmp_path, monkeypatch, capsys
):
    path, raw = capsule_fixture(tmp_path)
    raw["producer"]["adapter"] = "triton_aot"
    path.write_text(json.dumps(raw), encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rocjitsu_replay.subprocess, "run", fake_run)
    capsule_sha256 = rocjitsu_replay.sha256_file(path)
    rc = rocjitsu_replay.execute_replay(
        capsule_path=path,
        output_dir=tmp_path / "replay",
        rocjitsu=Path("/usr/local/bin/rocjitsu"),
        config=Path("/opt/rocjitsu/gfx950.json"),
        hipcc=Path("/opt/rocm/bin/hipcc"),
        expected_adapter="triton_aot",
        expected_arch="gfx950",
        expected_kernel="kernel",
        expected_capsule_sha256=capsule_sha256,
    )

    assert rc == 0
    assert len(calls) == 2
    assert calls[0][0][0] == "/opt/rocm/bin/hipcc"
    assert calls[1][0][:4] == (
        "/usr/local/bin/rocjitsu",
        "--config",
        "/opt/rocjitsu/gfx950.json",
        "--",
    )
    assert calls[1][0][-1] == str(tmp_path)
    assert all(call_kwargs == {"check": False} for _, call_kwargs in calls)
    assert f"AKA_REPLAY_CAPSULE sha256={capsule_sha256}" in capsys.readouterr().out


def test_trusted_rocjitsu_replay_rejects_changed_capsule_before_compilation(
    tmp_path, monkeypatch
):
    path, raw = capsule_fixture(tmp_path)
    raw["producer"]["adapter"] = "triton_aot"
    path.write_text(json.dumps(raw), encoding="utf-8")
    original_sha256 = rocjitsu_replay.sha256_file(path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("subprocess must not run for a changed capsule")

    monkeypatch.setattr(rocjitsu_replay.subprocess, "run", unexpected_run)
    with pytest.raises(CapsuleValidationError, match="changed"):
        rocjitsu_replay.execute_replay(
            capsule_path=path,
            output_dir=tmp_path / "replay",
            rocjitsu=Path("/usr/local/bin/rocjitsu"),
            config=Path("/opt/rocjitsu/gfx950.json"),
            hipcc=Path("/opt/rocm/bin/hipcc"),
            expected_adapter="triton_aot",
            expected_arch="gfx950",
            expected_kernel="kernel",
            expected_capsule_sha256=original_sha256,
        )


def test_trusted_rocjitsu_replay_rejects_unknown_adapter_version(tmp_path):
    path, raw = capsule_fixture(tmp_path)
    raw["producer"]["adapter"] = "triton_aot"
    raw["producer"]["adapter_version"] = "future"
    path.write_text(json.dumps(raw), encoding="utf-8")
    capsule = ReplayCapsule.load(path)
    with pytest.raises(CapsuleValidationError, match="version is not supported"):
        rocjitsu_replay.validate_replay_identity(
            capsule,
            expected_adapter="triton_aot",
            expected_arch="gfx950",
            expected_kernel="kernel",
        )


def test_isolated_replay_entrypoint_cannot_be_shadowed_by_candidate_workspace(
    tmp_path,
):
    marker = tmp_path / "candidate-helper-imported"
    malicious = tmp_path / "src" / "eval_tools" / "adapters"
    malicious.mkdir(parents=True)
    for package in (tmp_path / "src", tmp_path / "src" / "eval_tools", malicious):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (malicious / "rocjitsu_replay.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('shadowed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    entrypoint = Path(rocjitsu_replay.__file__).with_name(
        "rocjitsu_replay_entrypoint.py"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-I", str(entrypoint), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Trusted sidecar entry point" in completed.stdout
    assert not marker.exists()


def test_triton_abi_always_appends_hidden_scratch_arguments():
    abi = build_triton_abi(
        {0: "*fp32", 1: "i32", 2: "constexpr"},
        constants={(2,): 256},
        pointer_bindings={0: ("out", 0)},
        scalar_values={1: 32},
    )
    assert [arg.name for arg in abi] == ["arg0", "arg1", "global_scratch", "profile_scratch"]


def test_flydsl_dynamic_layout_matches_no_padding_contract():
    packed = pack_dynamic_layout(
        [4, 8], [8, 1], dynamic_shape_indices=[0, 1], dynamic_stride_indices=[0], use_32bit_stride=False
    )
    assert len(packed) == 16  # i32 + i32 + i64, with no native padding


def test_flydsl_static_launch_parser_rejects_multi_dispatch():
    ir = '''
      %one = arith.constant 1 : index
      %threads = arith.constant 128 : index
      %smem = arith.constant 512 : i32
      gpu.launch_func @kernels::@race blocks in (%one, %one, %one)
          threads in (%threads, %one, %one) dynamic_shared_memory_size %smem
    '''
    parsed = parse_flydsl_static_launch(ir)
    assert parsed.kernel_name == "race"
    assert parsed.launch.block == (128, 1, 1)
    assert parsed.launch.dynamic_smem_bytes == 512
    with pytest.raises(CapsuleValidationError, match="requires one"):
        parse_flydsl_static_launch(ir + ir)
