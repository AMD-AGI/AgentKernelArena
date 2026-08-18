import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.eval_tools.contracts import (
    ArtifactKind,
    CapabilityCheck,
    CapabilityState,
    ExecutionRecord,
    FindingStatus,
    InstrumentationControl,
    KernelLanguage,
    TaskProfile,
    ToolCapability,
    ToolContext,
    ToolInvocation,
)
from src.eval_tools.plugins import get_plugin, plugin_ids, register_builtin_plugins
from src.eval_tools.registry import ToolRegistry


def profile(
    language: KernelLanguage,
    *,
    framework: str,
    artifact: ArtifactKind = ArtifactKind.PYTHON_JIT,
    control: InstrumentationControl = InstrumentationControl.COMPILER_CONTROLLED,
    adapter=None,
    evidence=None,
):
    return TaskProfile(
        task_type=f"test_{language.value}",
        language=language,
        artifact_kind=artifact,
        framework=framework,
        instrumentation_control=control,
        adapter=adapter,
        source_available=True,
        evidence=evidence or {},
    )


def context(tmp_path: Path, task_profile: TaskProfile, options=None):
    return ToolContext(
        workspace=str(tmp_path),
        task_config={},
        profile=task_profile,
        artifact_dir=str(tmp_path / "artifacts"),
        gpu_arch="gfx950",
        options=options or {},
    )


def runtime(**evidence):
    return CapabilityCheck.ready(**evidence)


def replay_capsule(tmp_path: Path, adapter: str, *, arch: str = "gfx950") -> Path:
    hsaco = b"\x7fELF" + b"\x00" * 60
    before = b"\x00" * 16
    expected = before
    (tmp_path / "kernel.hsaco").write_bytes(hsaco)
    (tmp_path / "input.bin").write_bytes(before)
    (tmp_path / "expected.bin").write_bytes(expected)
    digest = lambda value: hashlib.sha256(value).hexdigest()
    raw = {
        "schema_version": 1,
        "producer": {
            "adapter": adapter,
            "adapter_version": "1",
            "framework_version": "test",
            "image_digest": "sha256:test",
            "rocm_version": "7.2",
        },
        "target": {"gpu_arch": arch, "xnack": False, "code_object_version": 5},
        "case": {
            "task_id": "task",
            "case_id": "case0",
            "seed": 1,
            "candidate_scope": "module.kernel",
        },
        "code_object": {
            "path": "kernel.hsaco",
            "sha256": digest(hsaco),
            "kernel_name": "kernel",
        },
        "launch": {
            "grid": [1, 1, 1],
            "block": [64, 1, 1],
            "dynamic_smem_bytes": 0,
        },
        "abi": [
            {
                "index": 0,
                "name": "out",
                "kind": "pointer",
                "c_type": "pointer",
                "size": 8,
                "ref": "buf",
            }
        ],
        "allocations": [
            {
                "id": "buf",
                "byte_size": len(before),
                "before_blob": "input.bin",
                "before_sha256": digest(before),
                "expected_blob": "expected.bin",
                "expected_sha256": digest(expected),
            }
        ],
        "dispatch_count": 1,
    }
    path = tmp_path / "capsule.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_registry_uses_core_plugin_contracts(tmp_path):
    assert plugin_ids() == (
        "gpu_asan",
        "hip_fpsan",
        "rocjitsu",
        "rocjitsu_consan",
        "rocjitsu_waitcheck",
        "triton_fpsan",
    )
    plugin = get_plugin("triton-fpsan")
    capability = plugin.assess(
        context(tmp_path, profile(KernelLanguage.TRITON, framework="triton"), {"command": ["true"]}),
        runtime(triton_fpsan=True),
    )
    assert isinstance(capability, ToolCapability)
    assert capability.ready


def test_builtins_register_in_core_registry():
    requested = ToolRegistry()
    registry = register_builtin_plugins(requested)
    assert registry is requested
    assert tuple(registry) == (
        "gpu_asan",
        "hip_fpsan",
        "rocjitsu",
        "rocjitsu_consan",
        "rocjitsu_waitcheck",
        "triton_fpsan",
    )


def test_waitcheck_binds_code_object_kernel_and_entry(tmp_path):
    code_object = tmp_path / "kernel.hsaco"
    code_object.write_bytes(b"\x7fELFwaitcheck")
    ctx = context(
        tmp_path,
        profile(
            KernelLanguage.HIP,
            framework="standalone",
            artifact=ArtifactKind.HSACO_PRECOMPILED,
            adapter="waitcheck_code_object",
        ),
        {
            "code_object": code_object.name,
            "expected_kernel": "optimized_kernel",
            "kernel_entry": "0x40",
            "waitcheck_binary": "/opt/rocjitsu/bin/rj_waitcheck",
            "waitcheck_capi_wrapper": "/opt/rocjitsu/bin/aka-waitcheck-capi",
        },
    )
    plugin = get_plugin("rocjitsu_waitcheck")
    capability = plugin.assess(ctx, runtime(target_arch="gfx950"))
    invocation = plugin.build_invocation(ctx)

    assert capability.ready
    assert capability.adapter.evidence["code_object_sha256"] == hashlib.sha256(
        code_object.read_bytes()
    ).hexdigest()
    assert "--kernel-entry" in invocation.command
    assert invocation.metadata["kernel_entry"] == 0x40


def test_consan_requires_exact_hsaco_launcher_and_oracle(tmp_path):
    code_object = tmp_path / "kernel.hsaco"
    code_object.write_bytes(b"\x7fELFconsan")
    task_profile = profile(
        KernelLanguage.HIP,
        framework="standalone",
        artifact=ArtifactKind.HSACO_PRECOMPILED,
        adapter="consan_native",
    )
    missing_oracle = context(
        tmp_path,
        task_profile,
        {"code_object": code_object.name, "command": ["launcher", code_object.name]},
    )
    plugin = get_plugin("rocjitsu_consan")
    assert (
        plugin.assess(missing_oracle, runtime(gpu_arch="gfx950")).effective.state
        == CapabilityState.ADAPTER_REQUIRED
    )

    ctx = context(
        tmp_path,
        task_profile,
        {
            "code_object": code_object.name,
            "command": ["launcher", code_object.name],
            "oracle_command": ["oracle", code_object.name],
            "consan_hook": "/opt/rocjitsu/lib/librocjitsu_dbi_hooks.so",
        },
    )
    capability = plugin.assess(ctx, runtime(gpu_arch="gfx950"))
    invocation = plugin.build_invocation(ctx)

    assert capability.ready
    assert capability.adapter.evidence["code_object_fingerprint"].startswith(
        "fnv1a64:"
    )
    assert invocation.metadata["policy"] == "strict"
    assert invocation.metadata["mode"] == "record-replay"
    assert invocation.command.count("--command-arg") == 2
    assert invocation.command.count("--oracle-arg") == 2


def test_flydsl_gpu_asan_is_fail_closed(tmp_path):
    capability = get_plugin("gpu_asan").assess(
        context(tmp_path, profile(KernelLanguage.FLYDSL, framework="flydsl"), {"command": ["true"]}),
        runtime(gpu_asan=True, xnack_supported=True),
    )
    assert capability.effective.state == CapabilityState.UNSUPPORTED
    assert capability.effective.reason_code == "gpu_asan_flydsl_no_device_instrumentation"


def test_precompiled_aiter_gpu_asan_is_fail_closed(tmp_path):
    capability = get_plugin("gpu_asan").assess(
        context(
            tmp_path,
            profile(
                KernelLanguage.HIP,
                framework="aiter",
                artifact=ArtifactKind.HSACO_PRECOMPILED,
                control=InstrumentationControl.NONE,
            ),
            {"command": ["true"]},
        ),
        runtime(gpu_asan=True, xnack_supported=True),
    )
    assert capability.effective.state == CapabilityState.UNSUPPORTED
    assert capability.effective.reason_code == "gpu_asan_precompiled_code_object"


@pytest.mark.parametrize("framework", ["rocblas", "rccl"])
def test_library_kernel_gpu_asan_stays_unsupported_after_source_rebuild(
    tmp_path, framework
):
    capability = get_plugin("gpu_asan").assess(
        context(
            tmp_path,
            profile(
                KernelLanguage.HIP,
                framework=framework,
                artifact=ArtifactKind.HSACO_PRECOMPILED,
                control=InstrumentationControl.RECOMPILE,
                evidence={"rebuilt_from_source": True},
            ),
            {"command": ["true"]},
        ),
        runtime(gpu_asan=True, xnack_supported=True),
    )
    assert capability.effective.state == CapabilityState.UNSUPPORTED
    assert capability.effective.reason_code == "gpu_asan_library_kernel_out_of_scope"


@pytest.mark.parametrize("language", [KernelLanguage.TRITON, KernelLanguage.FLYDSL])
def test_rocjitsu_python_jit_requires_aot_adapter(tmp_path, language):
    capability = get_plugin("rocjitsu").assess(
        context(tmp_path, profile(language, framework=language.value)),
        runtime(config_path="/configs/gfx950.json"),
    )
    assert capability.engine.state == CapabilityState.READY
    assert capability.effective.state == CapabilityState.ADAPTER_REQUIRED


def test_gpu_asan_invocation_uses_fresh_triton_cache_and_xnack(tmp_path):
    ctx = context(
        tmp_path,
        profile(KernelLanguage.TRITON, framework="triton"),
        {
            "command": ["python", "probe.py"],
            "timeout_s": 12,
            "host_asan_preload": "/opt/rocm/lib/clang/libclang_rt.asan-x86_64.so",
            "host_asan_lib_dir": "/opt/rocm/lib/clang",
            "hip_asan_runtime": "/opt/rocm/lib/asan/libamdhip64.so",
            "asan_runtime_dir": "/opt/rocm/lib/asan",
            "normal_rocm_lib_dir": "/opt/rocm/lib",
        },
    )
    invocation = get_plugin("gpu_asan").build_invocation(ctx)
    assert isinstance(invocation, ToolInvocation)
    assert invocation.command == ("python", "probe.py")
    assert invocation.env["TRITON_ENABLE_ASAN"] == "1"
    assert invocation.env["HSA_XNACK"] == "1"
    assert invocation.env["AMDGCN_USE_BUFFER_OPS"] == "0"
    assert invocation.env["LD_PRELOAD"].split(":")[:2] == [
        "/opt/rocm/lib/clang/libclang_rt.asan-x86_64.so",
        "/opt/rocm/lib/asan/libamdhip64.so",
    ]
    assert invocation.env["LD_LIBRARY_PATH"].split(":")[:2] == [
        "/opt/rocm/lib/clang",
        "/opt/rocm/lib/asan",
    ]
    assert invocation.env["LD_LIBRARY_PATH"].split(":")[2] == "/opt/rocm/lib"
    assert invocation.timeout_s == 12


@pytest.mark.parametrize("tool", ["gpu_asan", "triton_fpsan", "hip_fpsan"])
def test_configured_attestation_path_is_shared_by_invocation_and_parser(
    tmp_path, tool
):
    if tool == "gpu_asan":
        task_profile = profile(
            KernelLanguage.HIP,
            framework="standalone",
            artifact=ArtifactKind.SOURCE_AOT,
            control=InstrumentationControl.RECOMPILE,
        )
        options = {"command": ["true"]}
    elif tool == "triton_fpsan":
        task_profile = profile(KernelLanguage.TRITON, framework="triton")
        options = {"comparison_command": ["true"]}
    else:
        task_profile = profile(
            KernelLanguage.HIP,
            framework="standalone",
            artifact=ArtifactKind.SOURCE_AOT,
            control=InstrumentationControl.RECOMPILE,
            evidence={"fpsan_ported": True},
        )
        options = {
            "comparison_command": ["true"],
            "include_dir": "/opt/hip-fpsan/include",
        }
    options["attestation_path"] = "custom/build_attestation.json"
    ctx = context(tmp_path, task_profile, options)

    invocation = get_plugin(tool).build_invocation(ctx)
    expected = Path(ctx.artifact_dir) / "custom" / "build_attestation.json"

    assert invocation.env["AKA_BUILD_ATTESTATION_PATH"] == str(expected)
    assert invocation.metadata["attestation_path"] == str(expected)


def test_attestation_path_cannot_escape_invocation_artifacts(tmp_path):
    ctx = context(
        tmp_path,
        profile(
            KernelLanguage.HIP,
            framework="standalone",
            artifact=ArtifactKind.SOURCE_AOT,
            control=InstrumentationControl.RECOMPILE,
        ),
        {"command": ["true"], "attestation_path": "../stale.json"},
    )
    with pytest.raises(ValueError, match="artifact directory"):
        get_plugin("gpu_asan").build_invocation(ctx)


def test_hip_fpsan_requires_explicit_source_port(tmp_path):
    capability = get_plugin("hip_fpsan").assess(
        context(tmp_path, profile(KernelLanguage.HIP, framework="hip"), {"command": ["true"]}),
        runtime(headers=True),
    )
    assert capability.effective.state == CapabilityState.ADAPTER_REQUIRED
    assert capability.effective.reason_code == "hip_fpsan_source_port_required"


@pytest.mark.parametrize("language", [KernelLanguage.TRITON, KernelLanguage.FLYDSL])
def test_rocjitsu_aot_capsule_makes_generated_hsaco_replayable(tmp_path, language):
    expected_adapter = f"{language.value}_aot"
    capsule = replay_capsule(tmp_path, expected_adapter)
    capability = get_plugin("rocjitsu").assess(
        context(
            tmp_path,
            profile(language, framework=language.value, adapter=expected_adapter),
            {
                "capsule": capsule.name,
            },
        ),
        runtime(
            rocjitsu_binary="/usr/local/bin/rocjitsu",
            config_path="/opt/rocjitsu/gfx950.json",
        ),
    )

    assert capability.ready
    assert capability.adapter.evidence["adapter"] == expected_adapter


@pytest.mark.parametrize("language", [KernelLanguage.TRITON, KernelLanguage.FLYDSL])
def test_rocjitsu_aot_rejects_arbitrary_launcher(tmp_path, language):
    expected_adapter = f"{language.value}_aot"
    capsule = replay_capsule(tmp_path, expected_adapter)
    capability = get_plugin("rocjitsu").assess(
        context(
            tmp_path,
            profile(language, framework=language.value, adapter=expected_adapter),
            {"capsule": capsule.name, "launcher": ["./untrusted"]},
        ),
        runtime(config_path="/opt/rocjitsu/gfx950.json"),
    )

    assert capability.effective.state == CapabilityState.ADAPTER_REQUIRED
    assert "arbitrary launchers are not accepted" in capability.adapter.detail


@pytest.mark.parametrize(
    ("language", "producer_adapter"),
    [
        (KernelLanguage.TRITON, "flydsl_aot"),
        (KernelLanguage.FLYDSL, "triton_aot"),
    ],
)
def test_rocjitsu_aot_rejects_capsule_language_mismatch(
    tmp_path, language, producer_adapter
):
    expected_adapter = f"{language.value}_aot"
    capsule = replay_capsule(tmp_path, producer_adapter)
    capability = get_plugin("rocjitsu").assess(
        context(
            tmp_path,
            profile(language, framework=language.value, adapter=expected_adapter),
            {"capsule": capsule.name},
        ),
        runtime(config_path="/opt/rocjitsu/gfx950.json"),
    )

    assert capability.effective.state == CapabilityState.ADAPTER_REQUIRED
    assert "producer adapter does not match" in capability.adapter.detail


def test_rocjitsu_aot_rejects_non_language_specific_profile_adapter(tmp_path):
    capsule = replay_capsule(tmp_path, "triton_aot")
    capability = get_plugin("rocjitsu").assess(
        context(
            tmp_path,
            profile(
                KernelLanguage.TRITON,
                framework="triton",
                adapter="replay_capsule",
            ),
            {"capsule": capsule.name},
        ),
        runtime(config_path="/opt/rocjitsu/gfx950.json"),
    )

    assert capability.effective.state == CapabilityState.ADAPTER_REQUIRED
    assert "must exactly match" in capability.adapter.detail


def test_rocjitsu_aot_rejects_non_gfx950_context_and_capsule(tmp_path):
    capsule = replay_capsule(tmp_path, "triton_aot", arch="gfx942")
    ctx = replace(
        context(
            tmp_path,
            profile(
                KernelLanguage.TRITON,
                framework="triton",
                adapter="triton_aot",
            ),
            {"capsule": capsule.name},
        ),
        gpu_arch="gfx942",
    )
    capability = get_plugin("rocjitsu").assess(
        ctx,
        runtime(config_path="/opt/rocjitsu/gfx950.json"),
    )

    assert capability.engine.state == CapabilityState.UNSUPPORTED
    assert capability.engine.reason_code == "rocjitsu_aot_gfx950_only"
    assert capability.adapter.state == CapabilityState.ADAPTER_REQUIRED


def test_rocjitsu_aot_rejects_capsule_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capsule = replay_capsule(tmp_path, "triton_aot")
    capability = get_plugin("rocjitsu").assess(
        context(
            workspace,
            profile(KernelLanguage.TRITON, framework="triton", adapter="triton_aot"),
            {"capsule": str(capsule)},
        ),
        runtime(config_path="/opt/rocjitsu/gfx950.json"),
    )

    assert capability.effective.state == CapabilityState.ADAPTER_REQUIRED
    assert "below the selected task workspace" in capability.adapter.detail


def test_rocjitsu_aot_builds_only_trusted_sidecar_helper_invocation(tmp_path):
    capsule = replay_capsule(tmp_path, "triton_aot")
    ctx = context(
        tmp_path,
        profile(KernelLanguage.TRITON, framework="triton", adapter="triton_aot"),
        {
            "capsule": capsule.name,
            "rocjitsu_binary": "/usr/local/bin/rocjitsu",
            "config_path": "/opt/rocjitsu/gfx950.json",
        },
    )
    invocation = get_plugin("rocjitsu").build_invocation(ctx)

    assert invocation.command[:3] == (
        "/opt/venv/bin/python",
        "-I",
        "/opt/aka-eval-tools/src/eval_tools/adapters/rocjitsu_replay_entrypoint.py",
    )
    assert "./launcher" not in invocation.command
    assert invocation.shell is False
    assert invocation.metadata["expected_kernel"] == "kernel"
    assert invocation.metadata["code_object_sha256"] == hashlib.sha256(
        (tmp_path / "kernel.hsaco").read_bytes()
    ).hexdigest()


def _aot_output(capsule: Path, *, dispatch=True, replay_pass=True) -> str:
    raw = json.loads(capsule.read_text(encoding="utf-8"))
    capsule_sha256 = hashlib.sha256(capsule.read_bytes()).hexdigest()
    lines = [
        "AKA_REPLAY_CAPSULE "
        f"sha256={capsule_sha256} "
        f"code_sha256={raw['code_object']['sha256']} "
        f"adapter={raw['producer']['adapter']} arch=gfx950 kernel=kernel"
    ]
    if dispatch:
        lines.append('[rocjitsu] Kernel dispatch: "kernel"')
    if replay_pass:
        lines.append("AKA_REPLAY_RESULT pass")
    return "\n".join(lines)


def test_rocjitsu_aot_clean_requires_capsule_dispatch_and_replay_attestation(tmp_path):
    capsule = replay_capsule(tmp_path, "triton_aot")
    ctx = context(
        tmp_path,
        profile(KernelLanguage.TRITON, framework="triton", adapter="triton_aot"),
        {"capsule": capsule.name},
    )
    plugin = get_plugin("rocjitsu")

    clean = plugin.parse(
        ctx,
        ExecutionRecord(
            command=("trusted-helper",),
            returncode=0,
            stdout=_aot_output(capsule),
        ),
    )
    missing_result = plugin.parse(
        ctx,
        ExecutionRecord(
            command=("trusted-helper",),
            returncode=0,
            stdout=_aot_output(capsule, replay_pass=False),
        ),
    )
    missing_dispatch = plugin.parse(
        ctx,
        ExecutionRecord(
            command=("trusted-helper",),
            returncode=0,
            stdout=_aot_output(capsule, dispatch=False),
        ),
    )

    assert clean.finding == FindingStatus.CLEAN
    assert clean.metadata["capsule_attested"] is True
    assert clean.metadata["dispatch_attested"] is True
    assert clean.metadata["replay_result_attested"] is True
    assert missing_result.finding == FindingStatus.INCONCLUSIVE
    assert missing_result.metadata["replay_result_attested"] is False
    assert missing_dispatch.finding == FindingStatus.INCONCLUSIVE
    assert missing_dispatch.metadata["dispatch_attested"] is False


def test_rocjitsu_aot_does_not_duplicate_file_and_stderr_race(tmp_path):
    capsule = replay_capsule(tmp_path, "triton_aot")
    ctx = context(
        tmp_path,
        profile(KernelLanguage.TRITON, framework="triton", adapter="triton_aot"),
        {"capsule": capsule.name},
    )
    race = """[rocjitsu] Kernel dispatch: "kernel"
RACE type=LDS reg=12 wave=0 lane=1 wg=0,0,0 conflict=unknown
Race on LDS byte 12
END_RACE
"""
    report = tmp_path / "artifacts" / "rocjitsu-report" / "race.log"
    report.parent.mkdir(parents=True)
    report.write_text(race, encoding="utf-8")

    result = get_plugin("rocjitsu").parse(
        ctx,
        ExecutionRecord(
            command=("trusted-helper",),
            returncode=0,
            stdout=_aot_output(capsule),
            stderr=race,
        ),
    )

    assert result.finding == FindingStatus.FOUND
    assert result.findings_count == 1


def test_rocjitsu_current_stderr_race_is_not_hidden_by_clean_file(tmp_path):
    ctx = context(
        tmp_path,
        profile(
            KernelLanguage.HIP,
            framework="standalone",
            artifact=ArtifactKind.SOURCE_AOT,
        ),
        {
            "launcher": ["./hip-launcher"],
            "rocjitsu_binary": "/usr/local/bin/rocjitsu",
            "config_path": "/opt/rocjitsu/gfx950.json",
            "race_report": "custom/race.log",
            "expected_kernel": "hip_kernel",
        },
    )
    plugin = get_plugin("rocjitsu")
    invocation = plugin.build_invocation(ctx)
    report = Path(invocation.metadata["race_report"])
    assert report.read_text(encoding="utf-8") == ""
    report.write_text('[rocjitsu] Kernel dispatch: "hip_kernel"', encoding="utf-8")
    race = '''[rocjitsu] Kernel dispatch: "hip_kernel"
RACE type=LDS reg=8 wave=0 lane=1 wg=0,0,0 conflict=unknown
END_RACE
'''

    result = plugin.parse(
        ctx,
        ExecutionRecord(
            command=invocation.command,
            returncode=0,
            stderr=race,
        ),
    )

    assert result.finding == FindingStatus.FOUND
    assert result.findings_count == 1


def test_rocjitsu_hip_native_launcher_behavior_is_preserved(tmp_path):
    ctx = context(
        tmp_path,
        profile(
            KernelLanguage.HIP,
            framework="standalone",
            artifact=ArtifactKind.SOURCE_AOT,
        ),
        {
            "launcher": ["./hip-launcher", "--case", "safe"],
            "rocjitsu_binary": "/usr/local/bin/rocjitsu",
            "config_path": "/opt/rocjitsu/gfx950.json",
            "expected_kernel": "hip_kernel",
        },
    )
    plugin = get_plugin("rocjitsu")
    invocation = plugin.build_invocation(ctx)
    result = plugin.parse(
        ctx,
        ExecutionRecord(
            command=invocation.command,
            returncode=0,
            stdout='[rocjitsu] Kernel dispatch: "hip_kernel"',
        ),
    )

    assert invocation.command == (
        "/usr/local/bin/rocjitsu",
        "--config",
        "/opt/rocjitsu/gfx950.json",
        "--",
        "./hip-launcher",
        "--case",
        "safe",
    )
    assert result.finding == FindingStatus.CLEAN


def test_rocjitsu_aiter_python_runtime_is_fail_closed(tmp_path):
    capability = get_plugin("rocjitsu").assess(
        context(
            tmp_path,
            profile(KernelLanguage.TRITON, framework="aiter"),
            {"launcher": ["./launcher"], "capsule": "capsule.json"},
        ),
        runtime(
            rocjitsu_binary="/usr/local/bin/rocjitsu",
            config_path="/opt/rocjitsu/gfx950.json",
        ),
    )

    assert capability.effective.state == CapabilityState.UNSUPPORTED
    assert capability.effective.reason_code == "rocjitsu_framework_runtime_unsupported"


def test_hip_source_gpu_asan_is_ready_only_with_rebuild_command(tmp_path):
    task_profile = profile(
        KernelLanguage.HIP,
        framework="standalone",
        artifact=ArtifactKind.SOURCE_AOT,
        control=InstrumentationControl.RECOMPILE,
    )
    ready = get_plugin("gpu_asan").assess(
        context(tmp_path, task_profile, {"command": ["./build-and-run"]}),
        runtime(xnack_supported=True),
    )
    missing_adapter = get_plugin("gpu_asan").assess(
        context(tmp_path, task_profile), runtime(xnack_supported=True)
    )

    assert ready.ready
    assert missing_adapter.effective.state == CapabilityState.ADAPTER_REQUIRED


def test_explicitly_ported_hip_fpsan_source_is_ready(tmp_path):
    capability = get_plugin("hip_fpsan").assess(
        context(
            tmp_path,
            profile(
                KernelLanguage.HIP,
                framework="standalone",
                artifact=ArtifactKind.SOURCE_AOT,
                control=InstrumentationControl.RECOMPILE,
                evidence={"fpsan_ported": True},
            ),
            {"comparison_command": ["./compare"]},
        ),
        runtime(include_dir="/opt/hip-fpsan/include"),
    )

    assert capability.ready
