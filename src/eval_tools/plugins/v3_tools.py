"""Schema-v3 paths of the original sanitizer plugins.

The original plugins take a task-declared adapter (an argv, a code object, a
replay capsule). A v3 task declares none, so on that path:

* Triton FpSan and GPU ASan keep their own environment and attestation rules
  and receive the workspace runner's ``sanitize`` lane as their command.
* Waitcheck and the rocJITsu race detector run the image-owned
  ``aka_v3_entrypoint`` that captures the role's dispatched kernels on the GPU
  and analyses each one; the verdict aggregates one parse per kernel/capsule.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ..adapters.aka_v3_capsules import load_manifest
from ..adapters.replay_capsule import CapsuleValidationError, ReplayCapsule
from ..contracts import (
    CapabilityCheck,
    CapabilityState,
    KernelLanguage,
    ToolCapability,
    ToolContext,
    ToolInvocation,
)
from . import aka_v3
from .base import (
    FINDING,
    INCONCLUSIVE,
    PASS,
    TOOL_ERROR,
    ParseResult,
    blocked_check,
    parsed_to_run_result,
    ready_check,
    sidecar_path,
)
from .parsers import parse_rocjitsu, parse_waitcheck, rocjitsu_dispatch_kernels

V3_ENTRYPOINT = "/opt/aka-eval-tools/src/eval_tools/adapters/aka_v3_entrypoint.py"
SUPPORTED_ARCH = "gfx950"


def _engine(context: ToolContext, tool: str, ready_languages: set[str], reasons: dict[str, str]) -> CapabilityCheck:
    language = aka_v3.role_language(context)
    if language in ready_languages:
        if context.gpu_arch and context.gpu_arch != SUPPORTED_ARCH:
            return blocked_check(CapabilityState.UNSUPPORTED, f"{tool}_gfx950_not_selected",
                                 f"{tool} is qualified only for {SUPPORTED_ARCH}.",
                                 observed_gpu_arch=context.gpu_arch)
        return ready_check(adapter=aka_v3.ADAPTER, language=language)
    if language in reasons:
        return blocked_check(CapabilityState.UNSUPPORTED, f"{tool}_{language}_unsupported", reasons[language])
    return blocked_check(CapabilityState.NOT_APPLICABLE, f"{tool}_language_not_applicable",
                         f"{tool} does not apply to {language} kernels.")


def capability(plugin, context: ToolContext, runtime: CapabilityCheck, engine: CapabilityCheck) -> ToolCapability:
    adapter = ready_check(adapter=aka_v3.ADAPTER)
    return ToolCapability(plugin.name, engine, adapter, runtime)


def _with_command(context: ToolContext, key: str, lane: str) -> ToolContext:
    invocation = aka_v3.lane_invocation("_", context, lane)
    return replace(context, options={**dict(context.options), key: list(invocation.command)})


def _triton_profile(context: ToolContext) -> ToolContext:
    """Gluon compiles through Triton; its ASan and FpSan rules are Triton's."""
    language = aka_v3.role_language(context)
    target = KernelLanguage.TRITON if language in ("triton", "gluon") else aka_v3.language_enum(language)
    return replace(context, profile=replace(context.profile, language=target))


# ---------------------------------------------------------------- triton_fpsan


def fpsan_engine(context: ToolContext) -> CapabilityCheck:
    languages = {aka_v3.target_language(context), aka_v3.baseline_language(context)}
    if not languages <= {"triton", "gluon"}:
        return blocked_check(CapabilityState.NOT_APPLICABLE, "triton_fpsan_needs_triton_pair",
                             "FpSan compares the candidate with a Triton or Gluon baseline; "
                             f"this task pairs {sorted(languages)}.")
    return ready_check(adapter=aka_v3.ADAPTER, instrumentation_mode="fpsan")


def fpsan_invocation(plugin, context: ToolContext) -> ToolInvocation:
    if aka_v3.role(context) != "candidate":
        raise ValueError("FpSan is a candidate-versus-baseline comparison")
    invocation = plugin._legacy_build_invocation(_with_command(_triton_profile(context), "comparison_command", "fpsan"))
    return replace(invocation, cwd=context.workspace,
                   env={**dict(invocation.env), "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"})


# -------------------------------------------------------------------- gpu_asan


def asan_engine(context: ToolContext) -> CapabilityCheck:
    return _engine(context, "gpu_asan", {"hip", "triton", "gluon"},
                   {"flydsl": "No FlyDSL release inserts AMD GPU ASan instrumentation."})


def asan_invocation(plugin, context: ToolContext) -> ToolInvocation:
    language = aka_v3.role_language(context)
    if language not in ("hip", "triton", "gluon"):
        raise ValueError(f"GPU ASan does not instrument {language}")
    invocation = plugin._legacy_build_invocation(_with_command(_triton_profile(context), "command", "gpu_asan"))
    env = {**dict(invocation.env), **aka_v3.NO_CACHING, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    if language == "hip":
        arch = (context.gpu_arch or SUPPORTED_ARCH).split(":", 1)[0]
        env.update({"AKA_HIP_SANITIZER_CFLAGS": "-fsanitize=address -shared-libsan",
                    "AKA_HIP_OFFLOAD_ARCH": f"{arch}:xnack+"})
    return replace(invocation, cwd=context.workspace, env=env)


def asan_parse(plugin, context: ToolContext, execution):
    return plugin._legacy_parse(_triton_profile(context), execution)


# ------------------------------------------------------ waitcheck and rocjitsu


def waitcheck_engine(context: ToolContext) -> CapabilityCheck:
    return _engine(context, "rocjitsu_waitcheck", {"triton", "gluon", "hip", "flydsl"}, {})


def rocjitsu_engine(context: ToolContext) -> CapabilityCheck:
    return _engine(context, "rocjitsu", {"triton", "gluon"}, {
        "hip": "HIP candidates have no automatic replay capture: a launch needs its lowered ABI and memory.",
        "flydsl": "FlyDSL launches expose no lowered ABI to capture, so no replay capsule can be built.",
    })


def _output_dir(context: ToolContext, tool: str) -> Path:
    return Path(context.artifact_dir) / f"v3-{tool}"


def entrypoint_invocation(plugin, context: ToolContext, tool: str, extra: list[str]) -> ToolInvocation:
    output = _output_dir(context, tool)
    output.mkdir(parents=True, exist_ok=True)
    command = (aka_v3.SIDECAR_PYTHON, "-I", V3_ENTRYPOINT, tool, "--workspace", context.workspace,
               "--role", aka_v3.role(context), "--output-dir", str(output), "--target", SUPPORTED_ARCH,
               "--task-id", str(context.task_config.get("task_id") or context.task_config.get("name") or ""),
               # Bound each step inside the tool's own deadline.
               "--capture-timeout", "600", "--kernel-timeout", "600", *extra)
    return ToolInvocation(tool=plugin.name, command=command, cwd=context.workspace,
                          env={**dict(context.env), "AKA_EVAL_TOOL": plugin.name},
                          timeout_s=int(context.options.get("timeout_s", 1800)),
                          artifact_dir=context.artifact_dir,
                          metadata={"adapter": aka_v3.ADAPTER, "role": aka_v3.role(context)})


def waitcheck_invocation(plugin, context: ToolContext) -> ToolInvocation:
    binary = sidecar_path(context, "waitcheck_binary", required=True)
    wrapper = sidecar_path(context, "waitcheck_capi_wrapper", required=True)
    return entrypoint_invocation(plugin, context, "waitcheck",
                                 ["--waitcheck", str(binary), "--capi-wrapper", str(wrapper)])


def rocjitsu_invocation(plugin, context: ToolContext) -> ToolInvocation:
    binary = sidecar_path(context, "rocjitsu_binary", required=True)
    config = sidecar_path(context, "config_path", required=True)
    return entrypoint_invocation(plugin, context, "rocjitsu",
                                 ["--rocjitsu", str(binary), "--config", str(config)])


def _load_index(context: ToolContext, tool: str) -> tuple[dict | None, dict | None, ParseResult | None]:
    output = _output_dir(context, tool)
    try:
        index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, None, ParseResult(TOOL_ERROR, reason_code=f"{tool}_v3_index_missing", details=str(error))
    if index.get("error"):
        return index, None, ParseResult(INCONCLUSIVE, reason_code=f"{tool}_v3_capture_failed",
                                        details=str(index["error"])[:4000])
    try:
        manifest = load_manifest(output / "capture")
    except (OSError, ValueError, CapsuleValidationError) as error:
        return index, None, ParseResult(TOOL_ERROR, reason_code=f"{tool}_v3_capture_invalid", details=str(error))
    return index, manifest, None


def aggregate(tool: str, parsed: list[ParseResult], unverified: list[str]) -> ParseResult:
    findings = tuple(item for result in parsed if result.status == FINDING for item in result.findings)
    if findings:
        return ParseResult(FINDING, findings, f"{tool}_v3_finding", attested=True)
    problems = [result.reason_code or result.status for result in parsed if result.status != PASS] + unverified
    if not parsed:
        return ParseResult(INCONCLUSIVE, reason_code=f"{tool}_v3_nothing_analyzed",
                           details="; ".join(unverified)[:4000] or "no dispatched kernel was captured")
    if problems:
        return ParseResult(INCONCLUSIVE, reason_code=f"{tool}_v3_partial",
                           details="; ".join(str(item) for item in problems)[:4000], attested=True)
    return ParseResult(PASS, reason_code=f"{tool}_v3_clean", attested=True)


def _metadata(index: dict | None, parsed: list[ParseResult]) -> dict:
    return {"adapter": aka_v3.ADAPTER, "analyzed": len(parsed),
            "skipped": list((index or {}).get("skipped") or [])[:8],
            "language": (index or {}).get("language")}


def waitcheck_parse(plugin, context: ToolContext, execution):
    if execution.timed_out:
        return parsed_to_run_result(plugin.name, ParseResult(TOOL_ERROR, reason_code="waitcheck_v3_timeout"),
                                    execution)
    index, manifest, problem = _load_index(context, "waitcheck")
    if problem is not None:
        return parsed_to_run_result(plugin.name, problem, execution, metadata=_metadata(index, []))
    output = _output_dir(context, "waitcheck")
    captured = {item["code_object"]: item["sha256"] for item in manifest.get("code_objects", [])}
    parsed, unverified = [], list(index.get("skipped") or [])
    for entry in index.get("entries", []):
        if "error" in entry:
            unverified.append(f"{entry.get('code_object')}: {entry['error']}")
            continue
        if captured.get(entry["code_object"]) != entry["code_object_sha256"]:
            unverified.append(f"{entry['code_object']}: not the captured code object")
            continue
        log = (output / entry["log"]).read_text(encoding="utf-8", errors="replace")
        parsed.append(parse_waitcheck(log, "", entry["returncode"], expected_sha256=entry["code_object_sha256"],
                                      expected_target=SUPPORTED_ARCH, expected_kernel=entry["kernel_name"],
                                      expected_entry=int(entry["kernel_entry"]),
                                      expected_discovered=int(entry.get("kernels_in_code_object", 1))))
    return parsed_to_run_result(plugin.name, aggregate("waitcheck", parsed, unverified), execution,
                                metadata=_metadata(index, parsed))


def rocjitsu_parse(plugin, context: ToolContext, execution):
    if execution.timed_out:
        return parsed_to_run_result(plugin.name, ParseResult(TOOL_ERROR, reason_code="rocjitsu_v3_timeout"),
                                    execution)
    index, manifest, problem = _load_index(context, "rocjitsu")
    if problem is not None:
        return parsed_to_run_result(plugin.name, problem, execution, metadata=_metadata(index, []))
    output = _output_dir(context, "rocjitsu")
    captured = {item["directory"]: item["hsaco_sha256"] for item in manifest.get("launches", [])}
    parsed, unverified = [], list(index.get("skipped") or [])
    for entry in index.get("entries", []):
        if "error" in entry:
            unverified.append(f"{entry.get('directory')}: {entry['error']}")
            continue
        try:
            capsule = ReplayCapsule.load(output / entry["capsule"], verify_files=True)
        except (OSError, ValueError, CapsuleValidationError) as error:
            unverified.append(f"{entry['directory']}: capsule revalidation failed: {error}")
            continue
        if captured.get(entry["directory"]) != capsule.code_object.sha256:
            unverified.append(f"{entry['directory']}: capsule code object is not the captured one")
            continue
        log = (output / entry["log"]).read_text(encoding="utf-8", errors="replace")
        report = (output / entry["race_log"]).read_text(encoding="utf-8", errors="replace") \
            if entry.get("race_log") else ""
        combined = "\n".join((log, report))
        kernel = capsule.code_object.kernel_name
        attested = (kernel in rocjitsu_dispatch_kernels(combined) and "AKA_REPLAY_RESULT pass" in combined
                    and f"AKA_REPLAY_CAPSULE sha256={entry['capsule_sha256']} code_sha256={capsule.code_object.sha256} "
                        f"adapter=triton_aot arch={SUPPORTED_ARCH} kernel={kernel}" in combined)
        result = parse_rocjitsu(log, "", entry["returncode"], attested=attested, report_text=report)
        if result.status == FINDING and not attested:
            # A replay that did not reproduce the GPU's output (or whose dispatch
            # and capsule were not attested) is not the launch the GPU ran, so
            # its race reports say nothing definite about the role's kernel.
            result = ParseResult(INCONCLUSIVE, reason_code="rocjitsu_v3_race_on_unattested_replay",
                                 details="; ".join(f"{item.kind} {item.location}" for item in result.findings)[:4000])
        parsed.append(result)
    return parsed_to_run_result(plugin.name, aggregate("rocjitsu", parsed, unverified), execution,
                                metadata=_metadata(index, parsed))
