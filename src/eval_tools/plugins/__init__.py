"""Sanitizer tool plugins."""

from .attestation import BuildAttestation, attest_artifact, sha256_file
from ..contracts import (
    CapabilityCheck,
    ExecutionRecord,
    Finding,
    ToolCapability,
    ToolInvocation,
    ToolRunResult,
)
from .base import (
    ADAPTER_REQUIRED,
    FINDING,
    INCONCLUSIVE,
    NOT_APPLICABLE,
    PASS,
    READY,
    TOOL_ERROR,
    UNAVAILABLE_RUNTIME,
    UNSUPPORTED,
    FindingRecord,
    ParseResult,
    execute_invocation,
)
from .gpu_asan import GPU_ASAN_ENV, GpuAsanPlugin, hip_asan_build_flags
from .hip_fpsan import HipFpSanPlugin
from .consan import ConSanPlugin
from .registry import get_plugin, iter_plugins, plugin_ids, register_builtin_plugins
from .rocjitsu import RocJitsuPlugin
from .triton_fpsan import TritonFpSanPlugin
from .waitcheck import WaitcheckPlugin

__all__ = [
    "ADAPTER_REQUIRED",
    "BuildAttestation",
    "CapabilityCheck",
    "ExecutionRecord",
    "FINDING",
    "FindingRecord",
    "GPU_ASAN_ENV",
    "GpuAsanPlugin",
    "HipFpSanPlugin",
    "ConSanPlugin",
    "INCONCLUSIVE",
    "NOT_APPLICABLE",
    "PASS",
    "ParseResult",
    "ToolCapability",
    "ToolRunResult",
    "READY",
    "RocJitsuPlugin",
    "TOOL_ERROR",
    "ToolInvocation",
    "TritonFpSanPlugin",
    "WaitcheckPlugin",
    "UNAVAILABLE_RUNTIME",
    "UNSUPPORTED",
    "attest_artifact",
    "execute_invocation",
    "get_plugin",
    "hip_asan_build_flags",
    "iter_plugins",
    "plugin_ids",
    "register_builtin_plugins",
    "sha256_file",
]
