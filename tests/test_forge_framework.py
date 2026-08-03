"""Tests for optional source-owner metadata passed to forge-loop."""
from __future__ import annotations

from agents.forge.launch_agent import _resolve_framework


def test_explicit_aiter_source_owner_is_forwarded():
    cfg = {"kernel_identity": {"source_owner": "aiter"}}
    assert _resolve_framework(cfg) == "aiter"


def test_explicit_vllm_source_owner_is_forwarded():
    cfg = {"kernel_identity": {"source_owner": "vllm"}}
    assert _resolve_framework(cfg) == "vllm"


def test_explicit_sglang_source_owner_is_forwarded():
    cfg = {"kernel_identity": {"source_owner": "sglang"}}
    assert _resolve_framework(cfg) == "sglang"


def test_explicit_aiter_meta_alias_maps_to_aiter():
    cfg = {"kernel_identity": {"source_owner": "aiter_meta"}}
    assert _resolve_framework(cfg) == "aiter"


def test_legacy_explicit_source_owner_is_forwarded():
    assert _resolve_framework({"source_owner_framework": "AITER"}) == "aiter"


def test_absent_source_owner_is_omitted_even_when_path_is_inferable():
    cfg = {"image_repo_path": "/usr/local/lib/python3.12/dist-packages/vllm"}
    assert _resolve_framework(cfg) == ""
