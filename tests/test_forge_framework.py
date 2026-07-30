"""Tests for KB framework resolution passed to forge-loop (--framework).

Pure-Python: no GPU / network. The framework string forms the KB kernel-page
slug and MUST match what a Hyperloom forge-loop resolves for the same operator,
so a solution written by an arena long-run is found and reused downstream.
"""
from __future__ import annotations

from agents.forge.launch_agent import _resolve_framework


def test_framework_kernel_in_aiter_repo():
    # kernel physically lives in aiter, even when the serving stack is sglang.
    cfg = {"image_repo_path": "/sgl-workspace/aiter",
           "source_file_path": ["csrc/py_itfs_ck/mha.cu"]}
    assert _resolve_framework(cfg, "/ws/csrc/py_itfs_ck/mha.cu") == "aiter"


def test_framework_kernel_directly_in_vllm_deep_subdir():
    # Real flagship case: image_repo_path points DEEP into vllm; basename would be
    # 'fused_moe' (wrong). Must resolve the owning package 'vllm'.
    cfg = {"image_repo_path":
           "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe",
           "source_file_path": ["fused_moe.py"]}
    assert _resolve_framework(cfg, "/ws/fused_moe.py") == "vllm"


def test_framework_kernel_directly_in_sglang():
    cfg = {"image_repo_path": "/sgl-workspace/sglang/python/sglang",
           "source_file_path": ["kernels/ops/moe/mxfp8.py"]}
    assert _resolve_framework(cfg, "/ws/kernels/ops/moe/mxfp8.py") == "sglang"


def test_framework_aiter_meta_maps_to_aiter():
    cfg = {"image_repo_path": "/usr/local/lib/python3.12/dist-packages/aiter_meta",
           "source_file_path": ["csrc/ck_gemm_a8w8_blockscale/gemm.cu"]}
    assert _resolve_framework(cfg, "/ws/csrc/gemm.cu") == "aiter"


def test_framework_vllm_deep_ops_dir_not_basename():
    cfg = {"image_repo_path":
           "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops"}
    assert _resolve_framework(cfg, "/ws/paged.py") == "vllm"


def test_framework_falls_back_to_kernel_path_scan():
    assert _resolve_framework({}, "/ws/sglang/srt/layers/attn/k.py") == "sglang"


def test_framework_empty_when_unknown():
    # Unresolvable -> "" so the launcher OMITS --framework and forge-loop infers.
    assert _resolve_framework({}, "/tmp/scratch/kernel.py") == ""
    assert _resolve_framework({"image_repo_path": "/opt/rocmbench"}, "/ws/snippet.py") == ""


def test_framework_case_insensitive():
    assert _resolve_framework({"image_repo_path": "/x/vLLM/sub"}, "/ws/k.py") == "vllm"
