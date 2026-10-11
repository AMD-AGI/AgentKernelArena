import ctypes
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import secrets

from jinja2 import Template

from csrc.cpp_itfs.utils import AITER_CORE_DIR, BUILD_DIR, compile_template_op, str_to_bool

TASK_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = Path(__file__).resolve().parent
SOURCE = TASK_ROOT/'source/pa_kernels.cuh'
MD_NAME = "qwen_paged_attention_native"
# The closed GPU source cannot observe or prepopulate this host-owned build
# namespace. Repeated calls in this process reuse its validated specialization.
BUILD_NONCE = secrets.token_hex(16)
MARKERS = ['__inline__ __device__ void\n_paged_attention_kernel(',
           '__inline__ __device__ void _paged_attention_ll4mi_reduce_kernel(']

with (NATIVE_ROOT/'pa_ragged.cpp.jinja').open() as f:
    src_template = Template(f.read())


def source_identity():
    if SOURCE.is_symlink() or not SOURCE.is_file():
        raise ValueError('Native candidate must be a regular task-local device header')
    spec = importlib.util.spec_from_file_location('qwen_pa_cpp_guard',TASK_ROOT/'ut/cpp_body_guard.py')
    guard = importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.validate_cpp(SOURCE.read_text(),(NATIVE_ROOT/'pa_kernels.cuh.orig').read_text(),MARKERS)
    files = [SOURCE,NATIVE_ROOT/'pa_ragged.cuh',NATIVE_ROOT/'pa_common.cuh',
             NATIVE_ROOT/'pa_ragged.cpp.jinja',Path(__file__)]
    return {str(p.relative_to(TASK_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def compile(
    gqa_ratio: int,
    head_size: int,
    npar_loops: int,
    dtype: str,
    kv_dtype: str,
    fp8_kv_dtype: str,
    out_dtype: str,
    block_size: int,
    alibi_enabled: bool = False,
    partition_size: int = 256,
    mtp: int = 1,
    logits_soft_cap_enabled: bool = False,
    func_name: str | None = None,
):
    # Only the recorded native ABI is qualified by this task. No environment
    # variable or editable Python callback may choose a different kernel path.
    signature = (gqa_ratio,head_size,npar_loops,dtype,kv_dtype,fp8_kv_dtype,out_dtype,
                 block_size,alibi_enabled,partition_size,mtp,logits_soft_cap_enabled)
    expected = (8,256,1,'__hip_bfloat16','__hip_bfloat16','auto','__hip_bfloat16',
                1,False,256,1,False)
    if signature != expected or func_name is not None:
        raise ValueError('Native paged attention compile request differs from recorded ABI')
    sources = source_identity()
    fingerprint = hashlib.sha256(json.dumps({'sources':sources,'signature':signature,'build_nonce':BUILD_NONCE},sort_keys=True).encode()).hexdigest()
    func_name = MD_NAME+'_'+fingerprint[:32]
    function = compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            str(NATIVE_ROOT/'pa_ragged.cuh'),
            str(SOURCE),
            str(NATIVE_ROOT/'pa_common.cuh'),
            f"{AITER_CORE_DIR}/csrc/include",
            f"{AITER_CORE_DIR}/csrc/include/ck_tile/",
        ],
        gqa_ratio=gqa_ratio,
        head_size=head_size,
        npar_loops=npar_loops,
        dtype=dtype,
        kv_dtype=kv_dtype,
        fp8_kv_dtype=fp8_kv_dtype,
        out_dtype=out_dtype,
        block_size=block_size,
        partition_size=partition_size,
        mtp=mtp,
        alibi_enabled=alibi_enabled,
        logits_soft_cap_enabled=logits_soft_cap_enabled,
        func_name=func_name,
        # choice: [GOLDEN, EXPERIMENTAL]. Classify original kernel and experimental kernel
        version='GOLDEN',
    )
    built = Path(BUILD_DIR)/func_name/'include/pa_kernels.cuh'
    if not built.is_file() or hashlib.sha256(built.read_bytes()).hexdigest()!=sources['source/pa_kernels.cuh']:
        raise RuntimeError('Compiled device header differs from the submitted candidate')
    receipt = {'schema':'qwen-native-compile-v1','source_sha256':sources,
               'compile_fingerprint':fingerprint,'function_name':func_name,
               'build_nonce':BUILD_NONCE,
               'recorded_signature':list(signature),'compiled_header_sha256':sources['source/pa_kernels.cuh']}
    build = TASK_ROOT/'build';build.mkdir(exist_ok=True)
    (build/'native_compile_report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return function


def paged_attention_ragged(
    out,  # [num_seqs, num_heads, head_size]
    workspace_buffer,  # [num_seqs, num_heads, max_num_partitions]
    query,  # [num_seqs, num_heads, head_size]
    key_cache,  # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache,  # [num_blocks, num_heads, head_size, block_size]
    scale,
    kv_indptr,
    kv_page_indices,  # [num_seqs, max_num_blocks_per_seq]dd
    kv_last_page_lens,  # [num_seqs]
    block_size,
    max_num_partitions,
    alibi_slopes,
    kv_cache_dtype,
    kv_cache_layout,
    logits_soft_cap,
    k_scale,
    v_scale,
    fp8_out_scale=None,
    partition_size=256,
    mtp=1,
    q_scale=None,
):
    import torch

    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    warpSize = torch.cuda.get_device_properties(out.device).warp_size
    if kv_cache_dtype == "auto":
        if query.dtype == torch.bfloat16:
            dtype = "__hip_bfloat16"
            kv_dtype = "__hip_bfloat16"
        elif query.dtype == torch.float16:
            dtype = "_Float16"
            kv_dtype = "_Float16"
        else:
            raise ValueError(f"Unsupported data type: {query.dtype}")
    elif kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3":
        if query.dtype == torch.bfloat16:
            dtype = "__hip_bfloat16"
            kv_dtype = "uint8_t"
        elif query.dtype == torch.float16:
            dtype = "_Float16"
            kv_dtype = "uint8_t"
        else:
            raise ValueError(f"Unsupported data type: {query.dtype}")
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")

    if out.dtype == torch.bfloat16:
        out_dtype = "__hip_bfloat16"
    elif out.dtype == torch.float16:
        out_dtype = "_Float16"
    else:
        raise ValueError(f"Unsupported data type: {out.dtype}")

    num_kv_heads = key_cache.size(1) if kv_cache_layout == "HND" else key_cache.size(2)
    num_seqs = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    q_stride = query.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride = (
        key_cache.stride(1) if kv_cache_layout == "HND" else key_cache.stride(2)
    )
    kv_seq_stride = (
        key_cache.stride(2) if kv_cache_layout == "HND" else key_cache.stride(1)
    )
    gqa_ratio = int(num_heads / num_kv_heads)
    npar_loops = math.ceil(max_num_partitions / warpSize)
    func = compile(
        gqa_ratio,
        head_size,
        npar_loops,
        dtype,
        kv_dtype,
        kv_cache_dtype,
        out_dtype,
        block_size,
        alibi_slopes is not None,
        partition_size,
        mtp,
        bool(logits_soft_cap),
    )

    alibi_slopes_ptr = (
        ctypes.cast(alibi_slopes.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if alibi_slopes is not None
        else ctypes.POINTER(ctypes.c_int)()
    )
    kv_indptr_ptr = ctypes.cast(kv_indptr.data_ptr(), ctypes.POINTER(ctypes.c_int))
    kv_page_indices_ptr = ctypes.cast(
        kv_page_indices.data_ptr(), ctypes.POINTER(ctypes.c_int)
    )
    kv_last_page_lens_ptr = (
        ctypes.cast(kv_last_page_lens.data_ptr(), ctypes.POINTER(ctypes.c_int))
        if block_size > 1
        else ctypes.POINTER(ctypes.c_int)()
    )

    k_scale_ptr = ctypes.cast(k_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
    v_scale_ptr = ctypes.cast(v_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
    fp8_out_scale_ptr = (
        ctypes.cast(fp8_out_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if fp8_out_scale
        else ctypes.POINTER(ctypes.c_int)()
    )

    (
        out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        workspace_buffer_ptr,
        scale,
        logits_soft_cap,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        stream,
    ) = torch_to_c_types(
        out,
        query,
        key_cache,
        value_cache,
        workspace_buffer,
        scale,
        logits_soft_cap,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        torch.cuda.current_stream(),
    )
    q_scale_ptr = (
        ctypes.cast(q_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if q_scale is not None
        else ctypes.POINTER(ctypes.c_float)()
    )
    func(
        out_ptr,
        workspace_buffer_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        kv_indptr_ptr,
        kv_page_indices_ptr,
        kv_last_page_lens_ptr,
        alibi_slopes_ptr,
        q_scale_ptr,
        k_scale_ptr,
        v_scale_ptr,
        fp8_out_scale_ptr,
        scale,
        logits_soft_cap,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        stream,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_ratio", type=int, required=True)
    parser.add_argument("--head_size", type=int, required=True)
    parser.add_argument("--npar_loops", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--fp8_kv_dtype", type=str, required=True)
    parser.add_argument("--out_dtype", type=str, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--alibi_enabled", type=str_to_bool, required=True)
    parser.add_argument("--func_name", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))
