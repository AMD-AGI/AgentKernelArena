from __future__ import annotations
from typing import Any as Any
from typing import TypeGuard as TypeGuard
import math as math
import struct as struct
import torch as torch

def _initialize_mla_cache(cache, generator):
    """Write packed payloads and scale slots, using bounded scratch space."""
    page_size = cache.shape[1]
    raw = cache.view(torch.uint8).view(cache.shape[0], page_size * 584)
    raw.zero_()
    # Batch pages into bounded scratch rather than launching per physical page.
    pages_per_chunk = max(1, 8192 // page_size)
    for start in range(0, cache.shape[0], pages_per_chunk):
        stop = min(start + pages_per_chunk, cache.shape[0])
        count = stop - start
        values = torch.randn(
            (count, page_size, 512), dtype=torch.float32,
            device=cache.device, generator=generator,
        )
        payload = raw[start:stop, : page_size * 576].view(count, page_size, 576)
        payload[:, :, :448].copy_(
            values[:, :, :448].clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        payload[:, :, 448:].copy_(values[:, :, 448:].to(torch.bfloat16).view(torch.uint8))
        scales = raw[start:stop, page_size * 576 :].view(count, page_size, 8)
        scales[:, :, :7].random_(124, 128, generator=generator)



def _valid_tensor(
    value: Any, like: torch.Tensor | None = None
) -> TypeGuard[torch.Tensor]:
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.is_conj()
        and not value.is_neg()
        and (like is None or value.device == like.device)
    )


def _valid_cache(cache: Any, q: torch.Tensor) -> TypeGuard[torch.Tensor]:
    if (
        not _valid_tensor(cache, q)
        or cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or cache.ndim != 4
        or cache.shape[0] <= 0
        or cache.shape[1] not in (2, 64, 128, 256)
        or cache.shape[2:] != (1, 584)
        or cache.stride()[1:] != (584, 584, 1)
    ):
        return False
    # Allow page padding; the backend views each full page stride as uint32.
    stride = cache.stride(0)
    offset = cache.storage_offset()
    return (
        stride >= cache.shape[1] * 584
        and stride % 4 == 0
        and offset == 0
        and offset + cache.shape[0] * stride <= cache.untyped_storage().nbytes()
    )


def _valid_indices(indices: Any, lengths: Any, q: torch.Tensor) -> bool:
    if (
        not _valid_tensor(indices, q)
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
        or indices.ndim != 3
        or indices.shape[:2] != q.shape[:2]
        or indices.shape[-1] <= 0
        or indices.shape[-1] % 64 != 0
    ):
        return False
    return lengths is None or (
        _valid_tensor(lengths, q)
        and lengths.dtype == torch.int32
        and lengths.is_contiguous()
        and lengths.shape == (q.shape[0],)
    )


def _valid_scale(value: Any) -> bool:
    if type(value) not in (float, int):
        return False
    try:
        fp32 = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        return False
    return math.isfinite(fp32) and fp32 > 0


def check_init_buffers(inputs, tensor_names, seed=0) -> torch.Generator:
    """Validate metadata before writes; independent buffers must not overlap."""
    if type(inputs) is not dict:
        raise ValueError("inputs must be a dictionary of canonical input names")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    device = None
    ranges = []
    for name in tensor_names:
        tensor = inputs.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{name}: expected a dense strided Tensor")
        if not tensor.is_contiguous() or tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name}: expected a contiguous CPU/CUDA tensor")
        if device is not None and device != tensor.device:
            raise ValueError("all input buffers must use the same device")
        device = tensor.device
        if tensor.numel():
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            for other, low, high in ranges:
                if start < high and low < end:
                    raise ValueError(f"input buffers overlap: {other} and {name}")
            ranges.append((name, start, end))
    if device is None:
        raise ValueError("at least one input tensor is required")
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def initialize_mla_inputs(inputs, *, seed=0, cache_templates=None):
    """Fill compact replay buffers in place without changing scalars or global RNG."""
    required = {"q", "kv_cache", "sparse_indices", "sm_scale"}
    optional = {
        "sparse_lens",
        "extra_kv_cache",
        "extra_sparse_indices",
        "extra_sparse_lens",
        "sinks",
    }
    if type(inputs) is not dict or not required <= inputs.keys() <= required | optional:
        raise ValueError("expected canonical MLA inputs")
    rng = check_init_buffers(inputs, tuple(k for k in inputs if k != "sm_scale"), seed)
    q = inputs["q"]
    if (
        not _valid_tensor(q)
        or q.dtype != torch.bfloat16
        or q.ndim != 4
        or q.shape[0] <= 0
        or q.shape[1] != 1
        or q.shape[2] not in (64, 128)
        or q.shape[3] != 512
        or not _valid_scale(inputs["sm_scale"])
    ):
        raise ValueError("expected BF16 MLA queries and a positive finite FP32 scale")
    extra = "extra_kv_cache" in inputs
    if extra != ("extra_sparse_indices" in inputs) or (
        "extra_sparse_lens" in inputs and not extra
    ):
        raise ValueError("extra KV cache and indices must be supplied together")
    prefixes = ("", "extra_") if extra else ("",)
    for prefix in prefixes:
        cache, indices = inputs[prefix + "kv_cache"], inputs[prefix + "sparse_indices"]
        if not _valid_cache(cache, q) or not _valid_indices(
            indices, inputs.get(prefix + "sparse_lens"), q
        ):
            raise ValueError(f"invalid {prefix}cache, indices or lengths buffers")
        if cache.shape[0] * cache.shape[1] > 2**31 or indices.shape[-1] >= 2**31:
            raise ValueError(
                "cache capacity and sparse width must fit int32 indices/lengths"
            )
    sink = inputs.get("sinks")
    if sink is not None and (
        not _valid_tensor(sink, q)
        or sink.dtype != torch.float32
        or sink.shape != (q.shape[2],)
    ):
        raise ValueError("sinks must be an FP32 vector with one value per head")

    # Keep scaled logits moderate without overwriting the caller's scale.
    q.normal_(std=min(1.0, 1.0 / (math.sqrt(512) * inputs["sm_scale"])), generator=rng)
    for branch, prefix in enumerate(prefixes):
        cache, indices = inputs[prefix + "kv_cache"], inputs[prefix + "sparse_indices"]
        # Cache draws use an independent stream so cached and uncached creation
        # produces identical Q, index, length and sink tensors.
        key = (seed, prefix, tuple(cache.shape), cache.dtype, str(cache.device))
        if cache_templates is not None and key in cache_templates:
            cache.copy_(cache_templates[key])
        else:
            cache_rng = torch.Generator(device=q.device).manual_seed((seed + branch + 1) % (2**63))
            _initialize_mla_cache(cache, cache_rng)
            if cache_templates is not None:
                cache_templates[key] = cache.clone()
        indices.random_(0, cache.shape[0] * cache.shape[1], generator=rng)
        width = indices.shape[-1]
        lengths = torch.randint(
            0,
            width + 1,
            (q.shape[0],),
            dtype=torch.int32,
            device=q.device,
            generator=rng,
        )
        # Negative padding is applied independently by the case profile. Keep
        # valid nonnegative indices beyond runtime lengths to detect ignored lengths.
        if prefix + "sparse_lens" in inputs:
            inputs[prefix + "sparse_lens"].copy_(lengths)
    if sink is not None:
        sink.normal_(generator=rng)
    return inputs


_callable = initialize_mla_inputs


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
