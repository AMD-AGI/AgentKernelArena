"""Float64 reference of the paged-KV-cache scatter with different K and V head sizes."""
import torch


def reference(*, key, value, kv_cache, slot_mapping):
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    result = kv_cache.double()
    flat = result.view(-1, kv_cache.shape[2], kv_cache.shape[3])  # [slots, heads, head_size_k + head_size_v]
    head_size_k = key.shape[2]
    flat[slots, :, :head_size_k] = key[valid].double()
    flat[slots, :, head_size_k:] = value[valid].double()
    return result
