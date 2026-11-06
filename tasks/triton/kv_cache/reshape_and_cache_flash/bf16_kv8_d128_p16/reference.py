"""Float64 reference of a paged-KV-cache scatter: cache[slot] = k/v for every slot >= 0."""
import torch


def reference(*, key, value, key_cache, value_cache, slot_mapping):
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    out = []
    for source, cache in ((key, key_cache), (value, value_cache)):
        result = cache.double()
        flat = result.view(-1, cache.shape[2], cache.shape[3])  # [slots, heads, head_size]
        flat[slots] = source[valid].double()
        out.append(result)
    return tuple(out)
