"""Exact storage snapshots; sparse MLA pages retain original addresses and padding."""
import importlib
from pathlib import Path
import torch


def raw_storage(tensor):
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))


class Snapshot:
    def __init__(self, max_bytes=8 * 1024**3, device_copy=False):
        self.storages = {}; self.storage_keys = {}; self.max_bytes = max_bytes; self.used = 0
        self.device_copy = device_copy

    def _copy(self, value):
        return value.clone() if self.device_copy else value.cpu().clone()

    def tensor(self, value, page_indices=None):
        from contract import merge_ranges
        key = (str(value.device), value.untyped_storage().data_ptr(), value.untyped_storage().nbytes())
        sid = self.storage_keys.get(key)
        if sid is None:
            sid = str(len(self.storage_keys)); self.storage_keys[key] = sid
        storage = raw_storage(value)
        old = self.storages.get(sid)
        old_size = 0 if old is None else old['captured_bytes']
        if old is None or old['kind'] != 'full':
            # During graph capture, data-dependent sparse page discovery is forbidden.
            # Full-storage D2D clones are recorded before/after the native launch instead.
            if page_indices is None or self.device_copy:
                size = storage.numel()
                if self.used - old_size + size > self.max_bytes: raise MemoryError('Snapshot byte budget exceeded')
                self.storages[sid] = {'kind': 'full', 'data': self._copy(storage),
                    'nbytes': size, 'captured_bytes': size}
                self.used += size - old_size
            else:
                ids = page_indices.reshape(-1).to(torch.int64)
                ids = torch.unique(ids[ids >= 0] // value.shape[1], sorted=True).cpu().tolist()
                if ids and ids[-1] >= value.shape[0]: raise ValueError('KV index outside recorded pool')
                stride = value.stride(0) * value.element_size()
                offset = value.storage_offset() * value.element_size()
                ranges = [(offset+i*stride, offset+(i+1)*stride) for i in ids]
                if old is not None: ranges += [(x['start'],x['end']) for x in old['ranges']]
                ranges = merge_ranges(ranges)
                if any(end > storage.numel() for start,end in ranges): raise ValueError('KV padding outside storage')
                size = sum(end-start for start,end in ranges)
                if self.used-old_size+size > self.max_bytes: raise MemoryError('Merged sparse KV snapshot budget exceeded')
                self.storages[sid] = {'kind': 'sparse_ranges', 'ranges': [
                    {'start': start, 'end': end, 'data': self._copy(storage[start:end])}
                    for start,end in ranges], 'nbytes': storage.numel(), 'captured_bytes': size}
                self.used += size-old_size
        return {'kind': 'tensor', 'storage': sid, 'shape': list(value.shape), 'stride': list(value.stride()),
                'dtype': str(value.dtype), 'storage_offset': value.storage_offset(),
                'attributes': {k: self.encode(v) for k,v in vars(value).items()}}

    def encode(self, value):
        from contract import describe
        if torch.is_tensor(value): return self.tensor(value)
        if isinstance(value, tuple): return {'kind': 'tuple', 'items': [self.encode(x) for x in value]}
        if isinstance(value, list): return {'kind': 'list', 'items': [self.encode(x) for x in value]}
        if isinstance(value, dict): return {k: self.encode(v) for k, v in value.items()}
        return describe(value)

    def inputs(self, bindings, seam):
        out = {}
        pairs = {'k_cache': 'indices', 'extra_k_cache': 'extra_indices_in_kvcache'} if seam == 'mla' else {}
        for key, value in bindings.items():
            if key in pairs and torch.is_tensor(value):
                indices = bindings.get(pairs[key])
                if indices is None: raise ValueError('Paged KV requires exact indices for sparse snapshot')
                out[key] = self.tensor(value, indices)
            else: out[key] = self.encode(value)
        return {'tree': out, 'storages': self.storages}


def snapshot_to_cpu(snapshot):
    """Call only after the matching actual graph replay has completed on its stream."""
    def cpu(value):
        if torch.is_tensor(value): return value.cpu().clone()
        if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list): return [cpu(v) for v in value]
        return value
    return cpu(snapshot)


def restore(snapshot, device, module=None):
    storages = {}
    for sid, record in snapshot['storages'].items():
        if record['kind'] == 'full':
            raw = record['data'].to(device=device).clone()
        elif record['kind'] == 'sparse_ranges':
            raw = torch.zeros(record['nbytes'], dtype=torch.uint8, device=device)
            for region in record['ranges']:
                raw[region['start']:region['end']] = region['data'].to(device=device)
        else: raise ValueError('Unknown storage format')
        storages[sid] = raw
    def decode(value):
        if not isinstance(value, dict): return value
        kind = value.get('kind')
        if kind == 'tensor':
            dtype = getattr(torch, value['dtype'].removeprefix('torch.'))
            tensor = torch.empty(0, dtype=dtype, device=device).set_(storages[value['storage']].untyped_storage(),
                value['storage_offset'], tuple(value['shape']), tuple(value['stride']))
            for name, attr in value.get('attributes', {}).items(): setattr(tensor, name, decode(attr))
            return tensor
        if kind in ('tuple', 'list'):
            items = [decode(x) for x in value['items']]
            return tuple(items) if kind == 'tuple' else items
        if kind == 'dataclass':
            if value['type'] == 'aiter.ops.opus.moe_stage2_a8w4:OpusA8W4LaunchConfig':
                # Reconstruct with this leg's current class and compare the full instance, not just its ID.
                instance = value['fields']['instance']['fields']
                launch = module.stage2_launch_config(instance['kid'])
                from contract import describe
                actual = describe(launch)
                actual['type'] = value['type']
                if actual != value: raise ValueError('Opus launch instance definition changed')
                return launch
            raise ValueError('Unsupported dataclass control state: ' + value['type'])
        if kind == 'aiter_activation':
            from aiter import ActivationType
            return ActivationType(value['value'])
        if kind == 'float': return float(value['value'])
        if kind == 'dtype': return getattr(torch, value['value'].removeprefix('torch.'))
        if kind == 'device': return torch.device(device)
        if kind == 'enum':
            module_name, name = value['type'].split(':', 1)
            obj = importlib.import_module(module_name)
            for part in name.split('.'): obj = getattr(obj, part)
            return obj[value['name']]
        if kind == 'opaque': raise ValueError('Unsupported control state; explicit codec required')
        return {k: decode(v) for k,v in value.items()}
    return decode(snapshot['tree'])
