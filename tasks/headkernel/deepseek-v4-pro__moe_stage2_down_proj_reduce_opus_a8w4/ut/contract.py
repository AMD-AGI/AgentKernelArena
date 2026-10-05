"""Portable metadata and coverage gates. No GPU imports at module import time."""
import dataclasses
import enum
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def describe(value):
    if hasattr(value, 'untyped_storage') and hasattr(value, 'stride'):
        return {'kind': 'tensor', 'shape': list(value.shape), 'stride': list(value.stride()),
                'dtype': str(value.dtype), 'storage_offset': value.storage_offset(),
                'storage_bytes': value.untyped_storage().nbytes(), 'device': str(value.device),
                'attributes': {k: describe(v) for k,v in vars(value).items()}}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {'kind': 'dataclass', 'type': type(value).__module__ + ':' + type(value).__qualname__,
                'fields': {f.name: describe(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if type(value).__name__ == 'ActivationType' and hasattr(value, 'value'):
        return {'kind': 'aiter_activation', 'value': int(value.value), 'name': str(value).split('.')[-1]}
    if isinstance(value, enum.Enum):
        return {'kind': 'enum', 'type': type(value).__module__ + ':' + type(value).__qualname__, 'name': value.name}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list', 'items': [describe(v) for v in value]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value): raise TypeError('Only string metadata keys supported')
        return {k: describe(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or abs(value) == float('inf'):
            return {'kind': 'float', 'value': str(value)}
        return value
    if type(value).__module__ == 'torch' and type(value).__name__ in ('dtype', 'device'):
        return {'kind': type(value).__name__, 'value': str(value)}
    return {'kind': 'opaque', 'type': type(value).__module__ + ':' + type(value).__qualname__}


def contains_opaque(value):
    if isinstance(value, dict): return value.get('kind') == 'opaque' or any(contains_opaque(v) for v in value.values())
    if isinstance(value, list): return any(contains_opaque(v) for v in value)
    return False


def case_identity(seam, inputs):
    """Do not include pointers, elapsed times, or observation counts in case identity."""
    return seam + '-' + digest(inputs)[:24]


def coverage(records, required_cases, observed_cases=()):
    ids = [x['case_id'] for x in records]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate case identities')
    found = set(ids)
    missing = sorted(set(required_cases) - found)
    missing_observed = sorted(set(observed_cases) - found)
    return {'complete': not missing and not missing_observed, 'missing_required': missing,
            'missing_observed': missing_observed, 'expected_count': len(set(required_cases) | set(observed_cases)),
            'actual_count': len(found)}


def require_latest(record, run_id, source_sha256):
    if record.get('run_id') != run_id or record.get('source_sha256') != source_sha256:
        raise ValueError('Historical or mismatched source capture cannot be labeled current')
    if record.get('origin') not in ('fresh_eager_call', 'fresh_graph_replay'):
        raise ValueError('Missing actual current call origin')
    if record.get('status') != 'captured' or contains_opaque(record.get('bindings')):
        raise ValueError('Incomplete input/control-state capture')
    if not record.get('input_snapshot') or not record.get('output_snapshot'):
        raise ValueError('Metadata is not a tensor oracle')


def merge_ranges(ranges):
    result=[]
    for start,end in sorted(ranges):
        if start < 0 or end < start: raise ValueError('Invalid byte range')
        if result and start <= result[-1][1]: result[-1][1]=max(result[-1][1],end)
        else: result.append([start,end])
    return result


def logical_m(seam, bindings):
    key = {'mla':'q', 'moe1':'a', 'moe2':'out'}[seam]
    shape = bindings[key]['shape']
    if not shape: raise ValueError('Missing logical token axis')
    return int(shape[0])
