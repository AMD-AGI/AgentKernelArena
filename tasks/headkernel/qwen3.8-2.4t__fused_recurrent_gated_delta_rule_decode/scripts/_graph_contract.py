"""Strict report parser for this task's protected graph replay contract."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def same_json(left, right):
    return json.dumps(left,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(right,sort_keys=True,separators=(',',':'),allow_nan=False)


def contract(root):
    path = Path(root)/'ut/benchmark_contract.json'
    value = json.loads(path.read_text())
    if value.get('schema') != 'qwen-protected-graph-v1':
        raise ValueError('Unsupported protected graph contract')
    return value


def source_hashes(root, policy):
    result = {}
    for name in policy['source_paths']:
        path = Path(root)/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError('Candidate source is not a regular task-local file')
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def provenance(root, policy):
    return {'source_sha256':source_hashes(root, policy),
            'case_contract_sha256':hashlib.sha256((Path(root)/'ut/benchmark_contract.json').read_bytes()).hexdigest()}


def validate_provenance(raw, root, policy):
    expected = provenance(root, policy)
    if any(raw.get(k) != v for k,v in expected.items()):
        raise ValueError('Measurement source or protected case contract differs')
    binding = policy.get('source_binding')
    if binding:
        source = expected['source_sha256'][binding['source_path']]
        if raw.get('source_binding') != {'launcher_source_sha256':source,
                'device_source_sha256':source, 'device_kernel':binding['device_kernel']}:
            raise ValueError('Runtime launcher/device binding does not match submitted source')


def benchmark_cases(raw, root):
    policy = contract(root)
    if not isinstance(raw, dict) or raw.get('timer') != 'cuda_graph':
        raise ValueError('A complete CUDA graph measurement is required')
    validate_provenance(raw, root, policy)
    validate_request(raw.get('request'), root, policy)
    for key, expected in [('warmup',10), ('iters',100)]:
        if type(raw.get(key)) is not int or raw[key] != expected:
            raise ValueError('Warmups and ordered sample counts are immutable')
    expected = {row['test_case_id']:row['params'] for row in policy['cases']}
    rows = raw.get('cases')
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Complete declared case coverage is required')
    seen, result = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid graph case')
        sig = row.get('sig')
        if sig not in expected or sig in seen or not same_json(row.get('params'),expected[sig]):
            raise ValueError('Case identity or full recorded ABI differs')
        seen.add(sig)
        if row.get('benchmark_method') != 'cuda_graph' or 'benchmark_fallback_reason' in row:
            raise ValueError('Event fallback is not supported for this graph contract')
        for key, value in [('benchmark_warmup',10), ('benchmark_samples',100), ('benchmark_effective_repeats',1)]:
            if type(row.get(key)) is not int or row[key] != value:
                raise ValueError('Graph timing metadata differs from the declared protocol')
        samples = row.get('samples_ms')
        if (not isinstance(samples,list) or len(samples)!=100
                or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in samples)):
            raise ValueError('All 100 ordered positive finite raw samples are required')
        for key, value in [('mean_ms',statistics.mean(samples)), ('median_ms',statistics.median(samples)), ('min_ms',min(samples))]:
            if (type(row.get(key)) not in (int,float) or not math.isfinite(row[key])
                    or not math.isclose(row[key],value,rel_tol=1e-12,abs_tol=0.0)):
                raise ValueError('Reported summary differs from the complete raw series')
        validation = row.get('validation')
        if not isinstance(validation,dict):
            raise ValueError('Measured replay validation is required')
        if type(validation.get('input_seed')) is not int or not 0<=validation['input_seed']<2**63:
            raise ValueError('The actual fresh CPU input seed must be retained')
        required = {'checked_invocations':118, 'timing_checked_invocations':114,
                    'fresh_input_sets':120, 'expected_device':'cpu',
                    'fresh_inputs_per_replay':True, 'measured_graph_validated':True,
                    'immutable_inputs_unchanged':True, 'returned_aliases_checked':True,
                    'tolerance':policy['tolerance'],
                    'correctness_challenges':[{'kind':kind,'passed':True} for kind in policy['challenges']],
                    **policy['validation_contract']}
        if any(type(validation.get(k)) is not type(v) or validation[k]!=v for k,v in required.items()):
            raise ValueError('Measured graph validation/reset contract is incomplete')
        error = validation.get('max_rel_err')
        if type(error) not in (int,float) or not math.isfinite(error) or error<0 or error>policy['tolerance']:
            raise ValueError('Measured graph output error exceeds the unchanged tolerance')
        result.append({'test_case_id':sig, 'execution_time_ms':row['mean_ms'],
                       'params':row['params'], 'samples_ms':samples, 'median_ms':row['median_ms'],
                       'min_ms':row['min_ms'], 'validation':validation,
                       **{k:v for k,v in row.items() if k.startswith('benchmark_')}})
    return result


def validate_request(request, root, policy):
    expected={**provenance(root,policy),
              'manifest_sha256':hashlib.sha256((Path(root)/'ut/meta.json').read_bytes()).hexdigest()}
    if (not isinstance(request,dict) or type(request.get('schema_version')) is not int or request['schema_version']!=1
            or request.get('phase')!='performance' or not isinstance(request.get('request_id'),str)
            or len(request['request_id'])!=48 or any(x not in '0123456789abcdef' for x in request['request_id'])
            or any(request.get(key)!=value for key,value in expected.items())):
        raise ValueError('Performance invocation/source/metadata binding differs')
    return request


def correctness_cases(raw, root):
    policy = contract(root)
    if not isinstance(raw,dict) or raw.get('status')!='ok' or raw.get('mode')!='cpu_truth_correctness':
        raise ValueError('Fresh independent correctness challenges are required')
    validate_provenance(raw,root,policy)
    expected = {row['test_case_id']:row['params'] for row in policy['cases']}
    rows = raw.get('cases',[])
    if len(rows)!=len(expected) or {row.get('sig') for row in rows}!=set(expected):
        raise ValueError('Correctness challenge coverage differs from the protected cases')
    for row in rows:
        if (not same_json(row.get('params'),expected[row['sig']]) or row.get('expected_device')!='cpu'
                or row.get('checks')!=[{'kind':k,'passed':True} for k in ('ordinary',*policy['challenges'])]
                or row.get('checked_invocations')!=5 or row.get('tolerance')!=policy['tolerance']):
            raise ValueError('Incorrect CPU-truth challenge record')
        error = row.get('max_rel_err')
        if type(error) not in (int,float) or not math.isfinite(error) or error<0 or error>policy['tolerance']:
            raise ValueError('CPU-truth challenge error exceeds the unchanged tolerance')
    return raw
