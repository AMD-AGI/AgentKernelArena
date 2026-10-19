"""Workload adapters: fixed requests, media requests and canonical trace windows.

The host prepares bindings before optimization. This module never resolves a
moving dataset or installs dependencies while measuring the candidate.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

_CLIENT_RUNTIMES = {}


def _client_data_filter(member, destination):
    if not member.issym():
        return tarfile.data_filter(member, destination)
    # CPython 3.10.12 resolves symlinks against the wrong base in data_filter.
    # Keep path filtering and explicitly validate their real, relative target;
    # strip ownership/mode just as data_filter does for symbolic links.
    root = Path(destination).resolve()
    parent = (root/member.name).parent.resolve()
    linked = (parent/member.linkname).resolve()
    if Path(member.name).is_absolute() or Path(member.linkname).is_absolute() or not linked.is_relative_to(root):
        raise ValueError('Client runtime link escapes its root')
    safe = copy.copy(tarfile.tar_filter(member, destination))
    safe.uid = safe.gid = safe.uname = safe.gname = safe.mode = None
    return safe


def composed_script(data, relative, lock):
    if lock['scenario'] == 'text' and relative == 'infx/bench_serving/benchmark_serving.py':
        text = data.decode()
        anchor = '    goodput_config_dict = check_goodput_args(args)\n'
        if text.count(anchor) != 1 or 'emit_request_plan' in text:
            raise ValueError('Upstream text client request-plan contract changed')
        return text.replace(anchor, '    from ._aka_request_plan import emit_request_plan\n'
                            '    emit_request_plan(input_requests)\n' + anchor).encode()
    if lock['scenario'] == 'multimodal' and relative == 'benchmarks/'+lock['benchmark_script']:
        text = data.decode()
        if text.count('    --save-result\n') != 1 or '--save-detailed' in text:
            raise ValueError('Upstream media client reporting contract changed')
        return text.replace('    --save-result\n', '    --save-detailed\n    --save-result\n').encode()
    return data


def _runtime_inventory(root):
    rows = {}
    for path in root.rglob('*'):
        name = str(path.relative_to(root))
        if path.is_symlink():
            rows[name] = ['link', os.readlink(path)]
        elif path.is_file():
            rows[name] = ['file', hashlib.sha256(path.read_bytes()).hexdigest()]
        elif path.is_dir():
            rows[name] = ['directory']
        else:
            raise ValueError('Unsupported client runtime file')
    return rows


def client_python(root, lock):
    """Extract a host-prepared, content-addressed interpreter, never install it.

    The archive includes Python and its complete package environment. Extraction
    and verification occur outside client timing. Each clean container has a
    private /tmp; repeated calls verify the original extracted inventory.
    """
    spec = lock.get('client_runtime')
    if not spec:
        raise ValueError('Packaged text client requires a prepared immutable Python runtime')
    if spec.get('kind', 'archive') == 'image':
        executable = Path(spec['executable'])
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError('Pinned image does not provide its declared client interpreter')
        return str(executable)
    archive = root/'assets'/spec['asset']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != lock['assets'][spec['asset']]['sha256']:
        raise ValueError('Client runtime archive differs from its locked digest')
    record = _CLIENT_RUNTIMES.get(digest)
    if record:
        destination, expected = record
        if _runtime_inventory(destination) != expected:
            raise ValueError('Prepared client runtime changed during the action')
    else:
        destination = Path(tempfile.mkdtemp(prefix='aka-client-python-'))
        with tarfile.open(archive) as source:
            members = source.getmembers()
            if len(members) > 100000 or sum(m.size for m in members) > 2*1024**3:
                raise ValueError('Client runtime exceeds extraction budget')
            for member in members:
                target = (destination/member.name).resolve()
                if not target.is_relative_to(destination) or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError('Unsafe client runtime archive member')
                if member.issym() or member.islnk():
                    linked = (target.parent/member.linkname) if member.issym() else (destination/member.linkname)
                    if Path(member.linkname).is_absolute() or not linked.resolve().is_relative_to(destination):
                        raise ValueError('Client runtime link escapes its root')
            source.extractall(destination, filter=_client_data_filter)
        expected = _runtime_inventory(destination)
        _CLIENT_RUNTIMES[digest] = destination, expected
    executable = destination/spec['executable']
    if not executable.resolve().is_relative_to(destination) or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError('Prepared client interpreter is missing or escapes its root')
    return str(executable)


def compose_client(client, lock, request_helper=None):
    """Detailed request lengths are report-only output, outside device timing."""
    if lock['scenario'] == 'multimodal':
        script = client/'benchmarks'/lock['benchmark_script']
        script.write_bytes(composed_script(script.read_bytes(), 'benchmarks/'+lock['benchmark_script'], lock))
    elif lock['scenario'] == 'text':
        script = client/'infx/bench_serving/benchmark_serving.py'
        script.write_bytes(composed_script(script.read_bytes(), str(script.relative_to(client)), lock))
        if request_helper is None:
            raise ValueError('Native request-plan reporting helper is required')
        (script.parent/'_aka_request_plan.py').write_bytes(Path(request_helper).read_bytes())


def substitute(value, root, port=0):
    replacements = {'{model}': str(root/'model'), '{client}': str(root/'client'),
                    '{root}': str(root), '{port}': str(port), '{assets}': str(root/'assets')}
    for key, target in replacements.items():
        value = value.replace(key, target)
    return value


def manifest_identity(config, lock, dtype):
    manifest = lock['request_manifest']
    params = dict(concurrency=int(config['envs']['CONC']), gpu_count=lock['gpu_count'],
                  dataset_sha256=manifest['dataset_sha256'])
    if 'TP' in config['envs']:
        params['tp'] = int(config['envs']['TP'])
    if lock['scenario'] == 'agentx':
        params.update(workload_kind='trace_window', duration_target_s=manifest['duration_s'],
            recipe_fingerprint=lock['agentx']['recipe_fingerprint'],
            trace_requests=manifest['requests'],
            client_sha256=hashlib.sha256(json.dumps(lock['agentx']['client'], sort_keys=True).encode()).hexdigest())
    else:
        params.update(num_requests=len(manifest['input_lens']), input_lens=manifest['input_lens'],
                      output_lens=manifest['output_lens'])
    return dict(test_case_id='serving', status='PASS', dtype=dtype,
                shape=[params['concurrency']], params=params)


def check_plan(config, lock, prompts):
    """Text AgentX probes cover locked trace lengths without imposing an op list."""
    requests = lock['request_manifest']['requests']
    indices = lock['correctness']['trace_probe_indices']
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(requests) for i in indices):
        raise ValueError('Invalid protected trace correctness coverage')
    groups = []
    for index in indices:
        record = requests[index]
        tokens = prompts[index % len(prompts)]
        length = record['input_tokens']
        expanded = (tokens * ((length + len(tokens) - 1)//len(tokens)))[:length]
        groups.append(dict(kind='trace_shape', concurrency=1, inputs=[expanded], outputs=[record['output_tokens']]))
    return groups


def media_plan(root, lock):
    data = json.loads((root/'media_checks.json').read_text())
    if hashlib.sha256((root/'media_checks.json').read_bytes()).hexdigest() != lock['correctness']['media_sha256']:
        raise ValueError('Multimodal correctness corpus differs from lock')
    if len(data) != lock['correctness']['model_samples'] or not data:
        raise ValueError('Multimodal correctness coverage changed')
    for case in data:
        if type(case.get('output_tokens')) is not int or case['output_tokens'] <= 0:
            raise ValueError('Invalid multimodal decode length')
        contents = [part for message in case['messages'] for part in message.get('content', []) if isinstance(part, dict)]
        images = [p['image_url']['url'] for p in contents if p.get('type') == 'image_url']
        if not images or any(not image.startswith('data:image/') for image in images):
            raise ValueError('Media probes must include frozen inline image inputs')
    # The opaque input marker is only an alignment key, not a claim about the
    # image's token count. Performance independently checks real token counts.
    return [dict(kind='media', concurrency=1, inputs=[[0] for _ in data],
                 outputs=[case['output_tokens'] for case in data], media=data)]


def media_generations(root, port, plan, send, *, return_logprob=True):
    samples = []
    from _aka_serving_backends import _token
    for group in plan:
        for case in group['media']:
            body = dict(model=str(root/'model'), messages=case['messages'], temperature=0,
                max_tokens=case['output_tokens'], ignore_eos=True, return_token_ids=True,
                return_tokens_as_token_ids=True, logprobs=return_logprob, top_logprobs=1 if return_logprob else 0)
            response = send(port, 'v1/chat/completions', body)
            choice = response['choices'][0]
            ids = choice.get('token_ids')
            if (not isinstance(ids, list) or len(ids) != case['output_tokens']
                    or any(type(i) is not int or i < 0 for i in ids)
                    or response['usage']['completion_tokens'] != len(ids)):
                raise ValueError('Multimodal generation omitted token IDs or changed work')
            item = dict(tokens=[0, *ids], input_tokens=[0], output_length=len(ids),
                check_kind='media', concurrency=1, return_logprob=return_logprob,
                media_sha256=hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest())
            if return_logprob:
                rows = choice['logprobs']['content']
                if len(rows) != len(ids) or [_token(row['token']) for row in rows] != ids:
                    raise ValueError('Multimodal numerical coverage changed')
                item['actual'] = [[None, 0], *[[row['logprob'], token] for row, token in zip(rows, ids)]]
            samples.append(item)
    return samples


def trace_metrics(records, manifest):
    """Account for every completed native replay request; never infer counts from TPS."""
    expected = {row['id']: row for row in manifest['requests']}
    if len(expected) != len(manifest['requests']):
        raise ValueError('Trace manifest contains duplicate turn IDs')
    inputs, outputs, ids, starts, ends = [], [], [], [], []
    for record in records:
        meta = record.get('metadata', {})
        if record.get('error') or meta.get('error') or record.get('status') in ('error', 'failed'):
            raise ValueError('Trace replay contains failed requests')
        if meta.get('was_cancelled') is not False or meta.get('cancellation_time_ns') is not None:
            raise ValueError('Trace replay contains cancelled or incomplete requests')
        if meta.get('context_overflow_skip', False) is not False:
            raise ValueError('Trace replay skipped request work')
        if meta.get('benchmark_phase') not in (None, 'profiling'):
            continue
        conversation = meta['conversation_id'].split('::', 1)[0]
        turn = str(meta['turn_index'])
        target = expected.get(conversation + ':' + turn)
        if target is None:
            raise ValueError('Replay used a turn outside the frozen trace corpus')
        def metric(name):
            result = record['metrics'][name]
            value = result['value'] if isinstance(result, dict) else result
            # Native exporters may encode an integer token count as 8.0.
            # Accept that exact representation, never round fractional work.
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value <= 0 or int(value) != value):
                raise ValueError('Trace token work must be an exact positive integer')
            return int(value)
        incoming, outgoing = metric('input_sequence_length'), metric('output_sequence_length')
        if incoming != target['input_tokens'] or outgoing != target['output_tokens']:
            raise ValueError('Replay changed per-turn token work')
        inputs.append(incoming); outputs.append(outgoing)
        # A trajectory can revisit a turn during a timed replay. Timestamp and
        # session identity distinguish those real calls from duplicated rows.
        identity = [meta.get('session_num'), meta['conversation_id'], turn,
                    meta['request_start_ns'], meta['request_end_ns']]
        if (any(type(meta[key]) is not int or meta[key] < 0
                for key in ('request_start_ns', 'request_end_ns'))
                or meta['request_end_ns'] <= meta['request_start_ns']):
            raise ValueError('Trace request has invalid native timestamps')
        ids.append(hashlib.sha256(json.dumps(identity).encode()).hexdigest())
        starts.append(meta['request_start_ns']); ends.append(meta['request_end_ns'])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Trace replay omitted or duplicated request evidence')
    duration = (max(ends) - min(starts))/1e9
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid native replay duration')
    return dict(duration_s=duration, completed_requests=len(ids), input_tokens=sum(inputs),
                output_tokens=sum(outputs), output_tokens_per_s=sum(outputs)/duration,
                input_lens=inputs, output_lens=outputs, request_ids=ids, failed_requests=0)


def verify_trace_export(aggregate, metrics):
    """Reconcile native counters so missing JSONL rows cannot alter the work."""
    if not isinstance(aggregate, dict) or aggregate.get('was_cancelled') is not False:
        raise ValueError('Canonical replay did not finish normally')
    def count(name, default=None):
        if name not in aggregate and default is not None:
            return default
        field = aggregate.get(name)
        value = field.get('avg') if isinstance(field, dict) else None
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or int(value) != value):
            raise ValueError('Missing or invalid native trace counter: ' + name)
        return int(value)
    completed = metrics['completed_requests']
    if count('request_count') != completed or count('completed_request_count', completed) != completed:
        raise ValueError('Native trace counters disagree with per-request evidence')
    for name in ('error_request_count', 'context_overflow_count', 'skipped_context_overflow_count'):
        if count(name, 0) != 0:
            raise ValueError('Canonical replay contains failed or skipped request work')


def _client_invocation(root, config, lock, client, port):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('AIPERF_', 'AGENTIC_', 'AGENTX_', 'SRT_'))}
    env.update({key: str(value) for key, value in config['envs'].items()})
    env.update(MODEL=str(root/'model'), PORT=str(port), RESULT_DIR=str(root),
        RESULT_FILENAME='inferencex_result', MAGPIE_RUN_PHASE='client', PROFILE='0', RUN_EVAL='false',
        PYTHONPATH=str(client), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    env.update(PIP_NO_INDEX='1', UV_OFFLINE='1', UV_PYTHON_DOWNLOADS='never',
               PYTHONDONTWRITEBYTECODE='1')
    if lock['scenario'] == 'text' or lock.get('client_runtime'):
        env['MAGPIE_BENCHMARK_PYTHON'] = client_python(root, lock)
    if lock['scenario'] == 'agentx':
        spec = lock['agentx']['client']
        command = [substitute(arg, root, port) for arg in spec['argv']]
        if lock.get('client_runtime'):
            command = [arg.replace('{client_python}', env['MAGPIE_BENCHMARK_PYTHON']) for arg in command]
        env.update({k: substitute(v, root, port) for k, v in spec['env'].items()})
    else:
        command = ['bash', str(client/'benchmarks'/config['benchmark_script'])]
    return command, env


def _run_client(root, config, lock, limits, client, port, *, prepare=False):
    command, env = _client_invocation(root, config, lock, client, port)
    prefix = 'prepared_' if prepare else ''
    (root/(prefix+'client_command.json')).write_text(json.dumps(command))
    plan = None
    with tempfile.TemporaryFile() as stream:
        if lock['scenario'] == 'text':
            env.update(AKA_REQUEST_PLAN_FD=str(stream.fileno()),
                       AKA_REQUEST_PLAN_MODE='prepare' if prepare else 'measure')
        completed = subprocess.run(command, cwd=client, env=env, text=True, capture_output=True,
            timeout=limits['client_timeout_s'], pass_fds=(stream.fileno(),) if lock['scenario'] == 'text' else ())
        (root/(prefix+'client.log')).write_text(completed.stdout + '\n' + completed.stderr)
        if completed.returncode:
            raise RuntimeError('Benchmark client failed: ' + (root/(prefix+'client.log')).read_text()[-8000:])
        if lock['scenario'] == 'text':
            stream.seek(0)
            raw = stream.read(32*1024*1024+1)
            count = int(config['envs'].get('NUM_PROMPTS', int(config['envs'].get('CONC', 32))*10))
            plan = read_request_plan(raw, count)
            (root/(prefix+'request_plan.json')).write_bytes(raw)
    return command, plan


def read_request_plan(raw, count):
    if not raw or len(raw) > 32*1024*1024:
        raise ValueError('Missing or oversized native request plan')
    plan = json.loads(raw)
    if set(plan) != {'version', 'requests'} or type(plan['version']) is not int or plan['version'] != 1:
        raise ValueError('Invalid native request plan protocol')
    rows = plan['requests']
    if not isinstance(rows, list) or len(rows) != count or not rows:
        raise ValueError('Native request plan changed request count')
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'prompt_sha256', 'input_tokens', 'output_tokens'}
                or not isinstance(row['prompt_sha256'], str) or len(row['prompt_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['prompt_sha256'])
                or any(type(row[key]) is not int or row[key] <= 0 for key in ('input_tokens', 'output_tokens'))):
            raise ValueError('Invalid native request plan record')
    return plan


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def text_identity(root, config, lock, limits, row):
    """Freeze real native payloads independently, before starting either server."""
    from _aka_serving import prepare_client
    client = prepare_client(lock)
    _, plan = _run_client(root, config, lock, limits, client, 0, prepare=True)
    prepare_client(lock)
    params = row['params']
    for name, key, scalar in [('input_lens', 'input_tokens', 'input_tokens'),
                              ('output_lens', 'output_tokens', 'output_tokens_per_request')]:
        actual = [item[key] for item in plan['requests']]
        nominal = params.get(name, [params.get(scalar)]*params['num_requests'])
        if actual != nominal:
            params[name] = actual
            params.pop(scalar, None)
    params['request_plan_sha256'] = plan_digest(plan)
    return row


def performance(root, config, lock, limits, client, port, row):
    command, plan = _run_client(root, config, lock, limits, client, port)
    if plan is not None and plan_digest(plan) != row['params'].get('request_plan_sha256'):
        raise ValueError('Timed client changed the prepared native request plan')
    from _aka_serving import prepare_client
    prepare_client(lock)  # Check the complete timed source tree after execution.
    if lock.get('client_runtime'):
        client_python(root, lock)  # Also reject interpreter/package mutations.
    if lock['scenario'] == 'agentx':
        artifact = root/'aiperf_artifacts/profile_export_aiperf.json'
        aggregate = json.loads(artifact.read_text())
        records = [json.loads(line) for line in (root/'aiperf_artifacts/profile_export.jsonl').read_text().splitlines()]
        metrics = trace_metrics(records, lock['request_manifest'])
        verify_trace_export(aggregate, metrics)
    else:
        measured = json.loads((root/'inferencex_result.json').read_text())
        metrics = dict(duration_s=measured['duration'], completed_requests=measured['completed'],
            input_tokens=measured['total_input_tokens'], output_tokens=measured['total_output_tokens'],
            output_tokens_per_s=measured['output_throughput'], p99_tpot_ms=measured['p99_tpot_ms'])
        for name in ('input_lens', 'output_lens'):
            expected = row['params'].get(name)
            if expected is None:
                length = int(config['envs']['ISL' if name == 'input_lens' else 'OSL'])
                expected = [length]*row['params']['num_requests']
            if measured.get(name) != expected:
                raise ValueError('Benchmark changed per-request ' + name)
            if name in row['params']:
                metrics[name] = measured[name]
    if metrics['duration_s'] < limits['minimum_measurement_s']:
        raise ValueError('Measurement window shorter than protected minimum')
    row.update(execution_time_ms=1000*metrics['duration_s'], benchmark_method='serving_wall_clock', metrics=metrics)
    metadata = dict(scenario=lock['scenario'], client_command=command,
        dataset_sha256=lock.get('request_manifest', {}).get('dataset_sha256'),
        request_plan_sha256=row['params'].get('request_plan_sha256'))
    if lock.get('client_runtime'):
        runtime = lock['client_runtime']
        metadata['client_runtime'] = dict(runtime,
            sha256=lock['assets'][runtime['asset']]['sha256'] if 'asset' in runtime else lock['image'])
    if os.environ.get('ARENA_EVAL_PHASE') == 'task_validation':
        paths = [('benchmarks/benchmark_lib.sh', 'upstream client and command construction')]
        if lock['scenario'] != 'agentx':
            paths.append(('benchmarks/'+config['benchmark_script'], 'executed client driver'))
        if lock['scenario'] == 'text':
            paths.extend((name, 'timed HTTP client') for name in (
                'infx/bench_serving/benchmark_serving.py', 'infx/bench_serving/backend_request_func.py'))
        sources = []
        for relative, role in paths:
            path = client/relative
            if path.is_file():
                text = path.read_text()
                sources.append(dict(path=relative, role=role, source=text,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), start_line=1))
        if lock['scenario'] == 'multimodal':
            spec = importlib.util.find_spec('vllm.benchmarks.serve')
            if spec is None or not spec.origin:
                raise ValueError('Cannot inspect the pinned native media measurement client')
            path = Path(spec.origin)
            sources.append(dict(path='image:vllm.benchmarks.serve', role='native timed media client',
                source=path.read_text(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), start_line=1))
        metadata.update(benchmark_sources=sources,
            client_log=(root/'client.log').read_text()[-24000:])
    return row, metadata
