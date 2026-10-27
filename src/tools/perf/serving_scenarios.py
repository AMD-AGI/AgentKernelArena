"""Workload adapters: fixed requests, media requests and canonical trace windows.

The host prepares bindings before optimization. This module never resolves a
moving dataset or installs dependencies while measuring the candidate.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

_CLIENT_RUNTIMES = {}


def validate_client_review_sources(spec):
    """Optional, digest-bound Python/shell sources from the native client archive."""
    if not isinstance(spec, dict):
        raise ValueError('Client review sources require a runtime mapping')
    sources = spec.get('review_sources', {})
    if (not isinstance(sources, dict) or len(sources) > 16
            or sources and spec.get('kind', 'archive') != 'archive'):
        raise ValueError('Client review sources require a bounded archive file mapping')
    for name, digest in sources.items():
        if not isinstance(name, str):
            raise ValueError('Client review source path must be relative')
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or path.as_posix() != name
                or path.suffix not in {'.py', '.sh'} or not isinstance(digest, str)
                or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('Client review sources require relative source paths and SHA-256 digests')
    return sources


def client_review_sources(root, lock):
    """Read the exact native timer sources from its immutable runtime archive.

    The executed extraction is verified against this archive before and after
    client execution. No imports, extractions or external commands are needed.
    """
    spec = lock.get('client_runtime')
    if spec is None:
        return []
    declared = validate_client_review_sources(spec)
    if not declared:
        return []
    archive = root/'assets'/spec['asset']
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha != lock['assets'][spec['asset']]['sha256']:
        raise ValueError('Native client source archive differs from the locked digest')
    records, total = [], 0
    with tarfile.open(archive) as package:
        members = package.getmembers()
        if len(members) > 100000 or sum(member.size for member in members) > 2*1024**3:
            raise ValueError('Client runtime exceeds the source inspection budget')
        for name, expected in declared.items():
            matches = [member for member in members if member.name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 256*1024:
                raise ValueError('Native client review source is missing, ambiguous or exceeds its budget')
            total += matches[0].size
            if total > 512*1024:
                raise ValueError('Native client review sources exceed the combined budget')
            with package.extractfile(matches[0]) as stream:
                data = stream.read(256*1024+1)
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise ValueError('Native client review source differs from its locked digest')
            records.append(dict(path='client-runtime:'+name, role='native timing and phase implementation',
                sha256=actual, archive_sha256=archive_sha, source=data.decode('utf-8'), start_line=1))
    return records


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
        if __package__:
            from ...measurement import trace_input_token_policy, trace_manifest_identity
        else:
            from _aka_measurement import trace_input_token_policy, trace_manifest_identity
        params.update(workload_kind='trace_window', duration_target_s=manifest['duration_s'],
            recipe_fingerprint=lock['agentx']['recipe_fingerprint'],
            trace_input_token_policy=trace_input_token_policy(manifest),
            **trace_manifest_identity(manifest['requests']),
            client_sha256=hashlib.sha256(json.dumps(lock['agentx']['client'], sort_keys=True).encode()).hexdigest())
    else:
        params.update(num_requests=len(manifest['input_lens']), input_lens=manifest['input_lens'],
                      output_lens=manifest['output_lens'])
    return dict(test_case_id='serving', status='PASS', dtype=dtype,
                shape=[params['concurrency']], params=params)


def check_plan(config, lock, prompts):
    """Text AgentX probes cover locked trace lengths without imposing an op list."""
    if not prompts or any(not tokens for tokens in prompts):
        raise ValueError('Model correctness prompts must contain tokens')
    requests = lock['request_manifest']['requests']
    indices = lock['correctness']['trace_probe_indices']
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(requests) for i in indices):
        raise ValueError('Invalid protected trace correctness coverage')
    batch = lock['correctness'].get('trace_batch_indices')
    concurrency = int(config['envs']['CONC'])
    if (not isinstance(batch, list) or concurrency <= 0 or len(batch) != concurrency
            or any(type(i) is not int or not 0 <= i < len(requests) for i in batch)):
        raise ValueError('Invalid protected concurrent trace correctness coverage')
    def inputs_for(selected):
        expanded = []
        for index in selected:
            tokens = prompts[index % len(prompts)]
            length = requests[index]['input_tokens']
            expanded.append((tokens * ((length + len(tokens) - 1)//len(tokens)))[:length])
        return expanded
    groups = []
    for index in indices:
        record = requests[index]
        groups.append(dict(kind='trace_shape', concurrency=1, inputs=inputs_for([index]),
                           outputs=[record['output_tokens']]))
    groups.append(dict(kind='trace_batch', concurrency=concurrency, inputs=inputs_for(batch),
                       outputs=[requests[index]['output_tokens'] for index in batch]))
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
    """Validate completed replay records before native phase-window accounting."""
    if __package__:
        from ...measurement import trace_input_token_policy
    else:
        from _aka_measurement import trace_input_token_policy
    input_policy = trace_input_token_policy(manifest)
    expected = {row['id']: row for row in manifest['requests']}
    if len(expected) != len(manifest['requests']):
        raise ValueError('Trace manifest contains duplicate turn IDs')
    inputs, outputs, ids, starts, ends, nominal_inputs = [], [], [], [], [], []
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
        # Native IDs include subagent and flattened-branch identities after
        # "::". They are not replay-session suffixes: stripping them aliases
        # distinct agents to their parent and can accept an unlisted child.
        conversation = meta['conversation_id']
        if not isinstance(conversation, str) or not conversation:
            raise ValueError('Trace replay omitted its native conversation identity')
        turn = str(meta['turn_index'])
        target = expected.get(conversation + ':' + turn)
        if target is None:
            raise ValueError('Replay used a turn outside the frozen trace corpus')
        def metric(name):
            result = record['metrics'].get(name)
            value = result['value'] if isinstance(result, dict) else result
            # Native exporters may encode an integer token count as 8.0.
            # Accept that exact representation, never round fractional work.
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value <= 0 or int(value) != value):
                raise ValueError('Trace token work must be an exact positive integer')
            return int(value)
        incoming, outgoing = metric('input_sequence_length'), metric('output_sequence_length')
        # A native reconstruction budget counts token IDs before decoding
        # disjoint corpus blocks into text and applying chat formatting. BPE
        # re-encoding can merge or split them; it has no fixed ordering with
        # the wire count. Keep both domains, rather than inventing a tolerance.
        if ((input_policy == 'exact' and incoming != target['input_tokens'])
                or outgoing != target['output_tokens']):
            raise ValueError('Replay changed per-turn token work')
        if input_policy == 'native_reconstruction' and (
                metric('usage_prompt_tokens') != incoming
                or metric('usage_completion_tokens') != outgoing):
            raise ValueError('Native usage counters contradict per-turn token work')
        inputs.append(incoming); outputs.append(outgoing)
        nominal_inputs.append(target['input_tokens'])
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
    result = dict(duration_s=duration, completed_requests=len(ids), input_tokens=sum(inputs),
                output_tokens=sum(outputs), output_tokens_per_s=sum(outputs)/duration,
                input_lens=inputs, output_lens=outputs, request_ids=ids, failed_requests=0)
    if input_policy == 'native_reconstruction':
        result.update(nominal_input_lens=nominal_inputs, nominal_input_tokens=sum(nominal_inputs))
    return result


def verify_trace_export(aggregate, metrics):
    """Reconcile native counters so missing JSONL rows cannot alter the work."""
    if not isinstance(aggregate, dict) or aggregate.get('was_cancelled') is not False:
        raise ValueError('Canonical replay did not finish normally')
    verdict = aggregate.get('metadata')
    if (not isinstance(verdict, dict) or verdict.get('scenario') != 'inferencex-agentx-mvp'
            or verdict.get('submission_valid') is not True
            or verdict.get('submission_invalid_reasons', []) != []):
        raise ValueError('Canonical replay lacks an accepted native scenario verdict')
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


def trace_measurement(records, manifest, aggregate):
    """Reconcile native work and retain the entire native observation window.

    First-request/last-response span omits initial idle time and terminal
    grace/drain time. The native client includes those in throughput, even
    though its separately exported benchmark_duration is only a request span.
    """
    metrics = trace_metrics(records, manifest)
    verify_trace_export(aggregate, metrics)
    if aggregate.get('error_summary', []) != []:
        raise ValueError('Native trace export contains request errors')

    def boundary(name):
        value = aggregate.get(name)
        if not isinstance(value, str) or 'T' not in value:
            raise ValueError('Missing native trace observation window')
        try:
            stamp = datetime.fromisoformat(value)
            # The fixed client runtime exports naive UTC datetimes.
            stamp = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
            delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('Invalid native trace observation window') from error
        return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000

    start, end = boundary('start_time'), boundary('end_time')
    selected = [r for r in records if r.get('metadata', {}).get('benchmark_phase') in (None, 'profiling')]
    first = min(r['metadata']['request_start_ns'] for r in selected)
    last = max(r['metadata']['request_end_ns'] for r in selected)
    # Native JSON datetime serialization rounds nanosecond clocks to microseconds.
    if end <= start or first < start - 1000 or last > end + 1000:
        raise ValueError('Native trace records lie outside the observation window')
    duration = (end - start) / 1e9
    for name, expected in (('total_isl', metrics['input_tokens']), ('total_osl', metrics['output_tokens'])):
        field = aggregate.get(name)
        value = field.get('avg') if isinstance(field, dict) else None
        if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
            raise ValueError('Native trace token totals disagree with completed records')
    field = aggregate.get('output_token_throughput')
    rate = field.get('avg') if isinstance(field, dict) else None
    calculated = metrics['output_tokens'] / duration
    if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
            or not math.isclose(rate, calculated, rel_tol=1e-8)):
        raise ValueError('Native trace throughput contradicts its observation window')
    metrics.update(request_span_s=metrics['duration_s'], duration_s=duration,
        output_tokens_per_s=calculated, observation_start_ns=start, observation_end_ns=end)
    return metrics


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
        export_bytes = artifact.read_bytes()
        records_bytes = (root/'aiperf_artifacts/profile_export.jsonl').read_bytes()
        aggregate = json.loads(export_bytes)
        records = [json.loads(line) for line in records_bytes.splitlines()]
        metrics = trace_measurement(records, lock['request_manifest'], aggregate)
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
    if lock['scenario'] == 'agentx':
        metadata['native_replay_evidence'] = dict(
            export_sha256=hashlib.sha256(export_bytes).hexdigest(),
            records_sha256=hashlib.sha256(records_bytes).hexdigest(),
            export_summary={name: aggregate.get(name) for name in (
                'metadata', 'was_cancelled', 'error_summary', 'start_time', 'end_time',
                'request_count', 'total_isl', 'total_osl', 'output_token_throughput')})
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
        sources.extend(client_review_sources(root, lock))
        metadata.update(benchmark_sources=sources,
            client_log=(root/'client.log').read_text()[-24000:])
    return row, metadata
