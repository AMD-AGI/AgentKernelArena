"""Self-contained serving adapter, materialized beside the task runner.

The protected task owns the workload and integration. Runtime assets are pinned,
prepared by the host, and mounted read-only; no downloads occur during scoring.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ModelQualityFailure(ValueError):
    pass


class ModelNumericalFailure(ValueError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PRECISIONS = {'fp16': 'float16', 'bf16': 'bfloat16', 'fp8': 'float8'}


def normalize_executable_path():
    """Keep image tool precedence while removing directories this UID cannot search.

    Root-built images can include private home directories in PATH. An absent
    optional tool then raises PermissionError instead of FileNotFoundError,
    defeating framework probes that correctly handle missing tools on ROCm.
    """
    os.environ['PATH'] = os.pathsep.join(
        entry for entry in os.get_exec_path()
        if os.path.isdir(entry or '.') and os.access(entry or '.', os.X_OK)
    )


def declared_timeout(role, action):
    """The task's own action deadline; the host executor caps requests by it anyway."""
    evaluation = yaml.safe_load((ROOT / 'config.yaml').read_text())['evaluation']
    default = evaluation.get('timeout_s', 3600)
    if role == 'task':
        return evaluation.get('task', {}).get('timeout_s', default)
    return evaluation.get(role, {}).get(action, {}).get('timeout_s', default)


def bridge(args):
    # Agent-invoked checks use the same clean service as formal evaluation.
    timeout = declared_timeout(args.role, args.action)
    request = dict(task_id=json.loads((ROOT / 'runtime.lock.json').read_text())['task_id'],
                   workspace=str(ROOT), role=args.role, action=args.action,
                   phase='candidate_evaluation', timeout=timeout)
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout + 30)
        client.connect(os.environ['AKA_SERVING_SOCKET'])
        client.sendall((json.dumps(request) + '\n').encode())
        result = json.loads(client.makefile('rb').readline(64 * 1024 * 1024))
    if 'error' in result:
        raise RuntimeError(result['error'])
    print(result['stdout'], end='')
    print(result['stderr'], end='', file=sys.stderr)
    return result['returncode']


def identity(config):
    env = config['envs']
    if config['precision'] not in PRECISIONS:
        raise ValueError(f"Unsupported serving precision {config['precision']!r}; declare its dtype before use")
    count = int(env.get('NUM_PROMPTS', int(env['CONC']) * 10))
    params = dict(num_requests=count, concurrency=int(env['CONC']), tp=int(env['TP']))
    ratio = float(env.get('RANDOM_RANGE_RATIO', 1))
    if not 0 < ratio <= 1:
        raise ValueError('Random length ratio must be in (0, 1]')
    if ratio == 1:
        params.update(output_tokens_per_request=int(env['OSL']), input_tokens=int(env['ISL']))
    else:
        # The pinned InferenceX random generator draws input lengths, then
        # output lengths, with seed 0 and no shared prefix or chat template.
        import numpy as np
        generator = np.random.RandomState(0)
        for field, length in [('input_lens', int(env['ISL'])), ('output_lens', int(env['OSL']))]:
            params[field] = generator.randint(int(length * ratio), length + 1, size=count).tolist()
    return dict(test_case_id='serving', status='PASS', dtype=PRECISIONS[config['precision']],
                shape=[int(env['CONC']), int(env['ISL']), int(env['OSL'])],
                params=params)


def operator_checks(lock):
    from checks import operator_checks as check
    import torch
    if torch.cuda.device_count() != lock["gpu_count"]:
        raise ValueError("Visible GPU count differs from runtime lock")
    return check(lock)


def model_inputs(lock):
    from transformers import AutoTokenizer
    prompts = json.loads((ROOT / 'correctness.json').read_text())
    if len(prompts) != lock['correctness']['model_samples']:
        raise ValueError('Model correctness sample count changed')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    return [tokenizer.encode(text) for text in prompts]


def model_reference(sequences, *, with_best=False):
    import torch
    torch.set_num_threads(8)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(ROOT / 'model', torch_dtype=torch.float32,
                                               local_files_only=True, attn_implementation='eager').cuda().eval()
    outputs = []
    with torch.inference_mode():
        for tokens in sequences:
            ids = torch.tensor([tokens], device='cuda')
            logits = model(ids).logits[0, :-1].float().log_softmax(-1)
            values = logits.gather(1, ids[0, 1:, None]).flatten().cpu().tolist()
            outputs.append(dict(values=values, best=logits.max(-1).values.cpu().tolist())
                           if with_best else values)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def logprob_error(tokens, expected, rows):
    if len(rows) != len(tokens) or not rows or rows[0][:2] != [None, tokens[0]]:
        raise ModelNumericalFailure('Model logprob coverage differs from reference')
    if [row[1] for row in rows] != tokens or len(expected) != len(tokens) - 1:
        raise ModelNumericalFailure('Model logprob tokens differ from the scored sequence')
    actual = [float(row[0]) for row in rows[1:]]
    if not all(math.isfinite(v) for v in [*actual, *expected]):
        raise ModelNumericalFailure('Model logprob values must be finite')
    if any(v > 0 for v in [*actual, *expected]):
        raise ModelNumericalFailure('Model logprob values must be nonpositive')
    return max(abs(a-b) for a,b in zip(actual, expected))


def generation_sample(port, tokens, output_length, request_fn=None, *, return_logprob=True):
    body = dict(input_ids=tokens, return_logprob=return_logprob,
                sampling_params=dict(temperature=0, max_new_tokens=output_length, ignore_eos=True))
    if return_logprob:
        body['logprob_start_len'] = 0
    result = (request_fn or request)(port, 'generate', body)
    generated = result['output_ids']
    if not isinstance(generated, list) or len(generated) != output_length:
        raise ModelNumericalFailure('Model correctness decode length changed')
    if any(type(token) is not int or token < 0 for token in generated):
        raise ModelNumericalFailure('Model correctness returned invalid token IDs')
    if not return_logprob:
        return dict(tokens=tokens + generated)
    rows = result['meta_info']['input_token_logprobs'] + result['meta_info']['output_token_logprobs']
    sequence = tokens + generated
    # Validate every returned token and probability before retaining the sample.
    logprob_error(sequence, [float(row[0]) for row in rows[1:]], rows)
    return dict(tokens=sequence, actual=rows)


def model_check_plan(config, lock, prompts, *, params=None):
    """Keep short smoke cases and cover one full batch of the scored workload."""
    if not prompts or any(not tokens for tokens in prompts):
        raise ValueError('Model correctness prompts must contain tokens')
    smoke = dict(kind='smoke', concurrency=1, inputs=prompts,
                 outputs=[lock['correctness']['output_tokens']] * len(prompts))
    params = identity(config)['params'] if params is None else params
    count = params['concurrency']
    inputs = params.get('input_lens', [params.get('input_tokens')] * count)[:count]
    outputs = params.get('output_lens', [params.get('output_tokens_per_request')] * count)[:count]
    if len(inputs) != count or len(outputs) != count or any(type(n) is not int or n <= 0 for n in inputs + outputs):
        raise ValueError('Scored workload must provide one complete correctness batch')
    expanded = []
    for index, length in enumerate(inputs):
        tokens = prompts[index % len(prompts)]
        expanded.append((tokens * ((length + len(tokens) - 1) // len(tokens)))[:length])
    return [smoke, dict(kind='workload', concurrency=count, inputs=expanded, outputs=outputs)]


def generation_checks(port, plan, request_fn, *, return_logprob=True):
    samples = []
    for group in plan:
        barrier = threading.Barrier(group['concurrency']) if group['concurrency'] > 1 else None
        def sample(item):
            tokens, length = item
            if barrier is not None:
                barrier.wait(timeout=60)
            result = generation_sample(port, tokens, length, request_fn, return_logprob=return_logprob)
            result.update(input_tokens=tokens, output_length=length,
                          check_kind=group['kind'], concurrency=group['concurrency'],
                          return_logprob=return_logprob)
            return result
        requests = list(zip(group['inputs'], group['outputs']))
        if barrier is None:
            samples.extend(map(sample, requests))
        else:
            if len(requests) != group['concurrency']:
                raise ValueError('Concurrent correctness group must fill its declared batch')
            with ThreadPoolExecutor(max_workers=group['concurrency']) as pool:
                samples.extend(pool.map(sample, requests))
    return samples


def greedy_reference_error(sample, reference, tolerance):
    """Check each no-logprob generated token against the independent FP32 argmax.

    Near ties use the same locked logprob tolerance as numerical comparisons.
    Teacher forcing checks every actual continuation, so an early mismatch does
    not make later positions incomparable or skip the remaining decode steps.
    """
    values, best = reference['values'], reference['best']
    count = len(sample['tokens']) - 1
    start = len(sample['input_tokens']) - 1
    if len(values) != count or len(best) != count or count - start != sample['output_length']:
        raise ModelNumericalFailure('No-logprob reference coverage changed')
    if not all(math.isfinite(value) for value in values + best):
        raise ModelNumericalFailure('No-logprob reference contains nonfinite values')
    error = max(b - value for b, value in zip(best[start:], values[start:]))
    if error > tolerance:
        raise ModelNumericalFailure('No-logprob generated token differs from the reference argmax')
    return error


def production_tolerances(lock):
    """A frozen, task-owned policy calibrated before candidate optimization."""
    names = {'prefix_max_abs', 'prefix_mean_abs', 'prefix_p99_abs',
             'decode_max_regret', 'decode_mean_regret', 'decode_p99_regret'}
    rule = lock['correctness'].get('production_tolerances')
    if not isinstance(rule, dict) or set(rule) != names or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in rule.values()):
        raise ValueError('Production reference requires complete finite calibrated tolerances')
    return rule


def error_statistics(values):
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ModelNumericalFailure('Model comparison requires finite nonnegative errors')
    ordered = sorted(values)
    return dict(max=max(values), mean=math.fsum(values) / len(values),
                p99=ordered[math.ceil(.99 * len(ordered)) - 1], count=len(values))


def production_logprob_tolerances(lock):
    """An explicit decode budget, or the existing stricter prefix error budget.

    Both are frozen before search. An older binding never silently disables
    decode comparison; its initial baseline must also satisfy these bounds.
    """
    prefix = production_tolerances(lock)
    rule = lock['correctness'].get('decode_logprob_tolerances')
    if 'decode_logprob_tolerances' not in lock['correctness']:
        return {name+'_abs': prefix['prefix_'+name+'_abs'] for name in ('max', 'mean', 'p99')}
    if (not isinstance(rule, dict) or set(rule) != {'max_abs', 'mean_abs', 'p99_abs'}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in rule.values())):
        raise ValueError('Decode logprob comparison requires complete finite calibrated tolerances')
    return rule


def production_teacher_reference(port, sample, request_fn):
    """Score every actual continuation on the independent original.

    FP8 prefill and recurrent decode need not be bit-identical. The locked
    production policy bounds this difference; it is not an FP32 oracle.
    """
    tokens = sample['tokens']
    response = request_fn(port, 'generate', dict(input_ids=tokens,
        return_logprob=True, logprob_start_len=0, top_logprobs_num=1,
        sampling_params=dict(temperature=0, max_new_tokens=1, ignore_eos=True)))
    meta = response['meta_info']
    rows, tops = meta['input_token_logprobs'], meta['input_top_logprobs']
    logprob_error(tokens, [float(row[0]) for row in rows[1:]], rows)
    if len(tops) != len(tokens) or any(not isinstance(top, list) or len(top) != 1
            or not isinstance(top[0], list) or len(top[0]) < 2
            or type(top[0][1]) is not int or top[0][1] < 0
            for top in tops[1:]):
        raise ModelNumericalFailure('Original top-logprob coverage changed')
    values = [float(row[0]) for row in rows[1:]]
    best = [float(top[0][0]) for top in tops[1:]]
    if not all(math.isfinite(v) for v in values + best):
        raise ModelNumericalFailure('Original teacher-forcing values must be finite')
    if any(v > 0 for v in values + best):
        raise ModelNumericalFailure('Original teacher-forcing logprobs must be nonpositive')
    return dict(values=values, best=best)


def production_references(port, diagnostics, plain_samples, request_fn):
    """Use the original to score the exact tokens observed in both API modes.

    Independently generated continuations can have different prefixes. Their
    logprobs cannot replace a reference for the actual diagnostic decode.
    """
    for samples in (diagnostics, plain_samples):
        start = 0
        while start < len(samples):
            first = samples[start]
            concurrency = first.get('concurrency', 1)
            if type(concurrency) is not int or concurrency <= 0:
                raise ModelNumericalFailure('Original reference concurrency must be positive')
            end = start+1
            while (end < len(samples) and samples[end].get('check_kind') == first.get('check_kind')
                    and samples[end].get('concurrency', 1) == concurrency):
                end += 1
            group = samples[start:end]
            if concurrency > 1 and len(group) != concurrency:
                raise ModelNumericalFailure('Original reference batch coverage changed')
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                references = list(pool.map(
                    lambda item: production_teacher_reference(port, item, request_fn), group))
            for sample, reference in zip(group, references):
                sample['reference'] = reference
            start = end
    references = []
    for item in diagnostics:
        values = item['reference']['values']
        item['reference_rows'] = [[None, item['tokens'][0]]] + [
            [value, token] for value, token in zip(values, item['tokens'][1:])]
        references.append(values)
    return references


def production_numerical_checks(diagnostics, plain_samples, lock):
    rule = production_tolerances(lock)
    logprob_rule = production_logprob_tolerances(lock)
    if not diagnostics or len(diagnostics) != len(plain_samples):
        raise ModelNumericalFailure('Model comparison sample coverage changed')
    prefix, decode, diagnostic_decode, decode_logprobs = [], [], [], []
    for item, plain in zip(diagnostics, plain_samples):
        tokens = item['input_tokens']
        if plain['input_tokens'] != tokens:
            raise ModelNumericalFailure('Model comparison prompt identities changed')
        rows = item['actual'][:len(tokens)]
        reference_rows = item['reference_rows'][:len(tokens)]
        expected = [float(row[0]) for row in reference_rows[1:]]
        logprob_error(tokens, expected, reference_rows)
        logprob_error(tokens, expected, rows)
        prefix.append(error_statistics([abs(float(row[0]) - value)
            for row, value in zip(rows[1:], expected)]))
        reference = item.get('reference', {})
        values, best = reference.get('values', []), reference.get('best', [])
        sequence, length = item.get('tokens', []), item.get('output_length')
        if (type(length) is not int or length <= 0 or sequence[:len(tokens)] != tokens
                or len(sequence) != len(tokens) + length or len(values) != len(sequence)-1
                or len(best) != len(values)):
            raise ModelNumericalFailure('Diagnostic decode reference coverage changed')
        logprob_error(sequence, values, item['actual'])
        if not all(math.isfinite(v) and v <= 0 for v in best):
            raise ModelNumericalFailure('Diagnostic decode reference logprobs must be finite and nonpositive')
        start = len(tokens)-1
        decode_logprobs.append(error_statistics([abs(float(row[0])-value)
            for row, value in zip(item['actual'][len(tokens):], values[start:])]))
        diagnostic_decode.append(error_statistics([max(0., b-v)
            for v, b in zip(values[start:], best[start:])]))
        values, best = plain['reference']['values'], plain['reference']['best']
        start = len(tokens) - 1
        if (len(values) != len(plain['tokens']) - 1 or len(best) != len(values)
                or len(values) - start != plain['output_length']):
            raise ModelNumericalFailure('No-logprob reference coverage changed')
        if not all(math.isfinite(v) and v <= 0 for v in values + best):
            raise ModelNumericalFailure('No-logprob reference contains nonfinite or positive values')
        decode.append(error_statistics([max(0., b - v)
            for v, b in zip(values[start:], best[start:])]))
    # Bound every case, not only an average diluted by thousands of easy tokens.
    for label, cases, suffix in (('prefix', prefix, 'abs'), ('decode', decode, 'regret'),
                                 ('diagnostic_decode', diagnostic_decode, 'regret')):
        for index, metrics in enumerate(cases):
            for metric in ('max', 'mean', 'p99'):
                policy = 'decode' if label == 'diagnostic_decode' else label
                if metrics[metric] > rule[f'{policy}_{metric}_{suffix}']:
                    raise ModelNumericalFailure(
                        f'Model {label} case {index} {metric} error exceeds the locked threshold: '
                        f'{metrics[metric]} > {rule[f"{policy}_{metric}_{suffix}"]}')
    for index, metrics in enumerate(decode_logprobs):
        for metric in ('max', 'mean', 'p99'):
            if metrics[metric] > logprob_rule[metric+'_abs']:
                raise ModelNumericalFailure(
                    f'Model decode logprob case {index} {metric} error exceeds the locked threshold: '
                    f'{metrics[metric]} > {logprob_rule[metric+"_abs"]}')
    return dict(prefix_cases=prefix, decode_cases=decode, diagnostic_decode_cases=diagnostic_decode,
                decode_logprob_cases=decode_logprobs, thresholds=rule, decode_logprob_thresholds=logprob_rule)


def request(port, path, value=None, *, timeout=120):
    data = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/{path}', data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def deadline_client(timeout_s):
    """Share one client budget across queued and concurrent correctness requests.

    First-use compilation can exceed a short HTTP timeout. Allow the declared
    client budget without retrying failed samples or resetting it per request.
    The evaluator's outer action deadline also bounds the server and this client.
    """
    deadline = time.monotonic() + timeout_s

    def send(port, path, value=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Model correctness client deadline exceeded')
        result = request(port, path, value, timeout=remaining)
        if time.monotonic() >= deadline:
            raise TimeoutError('Model correctness client deadline exceeded')
        return result

    return send


def prepare_client(lock):
    source = ROOT / 'runtime' / 'inferencex' / lock['dependencies']['inferencex']['subdirectory']
    dest = ROOT / 'client'
    if not dest.exists():
        shutil.copytree(source, dest)
        for script in (ROOT / 'runtime/magpie/Magpie/scripts/benchmark').glob('*.sh'):
            shutil.copy2(script, dest / 'benchmarks' / script.name)
        if lock.get('catalog_version'):
            from _aka_serving_scenarios import compose_client
            compose_client(dest, lock, ROOT/'scripts/_aka_serving_request_plan.py')
    scripts = {str(p.relative_to(dest)): digest(p) for p in (dest / 'benchmarks').glob('*.sh')}
    if scripts != lock['benchmark_scripts']:
        raise ValueError('Composed benchmark scripts differ from runtime lock')
    if lock.get('catalog_version'):
        from _aka_serving_scenarios import composed_script
        expected = {str(p.relative_to(source)): digest(p) for p in source.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts}
        for script in (ROOT/'runtime/magpie/Magpie/scripts/benchmark').glob('*.sh'):
            expected['benchmarks/'+script.name] = digest(script)
        if lock['scenario'] == 'multimodal':
            name = 'benchmarks/'+lock['benchmark_script']
            path = ROOT/'runtime/magpie/Magpie/scripts/benchmark'/lock['benchmark_script']
            expected[name] = hashlib.sha256(composed_script(path.read_bytes(), name, lock)).hexdigest()
        elif lock['scenario'] == 'text':
            name = 'infx/bench_serving/benchmark_serving.py'
            expected[name] = hashlib.sha256(composed_script((source/name).read_bytes(), name, lock)).hexdigest()
            expected['infx/bench_serving/_aka_request_plan.py'] = digest(ROOT/'scripts/_aka_serving_request_plan.py')
        actual = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        if actual != expected:
            raise ValueError('Timed client source tree differs from pinned runtime sources')
    return dest


def gpu_memory_used_mib():
    """Query from a child process so no HIP context of ours lingers beside the server."""
    code = ('import json, torch; print(json.dumps([int((total - free) // 2**20) for free, total in '
            '(torch.cuda.mem_get_info(index) for index in range(torch.cuda.device_count()))]))')
    completed = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=120)
    if completed.returncode:
        raise RuntimeError('GPU idle check failed: ' + completed.stderr[-2000:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def assert_gpu_idle(limits):
    """Fail closed when another process already holds the measured device."""
    used = gpu_memory_used_mib()
    if not used:
        raise RuntimeError('No visible GPU for the serving measurement')
    limit = limits['idle_vram_limit_mib']
    if any(value > limit for value in used):
        raise RuntimeError(f'GPU memory in use before measurement: {used} MiB exceeds {limit} MiB; '
                           'another process holds the device')
    return used


def launch_command(config, port):
    """Framework-specific server argv and readiness endpoint for the pinned image."""
    env = config['envs']
    import checks
    if hasattr(checks, 'launch_command'):
        return checks.launch_command(config, port)
    if config['framework'] == 'sglang':
        command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(ROOT / 'model'),
                   '--host', '127.0.0.1', '--port', str(port), '--tensor-parallel-size', str(env['TP']),
                   '--context-length', str(env['MAX_MODEL_LEN']), *shlex.split(env['EXTRA_SGLANG_ARGS'])]
        lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        if 'server_seed' in lock:
            command.extend(['--random-seed', str(lock['server_seed'])])
        return command, 'get_model_info'
    raise ValueError(f"Unsupported serving framework {config['framework']!r}; add a launcher and hook before use")


def benchmark_sources(client, config):
    """Make the executed client boundary inspectable in trusted validation evidence."""
    sources = []
    for relative in ('benchmarks/' + config['benchmark_script'], 'benchmarks/benchmark_lib.sh',
                     'infx/bench_serving/benchmark_serving.py'):
        path = client / relative
        text = path.read_text()
        start_line = 1
        if relative.endswith('benchmark_lib.sh'):
            start = text.index('run_benchmark_serving() {')
            end = text.index('\n}\n', start) + 2
            start_line = text[:start].count('\n') + 1
            text = text[start:end]
        sources.append(dict(path=relative, sha256=digest(path), start_line=start_line, source=text))
    return sources


def serving_port():
    # Model loading leaves a long gap before bind. Avoid ports the OS may assign
    # to readiness probes or distributed initialization during that interval.
    try:
        low, high = map(int, Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    except OSError:
        low, high = 32768, 65535
    ports = [p for p in range(10240, 55001)
             if not low <= p <= high and not low <= p + 10000 <= high]
    random.SystemRandom().shuffle(ports)
    for port in ports[:128]:
        try:
            # The pinned framework also derives an auxiliary port as HTTP+10000.
            with socket.socket() as http, socket.socket() as auxiliary:
                http.bind(('127.0.0.1', port))
                auxiliary.bind(('127.0.0.1', port + 10000))
            return port
        except OSError:
            continue
    raise RuntimeError('No serving port pair available outside the ephemeral range')


def server_log_tail(path):
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 12000))
        return stream.read().decode(errors='replace')


def readiness_probe(port, path, *, timeout):
    # Native health endpoints may return an empty or plain-text 2xx response.
    # JSON parsing remains mandatory for actual inference in request().
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/{path}', timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise OSError(f'Serving health endpoint returned HTTP {response.status}')


def start_server(config, limits, *, install_kernel=True):
    port = serving_port()
    env = dict(os.environ)
    env.update({key: str(value) for key, value in config['envs'].items()})
    env.update(AKA_INSTALL_KERNEL='1' if install_kernel else '0', PYTHONPATH=str(ROOT / 'scripts'),
               NO_PROXY='127.0.0.1,localhost,0.0.0.0', no_proxy='127.0.0.1,localhost,0.0.0.0')
    checks = sys.modules.get('checks')
    if hasattr(checks, 'server_environment'):
        env.update(checks.server_environment(config, port, install_kernel))
    command, ready_endpoint = launch_command(config, port)
    label = 'server' if install_kernel else 'reference_server'
    (ROOT / (label + '_command.json')).write_text(json.dumps(command))
    log_path = ROOT / (label + '.log')
    log = log_path.open('w')
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        startup = limits['server_startup_timeout_s']
        deadline = time.monotonic() + startup
        while time.monotonic() < deadline:
            tail = server_log_tail(log_path)
            if process.poll() is not None:
                raise RuntimeError('Serving process exited: ' + tail)
            if 'address already in use' in tail.lower():
                raise RuntimeError('Serving port bind failed: ' + tail)
            try:
                readiness_probe(port, ready_endpoint, timeout=min(5, max(0.001, deadline - time.monotonic())))
                return process, port, log
            except (OSError, ValueError):
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        raise TimeoutError(f'Model startup exceeded {startup} seconds')
    except BaseException:
        stop_server(process, log)
        raise


def stop_server(process, log):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    log.close()


def audit(lock):
    from _aka_serving_bundle import audit as audit_bundle
    return audit_bundle(ROOT, lock)


def runtime_identity(lock_digest, calls):
    settings, environment = [], []
    for call in calls:
        if 'observer' in call and not call['observer']:
            continue
        actual = call.get('server_settings')
        if not isinstance(actual, dict) or not actual:
            raise ValueError('A server rank did not expose effective launch settings')
        stable = {k:v for k,v in actual.items()
                  if k not in ('port', 'nccl_port', 'grpc_port', '_runtime_instance_id')}
        if call.get('backend') == 'vllm' and isinstance(stable.get('cache_config'), dict):
            # Derived allocator capacity can change with an implementation's
            # workspace footprint. Keep it in raw rank evidence; configured
            # overrides, block size and cache precision remain compared.
            stable['cache_config'] = {k:v for k,v in stable['cache_config'].items()
                                     if k not in ('num_gpu_blocks', 'num_cpu_blocks')}
        if call.get('backend') in {'vllm', 'atom'} and isinstance(stable.get('compilation_config'), dict):
            # Source changes legitimately alter these init=False observations.
            # Preserve them in raw rank evidence, while comparing all configured
            # compilation/graph/backend/custom-op switches unchanged.
            derived = {'compilation_time'}
            if call['backend'] == 'vllm':
                derived.update(('traced_files', 'encoder_compilation_time',
                                'enabled_custom_ops', 'disabled_custom_ops'))
            stable['compilation_config'] = {k:v for k,v in stable['compilation_config'].items() if k not in derived}
            inputs = call.get('configuration_inputs')
            if isinstance(inputs, dict) and isinstance(inputs.get('compilation_cache_dir'), str):
                # Native vLLM resolves the empty cache input to a directory keyed
                # by traced source. Compare its root and configured input, while
                # retaining the complete effective path in raw rank evidence.
                stable['_configuration_inputs'] = dict(inputs)
                cache = Path(stable['compilation_config'].get('cache_dir', ''))
                if (inputs['compilation_cache_dir'] == '' and cache.parent.name == 'torch_compile_cache'
                        and re.fullmatch(r'[0-9a-f]{10}', cache.name)):
                    stable['compilation_config']['cache_dir'] = str(cache.parent/'<source-key>')
        if call.get('backend') == 'atom' and isinstance(stable.get('parallel_config'), dict):
            # Native ATOM fills these transport-only fields at startup. Retain
            # raw addresses and ports in rank evidence, not model configuration
            # equality. DP/TP/EP/PP, master address/port and frozen env remain.
            stable['parallel_config'] = {k:v for k,v in stable['parallel_config'].items()
                if k not in ('control_address', 'pp_meta_addrs', 'pp_token_addr',
                             'pp_kv_status_addr', 'data_parallel_base_port')}
        settings.append(stable)
        # Keep the raw run identifier in the audit log, but not in configuration
        # equality: the framework generates a new diagnostic ID on every start.
        environment.append({k:v for k,v in call['runtime_settings'].items() if k != 'SGLANG_RUN_ID'})
    fingerprint = hashlib.sha256(json.dumps(dict(lock=lock_digest, settings=settings,
        rank_settings=environment), sort_keys=True).encode()).hexdigest()
    return fingerprint, settings


def fp32_token_checks(samples, lock):
    """Teacher-force every actual native token on the independent FP32 model.

    Unlike exact production-token equality, this explicitly locked policy accepts
    near-tied greedy choices within the declared log-probability regret bound.
    It never invents engine logprobs or falls back after a failed comparison.
    """
    if not samples or any(item.get('return_logprob') is not False for item in samples):
        raise ModelNumericalFailure('Native token reference requires actual no-logprob samples')
    reference = model_reference([item['tokens'] for item in samples], with_best=True)
    if len(reference) != len(samples):
        raise ModelNumericalFailure('Native token reference has incomplete request coverage')
    for item, expected in zip(samples, reference):
        item['reference'] = expected
    # Preserve continuations and reference values even if the numeric gate fails.
    (ROOT/'no_logprob_correctness.json').write_text(json.dumps(samples, indent=2))
    errors = [greedy_reference_error(item, item['reference'], lock['correctness']['model_logprob_atol'])
              for item in samples]
    return dict(policy='transformers_fp32_tokens', samples=len(samples),
                generated_tokens_checked=sum(item['output_length'] for item in samples),
                max_greedy_regret=max(errors), logprob_checks=False, reference_dtype='float32')


def exact_token_checks(samples, reference):
    """Compare every native token and request shape, without re-tokenizing text."""
    if not samples or len(samples) != len(reference):
        raise ModelNumericalFailure('Native token reference has incomplete request coverage')
    checked = 0
    for item, expected in zip(samples, reference):
        for key in ('input_tokens', 'output_length', 'check_kind', 'concurrency', 'tokens'):
            if item.get(key) != expected.get(key):
                raise ModelNumericalFailure('Native token output differs from the original serving reference: ' + key)
        if item.get('return_logprob') is not False or expected.get('return_logprob') is not False:
            raise ModelNumericalFailure('Native token reference must exercise the actual no-logprob path')
        checked += item['output_length']
        item['reference_tokens'] = expected['tokens']
    return dict(policy='exact_native_tokens', samples=len(samples), generated_tokens_checked=checked,
                logprob_checks=False)


def evaluate(args, config, lock, limits):
    from checks import source_policy
    source_policy()
    from _aka_serving_bundle import Bundle
    bundle = Bundle(ROOT, lock)
    bundle.check_sources()
    row = identity(config)
    import checks
    if lock.get('catalog_version') and lock['scenario'] == 'text' and args.action != 'compile':
        # The initial manifest and all later actions must bind the same realized
        # native requests, not add identity fields after the baseline was locked.
        row = checks.workload_identity(config, lock, limits, row)
    args.case_identity = copy.deepcopy(row)
    if args.action == 'validate-task':
        if digest(ROOT / 'benchmark.yaml') != lock['workload_sha256']:
            raise ValueError('Workload configuration differs from lock')
        initial = bundle.digest
        if initial != lock['initial_bundle_sha256']:
            raise ValueError('Initial implementation differs from the locked production source')
        row['checks'] = ['correctness', 'performance']
        return [row], dict(candidate_state='implemented', initial_source_sha256=initial)
    if args.action == 'compile':
        # Syntax/import checks for every submitted entrypoint. GPU JIT paths
        # are compiled and executed by the mandatory model correctness action.
        import checks
        prepared = checks.prepare_runtime(lock) if hasattr(checks, 'prepare_runtime') else {}
        from _aka_serving_bundle import effective_function, original_sources
        originals = original_sources(bundle)  # Freeze the image before candidate imports.
        for patch in bundle.patches:
            effective_function(ROOT, lock, patch['module'], patch['symbol'])
        return [], dict(**bundle.check_sources(), **prepared, gpu_execution='required_by_correctness',
                        original_compute_sources=originals)
    metadata = {}
    diagnostics = []
    reference_kind = lock['correctness'].get('model_reference', 'transformers_fp32')
    if args.action == 'correctness':
        metadata['operator_cases'] = operator_checks(lock)
        (ROOT / 'operator_correctness.json').write_text(json.dumps(metadata['operator_cases'], indent=2))
        prompts = model_inputs(lock)
        plan = (model_check_plan(config, lock, prompts, params=row['params'])
                if lock.get('catalog_version') and lock['scenario'] == 'text'
                else model_check_plan(config, lock, prompts))
        metadata['model_coverage'] = [dict(kind=group['kind'], concurrency=group['concurrency'],
            return_logprob=mode,
            input_lengths=[len(tokens) for tokens in group['inputs']], output_lengths=group['outputs'])
            for mode in ((False,) if reference_kind in {'production_tokens', 'transformers_tokens'} else (True, False)) for group in plan]
        (ROOT / 'correctness_coverage.json').write_text(json.dumps(metadata['model_coverage'], indent=2))
    client = None
    if args.action == 'performance':
        # Another tenant on the device would distort both roles unequally.
        metadata['gpu_memory_used_before_mib'] = assert_gpu_idle(limits)
        metadata['limits'] = dict(limits)
        client = prepare_client(lock)
    process, port, log = start_server(config, limits)
    try:
        if args.action == 'correctness':
            correctness_request = deadline_client(limits['client_timeout_s'])
            diagnostics = ([] if reference_kind in {'production_tokens', 'transformers_tokens'} else
                           generation_checks(port, plan, correctness_request))
            plain_samples = generation_checks(port, plan, correctness_request, return_logprob=False)
            (ROOT / 'no_logprob_correctness.json').write_text(json.dumps(plain_samples, indent=2))
            if reference_kind == 'production_quality':
                import quality
                candidate_quality = quality.run(port, lock, correctness_request, 'candidate')
        elif hasattr(__import__('checks'), 'performance'):
            import checks
            row, extra = checks.performance(config, lock, limits, client, port, row)
            metadata.update(extra)
        else:
            env = dict(os.environ)
            env.update({key: str(value) for key,value in config['envs'].items()})
            env.update(MODEL=str(ROOT / 'model'), PORT=str(port), RESULT_DIR=str(ROOT), RESULT_FILENAME='inferencex_result',
                       MAGPIE_RUN_PHASE='client', PROFILE='0', RUN_EVAL='false',
                       NO_PROXY='127.0.0.1,localhost,0.0.0.0', no_proxy='127.0.0.1,localhost,0.0.0.0')
            completed = subprocess.run(['bash', 'benchmarks/' + config['benchmark_script']], cwd=client,
                                       env=env, text=True, capture_output=True, timeout=limits['client_timeout_s'])
            (ROOT / 'client.log').write_text(completed.stdout + '\n' + completed.stderr)
            if completed.returncode:
                raise RuntimeError('Benchmark client failed: ' + (ROOT / 'client.log').read_text()[-10000:])
            if os.environ.get('ARENA_EVAL_PHASE') == 'task_validation':
                metadata['benchmark_sources'] = benchmark_sources(client, config)
                metadata['client_log'] = (ROOT / 'client.log').read_text()[-24000:]
            measured = json.loads((ROOT / 'inferencex_result.json').read_text())
            if measured['duration'] < limits['minimum_measurement_s']:
                raise ValueError(f"Measured window {measured['duration']:.1f}s is shorter than the locked minimum "
                                 f"{limits['minimum_measurement_s']}s; raise NUM_PROMPTS in the workload")
            for name, count in (('input_lens', int(config['envs']['ISL'])),
                                ('output_lens', int(config['envs']['OSL']))):
                expected = row['params'].get(name, [count] * row['params']['num_requests'])
                if measured.get(name) != expected:
                    raise ValueError(f'Benchmark changed per-request {name}')
            row.update(execution_time_ms=1000*measured['duration'], benchmark_method='serving_wall_clock',
                       metrics=dict(duration_s=measured['duration'], completed_requests=measured['completed'],
                                    input_tokens=measured['total_input_tokens'],
                                    output_tokens=measured['total_output_tokens'], output_tokens_per_s=measured['output_throughput'],
                                    p99_tpot_ms=measured['p99_tpot_ms']))
            for name in ('input_lens', 'output_lens'):
                if name in row['params']:
                    row['metrics'][name] = measured[name]
        if args.action == 'performance':
            from _aka_measurement import validate_serving_case
            validate_serving_case(row)
            measurement = yaml.safe_load((ROOT/'config.yaml').read_text())['evaluation']['measurement']
            if 'max_p99_tpot_ms' in measurement:
                latency = row['metrics'].get('p99_tpot_ms')
                if latency is None or latency > measurement['max_p99_tpot_ms']:
                    raise ValueError('Serving tail latency is missing or exceeded the task limit')
        metadata['rank_calls'] = sorted(audit(lock), key=lambda row: (row['rank'], row['function']))
        # Read the actual ServerArgs in every worker, without mixing the HTTP
        # server's volatile scheduler counters into the workload identity.
        fingerprint, stable_settings = runtime_identity(digest(ROOT / 'runtime.lock.json'), metadata['rank_calls'])
        metadata['server_settings'] = stable_settings
        row['metadata'] = dict(runtime_fingerprint=fingerprint,
                               kernel_sha256=bundle.digest)
    finally:
        stop_server(process, log)
    if args.action == 'correctness':
        if reference_kind == 'transformers_tokens':
            metadata['native_token_checks'] = fp32_token_checks(plain_samples, lock)
            return [row], metadata
        if reference_kind == 'production_tokens':
            if lock['correctness'].get('token_match') != 'exact':
                raise ValueError('Native token comparison requires its locked exact-equality policy')
            reference_process, reference_port, reference_log = start_server(config, limits, install_kernel=False)
            try:
                plain_reference = generation_checks(reference_port, plan,
                    deadline_client(limits['client_timeout_s']), return_logprob=False)
                (ROOT/'original_token_reference.json').write_text(json.dumps(plain_reference, indent=2))
                metadata['native_token_checks'] = exact_token_checks(plain_samples, plain_reference)
            finally:
                stop_server(reference_process, reference_log)
            (ROOT/'no_logprob_correctness.json').write_text(json.dumps(plain_samples, indent=2))
            (ROOT/'original_token_reference.json').write_text(json.dumps(plain_reference, indent=2))
            return [row], metadata
        # Compare against a separate implementation with task-declared semantics.
        if reference_kind in ('production_serving', 'production_quality', 'production_numerical'):
            # The uninjected original scores the actual observed prefixes. Its
            # prefill and recurrent decode can differ numerically, so the task
            # freezes independently calibrated comparison bounds before search.
            reference_process, reference_port, reference_log = start_server(config, limits, install_kernel=False)
            try:
                reference = []
                reference_request = deadline_client(limits['client_timeout_s'])
                if reference_kind in ('production_quality', 'production_numerical'):
                    production_tolerances(lock)
                    production_logprob_tolerances(lock)
                    reference = production_references(reference_port, diagnostics, plain_samples, reference_request)
                    metadata['model_reference_mode'] = 'teacher_forced_actual_continuations'
                    if reference_kind == 'production_quality':
                        reference_quality = quality.run(reference_port, lock, reference_request, 'reference')
                else:
                    reference_samples = generation_checks(reference_port, plan, reference_request)
                    plain_reference = generation_checks(reference_port, plan, reference_request, return_logprob=False)
                    for item, sample in zip(plain_samples, plain_reference):
                        item['reference_tokens'] = sample['tokens']
                    for item, sample in zip(diagnostics, reference_samples):
                        rows = sample['actual']
                        item['reference_tokens'] = sample['tokens']
                        item['reference_rows'] = rows
                        reference.append([float(row[0]) for row in rows[1:]])
            finally:
                stop_server(reference_process, reference_log)
        elif reference_kind == 'transformers_fp32':
            full_reference = model_reference([item['tokens'] for item in diagnostics + plain_samples], with_best=True)
            reference = [item['values'] for item in full_reference[:len(diagnostics)]]
            for item, expected in zip(plain_samples, full_reference[len(diagnostics):]):
                item['reference'] = expected
        else:
            raise ValueError('Unknown model correctness reference')
        for item, expected in zip(diagnostics, reference):
            item['expected'] = expected
        (ROOT / 'model_correctness.json').write_text(json.dumps(diagnostics, indent=2))
        (ROOT / 'no_logprob_correctness.json').write_text(json.dumps(plain_samples, indent=2))
        if reference_kind == 'transformers_fp32':
            metadata['no_logprob_max_greedy_error'] = max(greedy_reference_error(
                item, item['reference'], lock['correctness']['model_logprob_atol']) for item in plain_samples)
        elif reference_kind == 'production_serving' and any(
                item['tokens'] != item['reference_tokens'] for item in plain_samples):
            raise ModelNumericalFailure('No-logprob generated tokens differ from the original serving reference')
        if reference_kind == 'production_numerical':
            metadata['production_numerical_checks'] = production_numerical_checks(diagnostics, plain_samples, lock)
            return [row], metadata
        if reference_kind == 'production_quality':
            # Normal variation is bounded on both actual continuations,
            # alongside answer accuracy and the independent operator oracle.
            calibration = args.role == 'baseline' and os.environ.get('ARENA_EVAL_PHASE') == 'task_validation'
            (ROOT / 'accuracy_report.json').write_text(json.dumps(dict(
                candidate=candidate_quality, reference=reference_quality,
                baseline_calibration=calibration, thresholds=lock['correctness'].get('quality')), indent=2))
            metadata['production_numerical_checks'] = production_numerical_checks(
                diagnostics, plain_samples, lock)
            try:
                metadata['model_quality'] = quality.validate(candidate_quality, reference_quality, lock)
            except ValueError as exc:
                label = 'Baseline calibration failed' if calibration else 'Candidate model quality failed'
                raise ModelQualityFailure(label + ': ' + str(exc)) from exc
            metadata['baseline_calibration'] = calibration
            return [row], metadata
        if reference_kind == 'production_serving':
            # Fail closed on divergent greedy continuations: logprobs under
            # different prefixes are not comparable. Preserve both sequences.
            for item in diagnostics:
                logprob_error(item['tokens'], item['expected'], item['reference_rows'])
        errors = [logprob_error(item['tokens'], item['expected'], item['actual']) for item in diagnostics]
        if not errors or max(errors) > lock['correctness']['model_logprob_atol']:
            raise ModelNumericalFailure(f'Model logprob error exceeds threshold: {max(errors, default=-1)}')
        metadata['model_max_logprob_error'] = max(errors)
    return [row], metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role_or_action', choices=['validate-task', 'baseline', 'candidate'])
    parser.add_argument('action', nargs='?', choices=['compile', 'correctness', 'performance'])
    args = parser.parse_args()
    if args.role_or_action == 'validate-task':
        args.role, args.action = 'task', 'validate-task'
    else:
        args.role = args.role_or_action
        if args.action is None:
            parser.error('baseline/candidate requires an action')
    if os.environ.get('AKA_SERVING_SOCKET'):
        return bridge(args)
    normalize_executable_path()
    report = dict(protocol='arena-eval-v1', role=args.role, action=args.action, status='PASS', cases=[])
    config = None
    try:
        from _aka_measurement import lock_limits
        config = yaml.safe_load((ROOT / 'benchmark.yaml').read_text())['benchmark']
        lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        if digest(ROOT / 'benchmark.yaml') != lock['workload_sha256']:
            raise ValueError('Workload differs from runtime lock')
        cases, metadata = evaluate(args, config, lock, lock_limits(lock))
        report.update(cases=cases, metadata=metadata)
    except Exception as exc:
        report.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
        report['metadata'] = dict(failure_category=(
            'model_quality' if isinstance(exc, ModelQualityFailure) else
            'model_numerical' if isinstance(exc, (ModelNumericalFailure, AssertionError)) else
            'infrastructure' if isinstance(exc, (OSError, TimeoutError, subprocess.TimeoutExpired)) else
            'task_or_runtime_error'))
        # Keep the known case identity on failures. An empty case list used to
        # mask the original correctness failure as a manifest mismatch.
        if config is not None and args.action in ('correctness', 'performance'):
            try:
                row = copy.deepcopy(getattr(args, 'case_identity', None)) or identity(config)
                row.update(status='FAIL', reason=report['reason'])
                report['cases'] = [row]
            except Exception:
                pass
    print('ARENA_EVAL_RESULT=' + json.dumps(report, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1
