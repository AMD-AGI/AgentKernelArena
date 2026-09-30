"""Self-contained serving adapter, materialized beside the task runner.

The protected task owns the workload and integration. Runtime assets are pinned,
prepared by the host, and mounted read-only; no downloads occur during scoring.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bridge(args):
    # Agent-invoked checks use the same clean service as formal evaluation.
    request = dict(task_id=json.loads((ROOT / 'runtime.lock.json').read_text())['task_id'],
                   workspace=str(ROOT), role=args.role, action=args.action,
                   phase='candidate_evaluation', timeout=1200)
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(1230)
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
    return dict(test_case_id='serving', status='PASS', dtype={'fp16': 'float16', 'bf16': 'bfloat16'}[config['precision']],
                shape=[int(env['CONC']), int(env['ISL']), int(env['OSL'])],
                params=dict(num_requests=int(env['NUM_PROMPTS']),
                            output_tokens_per_request=int(env['OSL']),
                            input_tokens=int(env['ISL']), concurrency=int(env['CONC']),
                            tp=int(env['TP'])))


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


def model_reference(sequences):
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
            outputs.append(values)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def logprob_error(tokens, expected, rows):
    if len(rows) != len(tokens) or not rows or rows[0][:2] != [None, tokens[0]]:
        raise ValueError('Model logprob coverage differs from reference')
    if [row[1] for row in rows] != tokens or len(expected) != len(tokens) - 1:
        raise ValueError('Model logprob tokens differ from the scored sequence')
    actual = [float(row[0]) for row in rows[1:]]
    if not all(math.isfinite(v) for v in [*actual, *expected]):
        raise ValueError('Model logprob values must be finite')
    return max(abs(a-b) for a,b in zip(actual, expected))


def request(port, path, value=None):
    data = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/{path}', data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def prepare_client(lock):
    source = ROOT / 'runtime' / 'inferencex' / lock['dependencies']['inferencex']['subdirectory']
    dest = ROOT / 'client'
    shutil.copytree(source, dest)
    for script in (ROOT / 'runtime/magpie/Magpie/scripts/benchmark').glob('*.sh'):
        shutil.copy2(script, dest / 'benchmarks' / script.name)
    scripts = {str(p.relative_to(dest)): digest(p) for p in (dest / 'benchmarks').glob('*.sh')}
    expected = lock['benchmark_scripts']
    if any(scripts.get(name) != value for name, value in expected.items()):
        raise ValueError('Composed benchmark scripts differ from runtime lock')
    return dest


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


def start_server(config):
    # The pinned framework derives an auxiliary port as HTTP port + 10000.
    # Keep that derived port in range even on hosts with high ephemeral ports.
    for _ in range(100):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        if port <= 55000:
            break
    else:
        raise RuntimeError('Could not allocate a valid serving port')
    env = dict(os.environ)
    env.update({key: str(value) for key, value in config['envs'].items()})
    env.update(AKA_INSTALL_KERNEL='1', PYTHONPATH=str(ROOT / 'scripts'), NO_PROXY='127.0.0.1,localhost,0.0.0.0')
    command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(ROOT / 'model'),
               '--host', '127.0.0.1', '--port', str(port), '--tensor-parallel-size', str(config['envs']['TP']),
               '--context-length', str(config['envs']['MAX_MODEL_LEN']),
               *shlex.split(config['envs']['EXTRA_SGLANG_ARGS'])]
    (ROOT / 'server_command.json').write_text(json.dumps(command))
    log = (ROOT / 'server.log').open('w')
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Serving process exited: ' + (ROOT / 'server.log').read_text()[-12000:])
            try:
                request(port, 'get_model_info')
                return process, port, log
            except (OSError, ValueError):
                time.sleep(1)
        raise TimeoutError('Model startup exceeded 600 seconds')
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
    calls = [json.loads(line) for line in (ROOT / 'rank_calls.jsonl').read_text().splitlines()]
    expected = digest(ROOT / lock['candidate_module'])
    if any(row['sha256'] != expected or row['source'] != str(ROOT / lock['candidate_module']) for row in calls):
        raise ValueError('A server rank loaded a different candidate')
    for rank in range(lock['gpu_count']):
        if {r['function'] for r in calls if r['rank'] == rank} != set(lock['target_functions']):
            raise ValueError(f'Rank {rank} did not invoke both candidate operators')
    return calls


def runtime_identity(lock_digest, calls):
    settings, environment = [], []
    for call in calls:
        actual = call.get('server_settings')
        if not isinstance(actual, dict) or not actual:
            raise ValueError('A server rank did not expose effective launch settings')
        settings.append({k:v for k,v in actual.items() if k not in ('port', 'nccl_port', 'grpc_port')})
        # Keep the raw run identifier in the audit log, but not in configuration
        # equality: the framework generates a new diagnostic ID on every start.
        environment.append({k:v for k,v in call['runtime_settings'].items() if k != 'SGLANG_RUN_ID'})
    fingerprint = hashlib.sha256(json.dumps(dict(lock=lock_digest, settings=settings,
        rank_settings=environment), sort_keys=True).encode()).hexdigest()
    return fingerprint, settings


def evaluate(args, config, lock):
    from checks import source_policy
    final_candidate = args.role == 'candidate' and os.environ.get('ARENA_EVAL_PHASE') == 'candidate_evaluation'
    source_policy(target_only=final_candidate)
    if final_candidate:
        from checks import target_kernels
        target_kernels()
    row = identity(config)
    if args.action == 'validate-task':
        if digest(ROOT / 'benchmark.yaml') != lock['workload_sha256']:
            raise ValueError('Workload configuration differs from lock')
        initial = digest(ROOT / lock['candidate_module'])
        if initial != lock['initial_candidate_sha256']:
            raise ValueError('Initial implementation differs from the locked production source')
        row['checks'] = ['correctness', 'performance']
        return [row], dict(candidate_state='implemented', initial_source_sha256=initial)
    if args.action == 'compile':
        if final_candidate:
            from checks import compile_candidate
            return [], compile_candidate(lock)
        return [], dict(operator_cases=operator_checks(lock))
    metadata = {}
    diagnostics = []
    if args.action == 'correctness':
        metadata['operator_cases'] = operator_checks(lock)
        prompts = model_inputs(lock)
    client = prepare_client(lock) if args.action == 'performance' else None
    process, port, log = start_server(config)
    try:
        if args.action == 'correctness':
            output_length = lock['correctness']['output_tokens']
            for tokens in prompts:
                result = request(port, 'generate', dict(input_ids=tokens, return_logprob=True, logprob_start_len=0,
                    sampling_params=dict(temperature=0, max_new_tokens=output_length, ignore_eos=True)))
                generated = result['output_ids']
                if len(generated) != output_length:
                    raise ValueError('Model correctness decode length changed')
                rows = result['meta_info']['input_token_logprobs'] + result['meta_info']['output_token_logprobs']
                diagnostics.append(dict(tokens=tokens + generated, actual=rows))
        else:
            env = dict(os.environ)
            env.update({key: str(value) for key,value in config['envs'].items()})
            env.update(MODEL=str(ROOT / 'model'), PORT=str(port), RESULT_DIR=str(ROOT), RESULT_FILENAME='inferencex_result',
                       MAGPIE_RUN_PHASE='client', PROFILE='0', RUN_EVAL='false',
                       NO_PROXY='127.0.0.1,localhost,0.0.0.0', no_proxy='127.0.0.1,localhost,0.0.0.0')
            completed = subprocess.run(['bash', 'benchmarks/' + config['benchmark_script']], cwd=client,
                                       env=env, text=True, capture_output=True, timeout=500)
            (ROOT / 'client.log').write_text(completed.stdout + '\n' + completed.stderr)
            if completed.returncode:
                raise RuntimeError('Benchmark client failed: ' + (ROOT / 'client.log').read_text()[-10000:])
            if os.environ.get('ARENA_EVAL_PHASE') == 'task_validation':
                metadata['benchmark_sources'] = benchmark_sources(client, config)
                metadata['client_log'] = (ROOT / 'client.log').read_text()[-24000:]
            measured = json.loads((ROOT / 'inferencex_result.json').read_text())
            for name, count in (('input_lens', int(config['envs']['ISL'])),
                                ('output_lens', int(config['envs']['OSL']))):
                if measured.get(name) != [count] * int(config['envs']['NUM_PROMPTS']):
                    raise ValueError(f'Benchmark changed per-request {name}')
            row.update(execution_time_ms=1000*measured['duration'], benchmark_method='serving_wall_clock',
                       metrics=dict(duration_s=measured['duration'], completed_requests=measured['completed'],
                                    input_tokens=measured['total_input_tokens'],
                                    output_tokens=measured['total_output_tokens'], output_tokens_per_s=measured['output_throughput'],
                                    p99_tpot_ms=measured['p99_tpot_ms']))
        if args.action == 'performance':
            from _aka_measurement import validate_serving_case
            validate_serving_case(row)
            measurement = yaml.safe_load((ROOT/'config.yaml').read_text())['evaluation']['measurement']
            if row['metrics']['p99_tpot_ms'] > measurement.get('max_p99_tpot_ms', float('inf')):
                raise ValueError('Serving tail latency exceeded the task limit')
        metadata['rank_calls'] = sorted(audit(lock), key=lambda row: (row['rank'], row['function']))
        # Read the actual ServerArgs in every worker, without mixing the HTTP
        # server's volatile scheduler counters into the workload identity.
        fingerprint, stable_settings = runtime_identity(digest(ROOT / 'runtime.lock.json'), metadata['rank_calls'])
        metadata['server_settings'] = stable_settings
        row['metadata'] = dict(runtime_fingerprint=fingerprint,
                               kernel_sha256=digest(ROOT / lock['candidate_module']))
    finally:
        stop_server(process, log)
    if args.action == 'correctness':
        # Score the actual generated sequence with a separate implementation.
        # This checks prefill and decode without requiring deterministic tokens.
        reference = model_reference([item['tokens'] for item in diagnostics])
        for item, expected in zip(diagnostics, reference):
            item['expected'] = expected
        (ROOT / 'model_correctness.json').write_text(json.dumps(diagnostics, indent=2))
        errors = [logprob_error(item['tokens'], item['expected'], item['actual']) for item in diagnostics]
        if not errors or max(errors) > lock['correctness']['model_logprob_atol']:
            raise ValueError(f'Model logprob error exceeds threshold: {max(errors, default=-1)}')
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
    report = dict(protocol='arena-eval-v1', role=args.role, action=args.action, status='PASS', cases=[])
    try:
        config = yaml.safe_load((ROOT / 'benchmark.yaml').read_text())['benchmark']
        lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        if digest(ROOT / 'benchmark.yaml') != lock['workload_sha256']:
            raise ValueError('Workload differs from runtime lock')
        cases, metadata = evaluate(args, config, lock)
        report.update(cases=cases, metadata=metadata)
    except Exception as exc:
        report.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
    print('ARENA_EVAL_RESULT=' + json.dumps(report, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1
