"""Pinned engine launch and numerical protocol differences, shared by all models."""
from __future__ import annotations

import dataclasses
from array import array
from enum import Enum
import math
from pathlib import Path
import shlex
import sys


def install_protocol(lock):
    """Expose existing native token IDs, without changing engine computation.

    ATOM's response builder drops generated IDs even when return_token_ids is
    requested. This reporting-only shim applies identically to original and
    candidate servers. Ordinary timed/streaming requests are untouched.
    """
    if lock.get('framework') != 'atom' or lock.get('correctness', {}).get('model_reference') not in {'production_tokens', 'transformers_tokens'}:
        return
    from atom.entrypoints.openai import serving_completion
    previous = serving_completion.build_completion_response
    if getattr(previous, '_aka_native_token_reporting', False):
        return
    def report(request_id, model, final_output):
        response = previous(request_id, model, final_output)
        if 'prompt_token_ids' in final_output:
            ids = final_output.get('token_ids')
            if (not isinstance(ids, (list, array)) or any(type(i) is not int or i < 0 for i in ids)):
                raise ValueError('Native engine omitted actual generated token IDs')
            if len(response.choices) != 1 or len(ids) != response.usage['completion_tokens']:
                raise ValueError('Native generated token IDs contradict response work')
            response.choices[0]['token_ids'] = list(ids)
        return response
    report._aka_native_token_reporting = True
    serving_completion.build_completion_response = report


def launch_command(root, config, port, lock):
    """Use the upstream server phase, including its architecture-specific defaults.

    AgentX has no generic shell recipe: its launch argv is resolved from the
    pinned single-node recipe before materialization, never by the optimizer.
    """
    root = Path(root)
    spec = lock.get('server_launch') or lock.get('agentx', {}).get('launch_spec')
    if spec:
        command = [arg.replace('{model}', str(root/'model')).replace('{port}', str(port))
                   for arg in spec['argv']]
        if config['framework'] == 'sglang':
            command = seeded_arguments(command, lock['server_seed'])
        return command, spec['health_path'].lstrip('/')
    script = root / 'client/benchmarks' / config['benchmark_script']
    return ['bash', str(script)], 'get_model_info' if config['framework'] == 'sglang' else 'v1/models'


def server_environment(root, config, port, install_kernel, lock):
    root = Path(root)
    env = dict(MODEL=str(root/'model'), PORT=str(port), RESULT_DIR=str(root),
        SERVER_LOG=str(root/('engine.log' if install_kernel else 'reference_engine.log')),
        MAGPIE_SERVER_PID_FILE=str(root/'server.pid'), MAGPIE_RUN_PHASE='server',
        MAGPIE_KEEP_CONTAINER_ALIVE='1', PROFILE='0', RUN_EVAL='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    spec = lock.get('server_launch') or lock.get('agentx', {}).get('launch_spec')
    if spec:
        env.update({k: v.replace('{model}', str(root/'model')).replace('{port}', str(port))
                    for k, v in spec['env'].items()})
    elif config['framework'] == 'sglang':
        extra = shlex.split(str(config['envs'].get('EXTRA_SGLANG_ARGS', '')))
        env['EXTRA_SGLANG_ARGS'] = shlex.join(seeded_arguments(extra, lock['server_seed']))
    return env


def seeded_arguments(argv, seed):
    argv = list(argv)
    values = [arg.split('=', 1)[1] for arg in argv if arg.startswith('--random-seed=')]
    for index, arg in enumerate(argv):
        if arg == '--random-seed':
            if index+1 >= len(argv):
                raise ValueError('Missing frozen server random seed')
            values.append(argv[index+1])
    if values and (len(values) != 1 or values[0] != str(seed)):
        raise ValueError('Server arguments conflict with the locked random seed')
    return argv if values else [*argv, '--random-seed', str(seed)]


def _plain(value):
    """Serialize effective configuration, including derived precision values.

    Unknown configuration objects fail closed instead of silently disappearing.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    kind = type(value)
    if (type(kind).__module__ == 'pybind11_builtins'
            and type(kind).__name__ == 'pybind11_type'
            and isinstance(getattr(kind, '__members__', None), dict)):
        # Native enum bindings are not Python Enum subclasses. Keep both the
        # symbolic choice and its numeric value; ordinary native objects do not
        # get a permissive string/attribute fallback.
        if not isinstance(value.name, str):
            raise ValueError('Native configuration enum has no symbolic name')
        return dict(enum_type=kind.__module__ + '.' + kind.__qualname__,
                    name=value.name, value=int(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
                if not f.name.startswith('_') and (f.init or f.name not in {
                    'static_forward_context', 'local_cache_dir'})}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_plain(v) for v in value]
        return sorted(values, key=str) if isinstance(value, (set, frozenset)) else values
    if hasattr(value, 'model_dump'):
        return _plain(value.model_dump(mode='json'))
    if hasattr(value, 'to_dict'):
        return _plain(value.to_dict())
    if type(value).__module__ == 'torch' and type(value).__name__ in {'dtype', 'device'}:
        return str(value)
    if isinstance(value, type):
        return value.__module__ + '.' + value.__qualname__
    if hasattr(type(value), '__struct_fields__'):
        return {name: _plain(getattr(value, name)) for name in type(value).__struct_fields__
                if not name.startswith('_')}
    for module_name in ('vllm.model_executor.layers.quantization.base_config', 'atom.config'):
        base = getattr(sys.modules.get(module_name), 'QuantizationConfig', None)
        if isinstance(base, type) and isinstance(value, base):
            # Native quantization configs are ordinary typed objects, not
            # dataclasses. Preserve their type and every field, including private
            # configuration state; plugin subclasses use this same protocol.
            # Unknown nested values still fail closed. The config's defining
            # engine already loaded its base; do not import unrelated engines.
            return dict(configuration_type=type(value).__module__ + '.' + type(value).__qualname__,
                        fields={name: _plain(item) for name, item in vars(value).items()})
    raise ValueError('Unsupported effective configuration value: '
                     + type(value).__module__ + '.' + type(value).__qualname__)


def _vllm_configuration(model):
    from vllm.config import get_current_vllm_config
    configurations = {}
    if model is not None:
        owners = model.modules() if callable(getattr(model, 'modules', None)) else [model]
        for owner in owners:
            value = vars(owner).get('vllm_config')
            if value is not None:
                configurations[id(value)] = value
    if len(configurations) > 1:
        raise ValueError('Model exposed multiple effective serving configurations')
    return next(iter(configurations.values())) if configurations else get_current_vllm_config()


def runtime_inputs(lock, model=None):
    """Retain configured cache storage before compilation resolves a code key."""
    if lock['framework'] not in {'vllm', 'atom'}:
        return {}
    if lock['framework'] == 'vllm':
        config = _vllm_configuration(model)
    else:
        from atom.config import get_current_atom_config
        config = get_current_atom_config()
    compiled = config.get('compilation_config') if isinstance(config, dict) else config.compilation_config
    value = compiled.get('cache_dir') if isinstance(compiled, dict) else compiled.cache_dir
    if not isinstance(value, str):
        raise ValueError('Compiler cache input must be an explicit directory or the native empty default')
    return dict(compilation_cache_dir=value)


def runtime_settings(lock, model=None):
    framework = lock['framework']
    if framework == 'sglang':
        try:
            from sglang.srt.runtime_context import get_context
        except ImportError:
            get_context = None
        context = get_context() if get_context else None
        resolved = getattr(context, 'resolved_server_args_dict', None)
        if callable(resolved):
            # Includes resolved namespace values and process-local overrides;
            # the newer raw get_server_args() is not an effective-settings API.
            settings = _plain(resolved())
        else:
            from sglang.srt.server_args import get_global_server_args
            settings = _plain(get_global_server_args())
        model_path = settings.get('model_path')
    elif framework == 'vllm':
        # The loading/compilation context is often gone at forward time.
        # Compiled model modules retain the actual shared configuration object;
        # serialize its current values, not a constructor-time snapshot.
        settings = _plain(_vllm_configuration(model))
        if 'instance_id' in settings:
            # vLLM assigns this diagnostic incarnation ID at every startup.
            settings['_runtime_instance_id'] = settings.pop('instance_id')
        model_path = settings.get('model_config', {}).get('model')
    elif framework == 'atom':
        from atom.config import get_current_atom_config
        settings = _plain(get_current_atom_config())
        model_path = settings.get('model') or settings.get('model_path')
    else:
        raise ValueError('No effective-settings observer for backend: ' + str(framework))
    if model_path != '/task/model':
        raise ValueError('Server did not load the pinned read-only model mount')
    return settings


def _token(value):
    if not isinstance(value, str) or not value.startswith('token_id:'):
        raise ValueError('Backend must expose exact token IDs for numerical checking')
    result = int(value.removeprefix('token_id:'))
    if result < 0:
        raise ValueError('Invalid token ID')
    return result


def completion_request(send, root, port, path, value=None, *, timeout=120, require_prompt_ids=False):
    """Translate token-ID numerical requests; retain the real no-logprob path."""
    if path != 'generate':
        return send(port, path, value, timeout=timeout)
    params = value['sampling_params']
    diagnostic = value.get('return_logprob', False)
    body = dict(model=str(Path(root)/'model'), prompt=value['input_ids'],
        max_tokens=params['max_new_tokens'], temperature=params.get('temperature', 0),
        ignore_eos=params.get('ignore_eos', True), logprobs=1 if diagnostic else None,
        echo=diagnostic, return_tokens_as_token_ids=True, return_token_ids=True)
    result = send(port, 'v1/completions', body, timeout=timeout)
    if require_prompt_ids and result.get('prompt_token_ids') != value['input_ids']:
        raise ValueError('Backend changed or omitted actual prompt token IDs')
    if len(result.get('choices', [])) != 1:
        raise ValueError('Numerical check requires one complete completion')
    choice = result['choices'][0]
    if not diagnostic:
        ids = choice.get('token_ids')
        if not isinstance(ids, list) or len(ids) != params['max_new_tokens'] or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('Backend omitted exact token IDs on the no-logprob generation path')
        return dict(output_ids=ids, meta_info={})
    logprobs = choice.get('logprobs') or {}
    tokens, probabilities, tops = (logprobs.get(k) for k in ('tokens', 'token_logprobs', 'top_logprobs'))
    if not isinstance(tokens, list) or not isinstance(probabilities, list) or len(tokens) != len(probabilities):
        raise ValueError('Backend omitted numerical token evidence')
    ids = [_token(token) for token in tokens]
    prompt = value['input_ids']
    if ids[:len(prompt)] != prompt or len(ids) != len(prompt) + params['max_new_tokens']:
        raise ValueError('Backend changed prompt IDs or decode work')
    rows = [[p, token] for p, token in zip(probabilities, ids)]
    rows[0][0] = None
    meta = dict(input_token_logprobs=rows[:len(prompt)], output_token_logprobs=rows[len(prompt):])
    if value.get('top_logprobs_num'):
        if not isinstance(tops, list) or len(tops) != len(ids):
            raise ValueError('Backend omitted teacher-forcing top probabilities')
        converted = [None]
        for top in tops[1:len(prompt)]:
            if not isinstance(top, dict) or not top:
                raise ValueError('Backend omitted a teacher-forcing position')
            token, prob = max(top.items(), key=lambda item: item[1])
            converted.append([[prob, _token(token)]])
        meta['input_top_logprobs'] = converted
    return dict(output_ids=ids[len(prompt):], meta_info=meta)
