"""Compile pinned Magpie recipes into self-contained, locked Arena serving tasks.

Catalog entries are reusable workload definitions. Materialization binds the
runtime before optimization; neither the agent nor a scoring action resolves a
moving image tag, model revision, dataset, or upstream checkout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import shlex
import subprocess
import tempfile

import yaml

from .measurement import lock_limits, trace_input_token_policy

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'tasks/e2e/magpie/catalog.json'
FRAMEWORKS = {'sglang', 'vllm', 'atom'}
PRECISIONS = {'fp16', 'bf16', 'fp8', 'fp4', 'mxfp4', 'mxfp8', 'w4a16'}
_REVISION = re.compile(r'[0-9a-f]{40}\Z')
_IMAGE = re.compile(r'[^\s]+@sha256:[0-9a-f]{64}\Z')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def _recipe_id(path):
    relative = PurePosixPath(path).relative_to('examples/benchmarks')
    return '/'.join(relative.with_suffix('').parts)


def refresh(checkout: Path, output: Path = CATALOG):
    """Import every tracked benchmark recipe from a clean, immutable checkout."""
    checkout = Path(checkout).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
    revision = git('rev-parse', 'HEAD')
    if not _REVISION.fullmatch(revision) or git('status', '--porcelain', '--', 'examples/benchmarks'):
        raise ValueError('Catalog refresh requires clean, committed upstream recipes')
    paths = git('ls-tree', '-r', '--name-only', revision, 'examples/benchmarks').splitlines()
    entries = []
    for path in sorted(paths):
        if PurePosixPath(path).suffix not in {'.yaml', '.yml'}:
            continue
        source = subprocess.check_output(['git', '-C', str(checkout), 'show', revision + ':' + path])
        value = yaml.safe_load(source)
        if not isinstance(value, dict) or not isinstance(value.get('benchmark'), dict):
            raise ValueError('Benchmark recipe must contain a benchmark mapping: ' + path)
        entry = dict(id=_recipe_id(path), upstream_path=path, sha256=sha256(source),
                     source=source.decode('utf-8'))
        describe(entry)  # Fail visibly when upstream introduces a new contract.
        entries.append(entry)
    if not entries:
        raise ValueError('No upstream benchmark recipes found')
    catalog = dict(version=1, repository='https://github.com/AMD-AGI/Magpie',
                   revision=revision, entries=entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, catalog)
    return catalog


def load_catalog(path: Path = CATALOG):
    catalog = json.loads(Path(path).read_text())
    if catalog.get('version') != 1 or not _REVISION.fullmatch(catalog.get('revision', '')):
        raise ValueError('Catalog must pin an upstream commit')
    ids = set()
    for entry in catalog['entries']:
        if entry['id'] in ids or _recipe_id(entry['upstream_path']) != entry['id']:
            raise ValueError('Duplicate or inconsistent catalog identity')
        ids.add(entry['id'])
        if sha256(entry['source'].encode()) != entry['sha256']:
            raise ValueError('Catalog recipe bytes differ from their digest')
        describe(entry)
    return catalog


def describe(entry):
    recipe = yaml.safe_load(entry['source'])['benchmark']
    framework = recipe.get('framework')
    if framework not in FRAMEWORKS:
        raise ValueError('No serving backend for upstream framework: ' + str(framework))
    precision = recipe.get('precision', 'fp8')
    if precision not in PRECISIONS:
        raise ValueError('Unknown precision; add an explicit representation: ' + str(precision))
    env = recipe.get('envs', {})
    if not isinstance(env, dict) or any(not isinstance(k, str) for k in env):
        raise ValueError('Recipe environment must be a mapping')
    agentx = recipe.get('agentx')
    enabled = (agentx.get('enabled', True) if isinstance(agentx, dict) else
               str(agentx).lower() in {'true', 'enable', 'enabled', 'yes', '1'})
    script = recipe.get('benchmark_script', '')
    scenario = 'agentx' if enabled else ('multimodal' if '_mm' in script or any(k.startswith('MM_') for k in env) else 'text')
    return dict(id=entry['id'], framework=framework, model=recipe.get('model'),
                precision=precision, scenario=scenario, run_mode=recipe.get('run_mode', 'docker'),
                profiling=bool(recipe.get('profiler', {}).get('torch_profiler', {}).get('enabled', False)),
                qualification='not_claimed')


def normalized_recipe(entry, binding):
    """Freeze computational settings; adapt only ownership, paths and transport."""
    recipe = copy.deepcopy(yaml.safe_load(entry['source'])['benchmark'])
    info = describe(entry)
    original_model = recipe['model']
    model = binding['model']['id']
    if PurePosixPath(original_model).is_absolute():
        if binding.get('source_model_alias') != original_model:
            raise ValueError('A host-local upstream model requires an explicit source_model_alias binding')
    elif model != original_model:
        raise ValueError('Runtime model differs from the selected workload')
    recipe['model'] = model
    recipe['run_mode'] = 'local'  # Inside the host-owned, fresh evaluation container.
    recipe['docker_image'] = binding['image']
    recipe.pop('ray_config', None)
    recipe.pop('server_lifecycle', None)
    recipe['gpu_selection'] = {'enabled': False}
    recipe['profiler'] = {'torch_profiler': {'enabled': False},
                          'system_profiler': {'enabled': False}, 'tracelens': {'enabled': False}}
    recipe['gap_analysis'] = {'enabled': False}
    recipe.pop('inferencex_path', None)
    recipe.pop('hf_cache_path', None)
    env = recipe.setdefault('envs', {})
    # External endpoints cannot supply candidate evidence. The same task-owned
    # server is used for original client-only and persisted-server examples.
    env.pop('BENCHMARK_BASE_URL', None)
    for key in ('PROFILE', 'ENABLE_PROFILE', 'ENABLE_PROFILIER'):
        env.pop(key, None)
    for name in ('RESULT_DIR', 'SERVER_LOG', 'MAGPIE_SERVER_PID_FILE', 'MODEL_PATH',
                 'HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        env.pop(name, None)
    if info['scenario'] != 'agentx':
        if 'TP' not in env:
            if not binding.get('server_topology'):
                raise ValueError('Client-only workload requires a frozen server_topology binding')
            env.update(binding['server_topology'])
        for key, default in [('CONC', 32), ('ISL', 1024), ('OSL', 1024),
                             ('RANDOM_RANGE_RATIO', 1)]:
            env.setdefault(key, default)
        if info['framework'] != 'sglang':
            env.setdefault('MAX_MODEL_LEN', 4096)
        count = binding.get('measurement_num_prompts')
        if count is not None:
            original_count = int(env.get('NUM_PROMPTS', int(env['CONC'])*10))
            if type(count) is not int or count < original_count:
                raise ValueError('Prepared measurement samples may increase, never decrease')
            env['NUM_PROMPTS'] = count
        runner = recipe.get('runner_type') or binding.get('runner_type', 'mi355x')
        script = recipe.get('benchmark_script')
        # Older native scripts own a complete deployment and cannot be safely
        # split by the generic client. Preserve their launch as an explicit
        # prepared argv and use the shared framework client for measurement.
        if script and not script.startswith(('sglang_', 'vllm_', 'atom_')):
            if not binding.get('server_launch'):
                raise ValueError('Legacy native recipe requires a frozen server_launch binding')
            script = None
        recipe['benchmark_script'] = script or f"{info['framework']}_{runner}.sh"
    else:
        if 'measurement_num_prompts' in binding:
            raise ValueError('Canonical trace replay uses its fixed duration, not a prompt-count override')
        agentx = recipe['agentx'] if isinstance(recipe['agentx'], dict) else {}
        if agentx.get('mode', 'canonical') != 'canonical':
            raise ValueError('Fast AgentX replay is diagnostic and cannot define a scoring task')
        # This resolved spec is prepared from the pinned upstream recipe before
        # search. Optimization can never supply launch_overrides.
        if not isinstance(binding.get('agentx'), dict):
            raise ValueError('AgentX requires a resolved, pinned launch and replay binding')
        recipe['agentx'] = {**agentx, 'enabled': True, 'mode': 'canonical'}
    # Preserve TP/EP/PP, quantization, speculative decoding and every engine
    # option verbatim; runtime fingerprinting also checks effective settings.
    return recipe


def validate_binding(binding, info):
    if not _IMAGE.fullmatch(binding.get('image', '')):
        raise ValueError('Materialization requires an immutable image digest')
    if not _REVISION.fullmatch(binding.get('model', {}).get('revision', '')):
        raise ValueError('Materialization requires an immutable model revision')
    if type(binding.get('gpu_count')) is not int or binding['gpu_count'] <= 0:
        raise ValueError('Materialization requires a GPU count')
    if not binding.get('gpu_arch') or not isinstance(binding.get('model_architectures'), list) or not binding['model_architectures']:
        raise ValueError('Declare GPU architecture and actual model implementation class names')
    if not isinstance(binding.get('correctness'), dict):
        raise ValueError('A protected, calibrated correctness policy is required')
    reference = binding['correctness'].get('model_reference')
    allowed = {'production_serving'} if info['scenario'] == 'multimodal' else {
        'transformers_fp32', 'production_quality', 'production_numerical'}
    if info['framework'] == 'atom' and info['scenario'] == 'text':
        allowed.update({'production_tokens', 'transformers_tokens'})
    if reference not in allowed:
        raise ValueError('Catalog tasks require an independent model numerical reference')
    inputs = binding.get('inputs', {})
    prompts = inputs.get('correctness.json')
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError('Declare protected model correctness prompts')
    if len(prompts) != binding['correctness'].get('model_samples'):
        raise ValueError('Model correctness sample count differs from protected inputs')
    if reference in {'production_quality', 'production_numerical'}:
        from .tools.perf.serving import production_logprob_tolerances, production_tolerances
        production_tolerances(binding)
        production_logprob_tolerances(binding)
    elif reference == 'production_tokens':
        if binding['correctness'].get('token_match') != 'exact':
            raise ValueError('Native token reference requires exact equality, with no fallback')
    else:
        tolerance = binding['correctness'].get('model_logprob_atol')
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Declare a positive finite numerical tolerance')
    for filename, expected in (
        ('quality.json', binding['correctness'].get('quality', {}).get('dataset_sha256')),
        ('media_checks.json', binding['correctness'].get('media_sha256'))):
        if filename in inputs and sha256(json_bytes(inputs[filename])) != expected:
            raise ValueError('Protected JSON bytes differ from their binding: ' + filename)
    limits = lock_limits(binding)
    if 'server_seed' in binding and (type(binding['server_seed']) is not int or not 0 <= binding['server_seed'] < 2**31):
        raise ValueError('Server seed must be a nonnegative 31-bit integer')
    scripts = binding.get('benchmark_scripts')
    if not isinstance(scripts, dict) or not scripts or any(not re.fullmatch(r'[0-9a-f]{64}', value) for value in scripts.values()):
        raise ValueError('Bind the composed benchmark script digests before materialization')
    for name, dep in binding.get('dependencies', {}).items():
        if not _REVISION.fullmatch(dep.get('revision', '')):
            raise ValueError('Dependency requires immutable commit: ' + name)
    if not {'magpie', 'inferencex'} <= binding.get('dependencies', {}).keys():
        raise ValueError('Declare pinned Magpie and InferenceX sources')
    from .prepare_serving_runtime import validate_assets
    validate_assets(binding.get('assets', {}))
    client = binding.get('client_runtime')
    if info['scenario'] == 'text' or client:
        if not isinstance(client, dict):
            raise ValueError('Text workloads require an immutable prepared client_runtime')
        from .tools.perf.serving_scenarios import validate_client_review_sources
        validate_client_review_sources(client)
        if client.get('kind', 'archive') == 'image':
            executable = PurePosixPath(client.get('executable', ''))
            if not executable.is_absolute() or len(executable.parts) < 2 or '..' in executable.parts or executable.parts[1] not in {'opt', 'usr'}:
                raise ValueError('Image client interpreter must use a fixed /opt or /usr path')
        elif client.get('kind', 'archive') != 'archive':
            raise ValueError('Unknown immutable client runtime kind')
        else:
            for name in ('asset', 'executable'):
                path = PurePosixPath(client.get(name, ''))
                if not client.get(name) or path.is_absolute() or '..' in path.parts:
                    raise ValueError('Client runtime paths must stay within their prepared roots')
            asset = binding.get('assets', {}).get(client['asset'], {})
            if not re.fullmatch(r'[0-9a-f]{64}', asset.get('sha256', '')):
                raise ValueError('Client runtime requires a digest-bound archive asset')
    if info['scenario'] in {'agentx', 'multimodal'}:
        manifest = binding.get('request_manifest')
        if not isinstance(manifest, dict) or not re.fullmatch(r'[0-9a-f]{64}', manifest.get('dataset_sha256', '')):
            raise ValueError('Trace/media tasks require a frozen request and dataset manifest')
        if info['scenario'] == 'multimodal':
            for name in ('input_lens', 'output_lens'):
                values = manifest.get(name)
                if not isinstance(values, list) or not values or any(type(v) is not int or v <= 0 for v in values):
                    raise ValueError('Media manifest requires complete per-request token counts')
            if len(manifest['input_lens']) != len(manifest['output_lens']):
                raise ValueError('Media input/output coverage differs')
            if 'media_checks.json' not in inputs:
                raise ValueError('Multimodal tasks require image-bearing correctness requests')
    if info['scenario'] == 'agentx':
        agentx = binding.get('agentx', {})
        if not re.fullmatch(r'[0-9a-f]{64}', agentx.get('recipe_fingerprint', '')):
            raise ValueError('AgentX requires its resolved canonical recipe fingerprint')
        _command(agentx.get('launch_spec'), 'AgentX server')
        _command(agentx.get('client'), 'AgentX client')
        if '--scenario' not in agentx['client']['argv'] or 'inferencex-agentx-mvp' not in agentx['client']['argv'] or '--benchmark-duration' not in agentx['client']['argv']:
            raise ValueError('AgentX must use the canonical native replay command')
        if 'aiperf' not in binding['dependencies'] or not binding.get('assets'):
            raise ValueError('AgentX requires pinned AIPerf sources and offline trace assets')
        duration = manifest.get('duration_s')
        if type(duration) is not int or duration <= 0:
            raise ValueError('AgentX requires a fixed positive measurement duration')
        if limits['client_timeout_s'] <= duration:
            raise ValueError('AgentX client timeout must exceed its fixed replay duration')
        argv = agentx['client']['argv']
        if trace_input_token_policy(manifest) == 'native_reconstruction' and (
                argv.count('--use-server-token-count') != 1 or '--no-use-server-token-count' in argv):
            raise ValueError('Native reconstruction requires --use-server-token-count')
        if str(duration) != argv[argv.index('--benchmark-duration')+1]:
            raise ValueError('AgentX replay command differs from its duration manifest')
        indices = binding['correctness'].get('trace_probe_indices')
        if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(manifest.get('requests', [])) for i in indices):
            raise ValueError('AgentX requires protected trace-shape correctness coverage')
        batch = binding['correctness'].get('trace_batch_indices')
        if not isinstance(batch, list) or not batch or any(type(i) is not int or not 0 <= i < len(manifest.get('requests', [])) for i in batch):
            raise ValueError('AgentX requires protected concurrent trace correctness coverage')
    if binding.get('server_launch'):
        _command(binding['server_launch'], 'Prepared server')


def _command(spec, label):
    if not isinstance(spec, dict) or not isinstance(spec.get('argv'), list) or not spec['argv'] or any(not isinstance(v, str) or '\0' in v for v in spec['argv']):
        raise ValueError(label + ' requires literal argv')
    if not isinstance(spec.get('env'), dict) or any(not isinstance(k, str) or not isinstance(v, str) or '\0' in k+v for k,v in spec['env'].items()):
        raise ValueError(label + ' requires literal environment values')


def materialize(catalog, recipe_id, binding, destination: Path, *, root=ROOT):
    """Generate one complete v2 task; never overwrite a task or experiment."""
    entries = {e['id']: e for e in catalog['entries']}
    if recipe_id not in entries:
        raise ValueError('Unknown catalog workload: ' + recipe_id)
    entry = entries[recipe_id]
    info = describe(entry)
    validate_binding(binding, info)
    if binding['dependencies']['magpie']['revision'] != catalog['revision']:
        raise ValueError('Runtime Magpie revision differs from catalog source')
    recipe = normalized_recipe(entry, binding)
    if info['scenario'] == 'multimodal':
        expected = int(recipe['envs'].get('NUM_PROMPTS', int(recipe['envs']['CONC'])*10))
        if len(binding['request_manifest']['input_lens']) != expected:
            raise ValueError('Media request manifest differs from the complete prepared sample count')
    if info['scenario'] == 'agentx':
        if len(binding['correctness']['trace_batch_indices']) != int(recipe['envs']['CONC']):
            raise ValueError('AgentX correctness requires one complete concurrent trace batch')
        client = binding['agentx']['client']
        argv = client['argv']
        for flag, expected in [('--concurrency', str(recipe['envs']['CONC'])),
                               ('--url', 'http://127.0.0.1:{port}')]:
            if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv) or argv[argv.index(flag)+1] != expected:
                raise ValueError('Canonical AgentX client must fix ' + flag)
        if client['env'].get('AIPERF_EXPERIMENTAL_FAST', '0') != '0' or '--unsafe-override' in argv:
            raise ValueError('Diagnostic/unsafe replay cannot define a scoring task')
    task_id = 'e2e/magpie/' + recipe_id
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing task: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.catalog-', dir=destination.parent) as tmp:
        task = Path(tmp) / 'task'
        (task / 'source').mkdir(parents=True)
        (task / 'scripts').mkdir()
        _write_json(task / 'source/patches.json', {'version': 1, 'patches': []})
        benchmark = yaml.safe_dump({'benchmark': recipe}, sort_keys=False).encode()
        (task / 'benchmark.yaml').write_bytes(benchmark)
        (task / 'upstream.yaml').write_text(entry['source'])
        provenance = dict(repository=catalog['repository'], revision=catalog['revision'],
                     path=entry['upstream_path'], sha256=entry['sha256'], adaptations=[
                         'fresh evaluator-owned server', 'offline immutable assets', 'profiling excluded from timing'])
        if 'measurement_num_prompts' in binding:
            upstream_env = yaml.safe_load(entry['source'])['benchmark'].get('envs', {})
            provenance['measurement_samples'] = dict(
                original=int(upstream_env.get('NUM_PROMPTS', int(upstream_env.get('CONC',32))*10)),
                prepared=binding['measurement_num_prompts'], fixed_before_optimization=True)
        _write_json(task / 'provenance.json', provenance)
        lock = copy.deepcopy(binding)
        lock.update(version=1, catalog_version=1, task_id=task_id, framework=info['framework'], scenario=info['scenario'],
                    benchmark_script=recipe.get('benchmark_script'), limits=lock_limits(binding),
                    workload_sha256=sha256(benchmark),
                    minimum_final_evaluation_s=binding.get('minimum_final_evaluation_s', 10800))
        if info['framework'] == 'sglang':
            from .tools.perf.serving_backends import seeded_arguments
            launch = binding.get('server_launch') or binding.get('agentx', {}).get('launch_spec')
            argv = launch['argv'] if launch else shlex.split(str(recipe['envs'].get('EXTRA_SGLANG_ARGS', '')))
            seeds = [arg.split('=', 1)[1] for arg in argv if arg.startswith('--random-seed=')]
            seeds += [argv[index+1] for index, arg in enumerate(argv[:-1]) if arg == '--random-seed']
            lock.setdefault('server_seed', int(seeds[0]) if seeds else 1729)
            seeded_arguments(argv, lock['server_seed'])
        # Package boundaries avoid a model-specific operator inventory. These
        # are compute submission locations, not permission to change config or
        # scheduling: source policy, model checks and semantic review still apply.
        common = ['aiter', 'sgl_kernel', 'flash_attn', 'fla', 'torch.nn', 'torch._inductor']
        prefixes = {name: [name, *common] for name in FRAMEWORKS}
        control = dict(sglang=['sglang.srt.server_args', 'sglang.srt.runtime_context',
                              'sglang.srt.arg_groups'],
                       vllm=['vllm.config', 'vllm.engine.arg_utils', 'vllm.envs'],
                       atom=['atom.config', 'atom.entrypoints.openai'])
        lock['candidate_bundle'] = dict(root='source', manifest='patches.json',
            module_prefixes=prefixes[info['framework']], allow_empty=True, scope='whole_model',
            protected_modules=control[info['framework']],
            observer={'kind': 'model_classes', 'classes': binding['model_architectures'],
                      'entrypoint': '__call__' if info['framework'] == 'vllm' else 'forward'})
        hashes = {'patches.json': sha256((task / 'source/patches.json').read_bytes())}
        lock['initial_bundle_sha256'] = sha256(json.dumps(hashes, sort_keys=True).encode())
        _write_json(task / 'runtime.lock.json', lock)
        config = dict(schema_version=2, description='Improve fixed-workload whole-model throughput using '
            'general GPU kernel implementations or fusion. Preserve all model semantics, precision, '
            'parallelism, serving options and request data.',
            candidate=dict(language='gpu', initial_state='implemented', editable=[dict(path='source', scope='tree')]),
            baseline=dict(kind='initial_candidate', correctness_policy='required'),
            platform_support=dict(required_arch=binding['gpu_arch']),
            evaluation=dict(runner=['python3', 'scripts/evaluate.py'], workloads='benchmark.yaml',
                timeout_s=binding.get('action_timeout_s', 14400),
                measurement=dict(kind='serving', runtime_lock='runtime.lock.json', pairs=3, minimum_speedup_ratio=1.05)),
            instructions=['README.md'])
        (task / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        required = ['correctness.json']
        if binding['correctness']['model_reference'] == 'production_quality':
            required.append('quality.json')
        if info['scenario'] == 'multimodal':
            required.append('media_checks.json')
        for name in required:
            if name not in binding.get('inputs', {}):
                raise ValueError('Missing protected correctness input: ' + name)
            _write_json(task / name, binding['inputs'][name])
        for source, target in [('catalog_serving.py', 'evaluate.py'),
                               ('serving_bundle.py', '_aka_serving_bundle.py'),
                               ('serving_backends.py', '_aka_serving_backends.py'),
                               ('serving_scenarios.py', '_aka_serving_scenarios.py'),
                               ('serving_request_plan.py', '_aka_serving_request_plan.py'),
                               ('serving.py', '_aka_serving.py')]:
            shutil.copy2(root / 'src/tools/perf' / source, task / 'scripts' / target)
        shutil.copy2(root / 'src/measurement.py', task / 'scripts/_aka_measurement.py')
        shutil.copy2(root / 'src/tools/perf/catalog_checks.py', task / 'scripts/checks.py')
        if binding['correctness']['model_reference'] == 'production_quality':
            shutil.copy2(root / 'tasks/e2e/qwen3_next_80b_sglang/scripts/quality.py', task / 'scripts/quality.py')
            shutil.copy2(root / 'tasks/e2e/qwen3_next_80b_sglang/QUALITY_LICENSE', task / 'QUALITY_LICENSE')
        shutil.copy2(root / 'tasks/e2e/MAGPIE_LICENSE', task / 'MAGPIE_LICENSE')
        (task / 'scripts/sitecustomize.py').write_text(
            "import os, json\nfrom pathlib import Path\n"
            "from _aka_serving_backends import install_protocol\n"
            "task = Path(__file__).resolve().parents[1]\n"
            "lock = json.loads((task/'runtime.lock.json').read_text())\n"
            "install_protocol(lock)\n"
            "if os.environ.get('AKA_INSTALL_KERNEL') == '1':\n"
            "    from _aka_serving_bundle import install\n"
            "    install(task, lock)\n")
        (task / 'README.md').write_text((root / 'src/tools/perf/catalog_instructions.md').read_text())
        task.rename(destination)
    return destination


def client_manifest(magpie: Path, inferencex: Path, *, scenario, script=None):
    """Compose exactly the runtime client tree, including report-only media data."""
    from .tools.perf.serving_scenarios import compose_client
    with tempfile.TemporaryDirectory(prefix='aka-client-') as temp:
        dest = Path(temp)/'client'
        shutil.copytree(inferencex, dest)
        for path in (magpie/'Magpie/scripts/benchmark').glob('*.sh'):
            shutil.copy2(path, dest/'benchmarks'/path.name)
        compose_client(dest, dict(scenario=scenario, benchmark_script=script),
                       Path(__file__).parent/'tools/perf/serving_request_plan.py')
        return {str(p.relative_to(dest)): sha256(p.read_bytes()) for p in sorted((dest/'benchmarks').glob('*.sh'))}


def bind(catalog, recipe_id, profile, magpie, inferencex):
    """Complete a maintainer-owned runtime profile against two pinned checkouts.

    Calibration/probe inputs are explicit. Do not invent numerical tolerances,
    resolve moving revisions during scoring or call a binding GPU-qualified.
    """
    entry = next(e for e in catalog['entries'] if e['id'] == recipe_id)
    binding = copy.deepcopy(profile)
    def revision(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if revision(magpie) != catalog['revision']:
        raise ValueError('Magpie checkout differs from the catalog pin')
    if subprocess.check_output(['git', '-C', str(magpie), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Runtime binding requires clean upstream sources')
    if subprocess.check_output(['git', '-C', str(inferencex), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Runtime binding requires clean InferenceX sources')
    binding['dependencies'] = {**binding.get('dependencies', {}),
        'magpie': dict(repository=catalog['repository'] + '.git', revision=catalog['revision']),
        'inferencex': dict(repository='https://github.com/SemiAnalysisAI/InferenceX.git',
                           revision=revision(inferencex), subdirectory='inferencex-e2e')}
    info = describe(entry)
    recipe = normalized_recipe(entry, binding)
    binding['benchmark_scripts'] = client_manifest(magpie, inferencex/'inferencex-e2e',
        scenario=info['scenario'], script=recipe.get('benchmark_script'))
    # The profile carries JSON values, while the generated package has canonical
    # bytes. Freeze their new representation here, before any optimization.
    for filename, field in [('quality.json', 'quality'), ('media_checks.json', 'media')]:
        if filename in binding.get('inputs', {}):
            content = json_bytes(binding['inputs'][filename])
            if field == 'quality':
                binding['correctness']['quality']['dataset_sha256'] = sha256(content)
            else:
                binding['correctness']['media_sha256'] = sha256(content)
    validate_binding(binding, info)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    commands = parser.add_subparsers(dest='command', required=True)
    update = commands.add_parser('refresh')
    update.add_argument('--checkout', type=Path, required=True)
    listing = commands.add_parser('list')
    listing.add_argument('--json', action='store_true')
    make = commands.add_parser('materialize')
    make.add_argument('--id', required=True)
    make.add_argument('--binding', type=Path, required=True)
    make.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('bind')
    prepare.add_argument('--id', required=True)
    prepare.add_argument('--profile', type=Path, required=True)
    prepare.add_argument('--magpie', type=Path, required=True)
    prepare.add_argument('--inferencex', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'refresh':
        catalog = refresh(args.checkout, args.catalog)
        print(f"Pinned {len(catalog['entries'])} recipes at {catalog['revision']}")
    elif args.command == 'list':
        if not args.json:
            print('WORKLOAD\tENGINE\tSCENARIO\tMODEL\tGPU QUALIFICATION')
        for entry in load_catalog(args.catalog)['entries']:
            info = describe(entry)
            print(json.dumps(info, ensure_ascii=False) if args.json else '\t'.join(str(info[k]) for k in
                ('id', 'framework', 'scenario', 'model', 'qualification')))
    elif args.command == 'materialize':
        path = materialize(load_catalog(args.catalog), args.id, json.loads(args.binding.read_text()), args.output)
        print(path)
    else:
        if args.output.exists():
            raise FileExistsError('Refusing to replace an existing runtime binding')
        binding = bind(load_catalog(args.catalog), args.id, json.loads(args.profile.read_text()), args.magpie, args.inferencex)
        _write_json(args.output, binding)
        print(args.output)


if __name__ == '__main__':
    main()
