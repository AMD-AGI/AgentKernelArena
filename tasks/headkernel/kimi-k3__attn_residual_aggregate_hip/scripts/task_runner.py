"""Protected Kimi entrypoint; the frozen fixture adapter supplies native bindings."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ut'))
from evaluation_contract import canonical, finalize_report, fingerprint, strict_json, validate_manifest
from fresh_runner import FreshCallbacks
from source_guard import validate_sources


def digest(path):
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Task inputs must be regular task-local files')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_for(phase, manifest, path):
    policy = strict_json((ROOT/'ut/source_guard_policy.json').read_text())
    sources = {name: digest(ROOT/name) for name in policy['sources']}
    if path is not None:
        request = strict_json(Path(path).read_text())
        if (request.get('phase') != phase or request.get('manifest_sha256') != fingerprint(manifest)
                or request.get('source_sha256') != sources):
            raise ValueError('Request does not bind the current phase, cases and candidate sources')
        return request
    package = {str(path.relative_to(ROOT)): digest(path) for path in sorted(ROOT.rglob('*'))
               if path.is_file() and 'build' not in path.relative_to(ROOT).parts
               and '__pycache__' not in path.parts}
    return {'schema_version': 1, 'phase': phase, 'request_id': secrets.token_hex(24),
            'manifest_sha256': fingerprint(manifest), 'source_sha256': sources,
            'package_sha256': fingerprint(package), 'challenge_seed': secrets.randbelow(2**30),
            'origin': 'protected-kimi-task-runner'}


def load_runtime(manifest, request):
    path = ROOT/'ut/runtime_adapter.py'
    if not path.is_file():
        raise RuntimeError('Current Kimi fixture/native runtime_adapter.py is not installed; task is not qualified')
    digest(path)
    spec = importlib.util.spec_from_file_location('_frozen_kimi_runtime', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.create_runtime(ROOT, manifest, request)


def capture(prepared, callbacks, seed, torch):
    # Eager specialization was already invoked and checked. Warm the capture
    # stream with identical preparation for every source leg, outside timing.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            callbacks.check_once(seed)
    torch.cuda.current_stream().wait_stream(stream)
    truth = callbacks.reset_inputs(seed)
    callbacks.initialize_outputs()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.invoke_candidate()
    callbacks.replay = graph.replay
    callbacks.replay()
    callbacks.verify(truth)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('compile', 'correctness', 'performance'))
    parser.add_argument('--request')
    args = parser.parse_args()
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    output = build/(args.phase+'_report.json')
    output.unlink(missing_ok=True)
    try:
        manifest = strict_json((ROOT/'cases.json').read_text())
        validate_manifest(manifest)
        if manifest.get('status') != 'FROZEN_CURRENT_CAPTURE':
            raise RuntimeError('Current observed cases and fixture coverage are not sealed')
        validate_sources(ROOT, ROOT)
        request = request_for(args.phase, manifest, args.request)
        runtime = load_runtime(manifest, request)
        import torch
        if not torch.version.hip or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:
            raise RuntimeError('The pinned ROCm gfx950 runtime is required')
        policy = manifest['measurement']
        report = {'schema_version': 1, 'status': 'ok', 'request': request, 'cases': []}
        compiled = []
        for case in manifest['cases']:
            progress={'phase':args.phase,'case_id':case['case_id'],'cases_completed':len(compiled),'case_count':len(manifest['cases']),
                      'state':'preparing_full_shape','observed_unix':time.time()}
            (build/(args.phase+'_progress.json')).write_text(canonical(progress)+'\n')
            print(json.dumps(progress),flush=True)
            if case['calls_per_sample'] != 1:
                raise ValueError('Each scored Kimi graph replay must represent one native call')
            prepared = runtime.prepare_case(case)
            callbacks = prepared.callbacks
            if not isinstance(callbacks, FreshCallbacks):
                raise TypeError('Native fixture adapter must use the protected FreshCallbacks')
            callbacks.replay = prepared.invoke_candidate
            callbacks.check_once(request['challenge_seed'])
            if canonical(prepared.observe()) != canonical(case):
                raise ValueError('Compiled native case does not match the frozen ABI')
            engagement = prepared.native_engagement()
            if (engagement.get('source_sha256') != request['source_sha256']
                    or engagement.get('candidate_invoked') is not True
                    or engagement.get('independent_reference_invoked') is not True):
                raise ValueError('Current candidate and independent native reference were not engaged')
            compiled.append({'case_id': case['case_id'], **engagement})
            if args.phase == 'compile':
                progress.update(state='case_complete',cases_completed=len(compiled),observed_unix=time.time())
                (build/(args.phase+'_progress.json')).write_text(canonical(progress)+'\n')
                continue
            graph = capture(prepared, callbacks, request['challenge_seed'], torch)
            if args.phase == 'correctness':
                row = callbacks.correctness_row(case, policy, observe=prepared.observe,
                    corrupt_outputs=prepared.corrupt_outputs, challenge_seed=request['challenge_seed'])
            else:
                row = callbacks.performance_row(case, policy, observe=prepared.observe,
                    challenge_seed=request['challenge_seed'])
            report['cases'].append(row)
            progress.update(state='case_complete',cases_completed=len(report['cases']),observed_unix=time.time())
            (build/(args.phase+'_progress.json')).write_text(canonical(progress)+'\n')
            print(json.dumps(progress),flush=True)
            del graph, prepared, callbacks
        report.update(compiled=True, compiled_specializations=compiled,
                      oracle_order='CPU input truth before candidate; CPU output and post-input snapshots before reference')
        report = finalize_report(report, manifest, request)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(canonical(report)+'\n')
        temporary.replace(output)
        print(json.dumps({'status': 'ok', 'phase': args.phase, 'case_count': len(manifest['cases'])}))
    except BaseException:
        output.unlink(missing_ok=True)
        output.with_suffix('.tmp').unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    main()
