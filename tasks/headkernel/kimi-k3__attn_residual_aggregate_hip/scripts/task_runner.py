"""Protected Kimi entrypoint; the frozen fixture adapter supplies native bindings."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
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


def load_runtime(manifest, request, *, build=None):
    path = ROOT/'ut/runtime_adapter.py'
    if not path.is_file():
        raise RuntimeError('Current Kimi fixture/native runtime_adapter.py is not installed; task is not qualified')
    digest(path)
    spec = importlib.util.spec_from_file_location('_frozen_kimi_runtime', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if build is None:
        return module.create_runtime(ROOT, manifest, request)
    return module.create_runtime(ROOT, manifest, request, build=build)


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


def run_case(runtime, case, phase, request, policy, torch):
    """Run one complete case using the original protected callbacks and cleanup."""
    prepared = callbacks = graph = None
    try:
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
        compiled = {'case_id': case['case_id'], **engagement}
        if phase == 'compile':
            return compiled, None
        graph = capture(prepared, callbacks, request['challenge_seed'], torch)
        if phase == 'correctness':
            row = callbacks.correctness_row(case, policy, observe=prepared.observe,
                corrupt_outputs=prepared.corrupt_outputs, challenge_seed=request['challenge_seed'])
        else:
            row = callbacks.performance_row(case, policy, observe=prepared.observe,
                challenge_seed=request['challenge_seed'])
        return compiled, row
    finally:
        # Bound callbacks and graph replays form cycles with Prepared.
        # Collect outside timing, including the compile-only return above.
        graph = callbacks = prepared = None
        gc.collect()


def select_cases(manifest, case_ids=None):
    """Select complete immutable case records, never a replacement manifest."""
    cases = manifest['cases']
    if case_ids is None:
        return cases
    if (not isinstance(case_ids, list) or not case_ids
            or any(not isinstance(value, str) or not value for value in case_ids)
            or len(set(case_ids)) != len(case_ids)):
        raise ValueError('Case selection must be a nonempty list of distinct IDs')
    wanted = set(case_ids)
    selected = [case for case in cases if case['case_id'] in wanted]
    if [case['case_id'] for case in selected] != case_ids:
        raise ValueError('Selected cases must be known and in full-manifest order')
    return selected


def evaluate_cases(runtime, manifest, phase, request, torch, *, case_ids=None, progress=None):
    """Return raw complete-case payloads; the caller owns final report scope."""
    if phase not in ('compile', 'correctness', 'performance'):
        raise ValueError('Unknown evaluation phase')
    cases = select_cases(manifest, case_ids)
    policy = manifest['measurement']
    compiled, rows = [], []
    for case in cases:
        event = {'phase': phase, 'case_id': case['case_id'], 'cases_completed': len(compiled),
                 'case_count': len(cases), 'state': 'preparing_full_shape', 'observed_unix': time.time()}
        if case_ids is not None:
            event['full_case_count'] = len(manifest['cases'])
            event['worker_case_selection'] = True
        if progress is not None:
            progress(event, True)
        if case['calls_per_sample'] != 1:
            raise ValueError('Each scored Kimi graph replay must represent one native call')
        receipt, row = run_case(runtime, case, phase, request, policy, torch)
        compiled.append(receipt)
        if phase != 'compile':
            rows.append(row)
        event.update(state='case_complete', cases_completed=len(compiled), observed_unix=time.time())
        if progress is not None:
            progress(event, phase != 'compile')
    return {'compiled': True, 'compiled_specializations': compiled, 'cases': rows,
            'oracle_order': 'CPU input truth before candidate; CPU output and post-input snapshots before reference'}


def validate_full_payload(payload, manifest, request):
    """Require explicit native engagement for the original complete case domain."""
    receipts = payload.get('compiled_specializations')
    expected = [case['case_id'] for case in manifest['cases']]
    if (payload.get('compiled') is not True or not isinstance(receipts, list)
            or any(not isinstance(row, dict) for row in receipts)
            or [row.get('case_id') for row in receipts] != expected):
        raise ValueError('Full compiled-case coverage is required before finalization')
    for row in receipts:
        if (row.get('source_sha256') != request['source_sha256']
                or row.get('candidate_invoked') is not True
                or row.get('independent_reference_invoked') is not True):
            raise ValueError('Full-task native engagement differs from the bound sources')


def runtime_on_gpu(manifest, request, *, build=None):
    runtime = load_runtime(manifest, request, **({'build': build} if build is not None else {}))
    import torch
    if not torch.version.hip or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('The pinned ROCm gfx950 runtime is required')
    return runtime, torch


def progress_writer(build, phase):
    def progress(event, emit):
        (build/(phase+'_progress.json')).write_text(canonical(event)+'\n')
        if emit:
            print(json.dumps(event), flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('compile', 'correctness', 'performance'))
    parser.add_argument('--request')
    parser.add_argument('--case-worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker_mode = args.case_worker is not None
    if worker_mode and not args.request:
        parser.error('--case-worker requires the original bound --request')
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    # Workers never publish, delete or replace the parent's ordinary report.
    output = None if worker_mode else build/(args.phase+'_report.json')
    if output is not None:
        output.unlink(missing_ok=True)
    try:
        manifest = strict_json((ROOT/'cases.json').read_text())
        validate_manifest(manifest)
        if manifest.get('status') != 'FROZEN_CURRENT_CAPTURE':
            raise RuntimeError('Current observed cases and fixture coverage are not sealed')
        validate_sources(ROOT, ROOT)
        request = request_for(args.phase, manifest, args.request)
        parallel = plan = None
        if (worker_mode or os.environ.get('AKA_CASE_GPU_PLAN')
                or os.environ.get('AKA_CASE_GPU_PLAN_SHA256')
                or 'case_parallelism' in request or 'gpu_group' in request):
            import case_parallel as parallel
            plan = parallel.load_plan(ROOT, manifest, request)
            if plan is None:
                raise ValueError('Explicit parallel execution requires an admitted GPU plan')
        if worker_mode:
            # The helper validates the complete parent/assignment and proves
            # one physical GPU before Torch or the native runtime is imported.
            context = parallel.enter_worker(ROOT, manifest, request, args.case_worker)
            selected = list(context.cases)
            case_ids = [case['case_id'] for case in selected]
            if canonical(selected) != canonical(select_cases(manifest, case_ids)):
                raise ValueError('Worker case records differ from the full manifest')
            private_build = Path(context.output_directory)
            if not private_build.resolve().is_relative_to(build.resolve()):
                raise ValueError('Worker output directory must remain inside task build')
            private_build.mkdir(parents=True, exist_ok=True)
            runtime, torch = runtime_on_gpu(manifest, request, build=private_build)
            payload = evaluate_cases(runtime, manifest, args.phase, request, torch, case_ids=case_ids,
                                     progress=progress_writer(private_build, args.phase))
            parallel.finish_worker(context, rows=payload['cases'],
                compiled_specializations=payload['compiled_specializations'],
                evidence={'completed_case_ids': [row['case_id'] for row in payload['compiled_specializations']],
                          'oracle_order': payload['oracle_order'], 'cpu_thread_policy': runtime.cpu_thread_policy})
            print(json.dumps({'status': 'partition_complete', 'phase': args.phase,
                              'assigned_case_count': len(case_ids), 'full_case_count': len(manifest['cases'])}))
            return
        parallel_evidence = None
        if plan is not None:
            # This parent remains CPU-only. All device work lives in the new
            # interpreters launched and joined by the shared protected helper.
            request = parallel.bind_request(request, plan)
            combined = parallel.run_workers(ROOT, manifest, request, plan, Path(__file__).resolve(), leg='framework')
            payload = {'compiled': True, 'compiled_specializations': combined['compiled_specializations'],
                       'cases': combined['rows'],
                       'oracle_order': 'CPU input truth before candidate; CPU output and post-input snapshots before reference'}
            parallel_evidence = combined['case_parallelism']
        else:
            runtime, torch = runtime_on_gpu(manifest, request)
            payload = evaluate_cases(runtime, manifest, args.phase, request, torch,
                                     progress=progress_writer(build, args.phase))
        validate_full_payload(payload, manifest, request)
        report = {'schema_version': 1, 'status': 'ok', 'request': request, **payload}
        if parallel_evidence is not None:
            report['case_parallelism'] = parallel_evidence
        report = finalize_report(report, manifest, request)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(canonical(report)+'\n')
        temporary.replace(output)
        print(json.dumps({'status': 'ok', 'phase': args.phase, 'case_count': len(manifest['cases'])}))
    except BaseException:
        if output is not None:
            output.unlink(missing_ok=True)
            output.with_suffix('.tmp').unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    main()
