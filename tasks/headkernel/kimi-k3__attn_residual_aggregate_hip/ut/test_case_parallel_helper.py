"""CPU integration with the canonical helper; only device work is simulated."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evaluation_contract import fingerprint
from test_case_parallel_entrypoint import inputs, payload


TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_kimi_actual_helper_test', TASK / 'scripts/task_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('phase', ['compile', 'correctness', 'performance'])
def test_full_task_command_uses_real_worker_receipts_and_full_reducer(tmp_path, monkeypatch, phase):
    # The shared owner materializes this exact canonical helper into ut/.
    parallel = pytest.importorskip('case_parallel', reason='canonical helper materialization pending')
    root = tmp_path / 'task'
    root.mkdir()
    full, _ = inputs(root, phase)
    for folder in ('source', 'scripts', 'ut'):
        (root / folder).mkdir()
    source = (TASK / 'source/attn_res_hip.py').read_bytes()
    (root / 'source/attn_res_hip.py').write_bytes(source)
    (root / 'scripts/task_runner.py').write_bytes((TASK / 'scripts/task_runner.py').read_bytes())
    (root / 'ut/source_guard_policy.json').write_text(json.dumps({'sources': {'source/attn_res_hip.py': {}}}))
    bindings = [{'worker_id': index, 'render_device': f'/dev/dri/renderD{128 + 8 * index}',
                 'drm_minor': 128 + 8 * index, 'kfd_node': str(index + 1),
                 'rocr_uuid': f'GPU-{index + 1:016x}', 'pci_bus_id': f'0000:01:{index:02x}.0'} for index in range(8)]
    assignments = [{'worker_id': index, 'case_ids': [case['case_id'] for case in full['cases'][index::8]]}
                   for index in range(8)]
    plan = {'schema': 'case-gpu-plan-v1', 'authorized': True, 'plan_id': 'cpu-integration-only',
            'job_id': '1', 'node': 'not-a-real-grant', 'restart_generation': 0,
            'allocation_receipt_sha256': '1' * 64, 'baseline_commit': 'a' * 40,
            'work_stop_unix': time.time() + 3600,
            'task_path': 'tasks/headkernel/kimi-k3__attn_residual_aggregate_hip',
            'manifest_sha256': fingerprint(full), 'runtime_image': full['runtime_image'],
            'bindings': bindings, 'assignments': assignments, 'assignment_sha256': fingerprint(assignments),
            'phase_timeouts': {'compile': 1800, 'correctness': 3600, 'performance': 32400},
            'resources': {'cpus_per_worker': 2, 'memory_gib_per_worker': 4,
                          'coordinator_cpus': 2, 'coordinator_memory_gib': 2},
            'allocation_capacity': {'cpus': 24, 'memory_gib': 64, 'gpus': 8}}
    plan_file = tmp_path / 'plan.json'
    plan_bytes = json.dumps(plan).encode()
    plan_file.write_bytes(plan_bytes)
    plan_file.chmod(0o444)
    monkeypatch.setenv('AKA_CASE_GPU_PLAN', str(plan_file))
    monkeypatch.setenv('AKA_CASE_GPU_PLAN_SHA256', hashlib.sha256(plan_bytes).hexdigest())
    calls, seen_requests = [], []

    def identity():
        selected = next(binding for binding in bindings if binding['rocr_uuid'] == os.environ['ROCR_VISIBLE_DEVICES'])
        calls.append(('preflight', selected['worker_id']))
        return {'hsa_gpu_uuids': [selected['rocr_uuid']], 'hip_device_count': 1,
                'hip_pci_bus_ids': [selected['pci_bus_id']]}

    def runtime(manifest, request, *, build=None):
        assert build is not None, 'coordinator initialized task runtime'
        index = int(build.name.split('-')[1])
        assert calls[-1] == ('preflight', index)
        calls.append(('runtime', index))
        seen_requests.append(copy.deepcopy(request))
        return SimpleNamespace(cpu_thread_policy={'intraop_after': 8}), object()

    def evaluate(runtime, manifest, selected_phase, request, torch, *, case_ids, progress):
        chosen = runner.select_cases(manifest, case_ids)
        progress({'phase': selected_phase, 'case_count': len(chosen), 'full_case_count': 69,
                  'cases_completed': len(chosen)}, False)
        return payload(manifest, request, selected_phase, chosen)

    def execute_workers(task_root, manifest, request, admitted_plan, entrypoint, *, leg):
        # Process creation/termination is covered by the shared helper tests.
        # Here the real task command, assignment, preflight, receipt and reducer
        # APIs run together while GPU calls and process exits are simulated.
        work = parallel.prepare_assignments(task_root, manifest, request, admitted_plan, leg=leg)
        request_path = task_root / 'build/case-parallel' / request['request_id'] / 'request.json'
        request_path.write_text(json.dumps(request))
        receipts = []
        for assignment in work:
            directory = task_root / assignment['output_directory']
            assignment_path = directory / 'assignment.json'
            assignment_path.write_text(json.dumps(assignment))
            with patch.dict(os.environ, ROCR_VISIBLE_DEVICES=assignment['expected_gpu']['rocr_uuid'],
                            HIP_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0', GPU_DEVICE_ORDINAL='0'), \
                    patch.object(sys, 'argv', [str(entrypoint), phase, '--request', str(request_path),
                                               '--case-worker', str(assignment_path)]):
                runner.main()
            receipt = directory / 'receipt.json'
            saved = json.loads(receipt.read_text())
            assert saved['status'] == 'partition_complete' and saved['full_task'] is False and saved['scoreable'] is False
            assert saved['parent_request'] == request
            assert not (task_root / 'build' / (phase + '_report.json')).exists()
            receipts.append(receipt)
        return parallel.collect_workers(manifest, request, admitted_plan, work, [0] * len(work), receipts)

    with patch.object(runner, 'ROOT', root), patch.object(runner, '__file__', str(root / 'scripts/task_runner.py')), \
            patch.object(runner, 'validate_sources'), patch.object(runner, 'runtime_on_gpu', runtime), \
            patch.object(runner, 'evaluate_cases', evaluate), patch.object(parallel, 'run_workers', execute_workers), \
            patch.object(parallel.gpu_binding, 'runtime_identity', identity), \
            patch.object(sys, 'argv', ['task_runner.py', phase]):
        runner.main()
    report = json.loads((root / 'build' / (phase + '_report.json')).read_text())
    assert len(report['compiled_specializations']) == 69 and len(report['case_parallelism']['workers']) == 8
    assert all(request == seen_requests[0] for request in seen_requests) and len(seen_requests) == 8
    assert calls == [(kind, index) for index in range(8) for kind in ('preflight', 'runtime')]
    parallel.validate_group_report(root, full, report, required=True)
    if phase == 'performance':
        assert len(report['test_cases']) == 69
        assert all(len(row['samples_ms']) == 100 for row in report['cases'])
