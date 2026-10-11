"""Complete-case worker hooks retain the protected replay and control policy."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest

from evaluation_contract import canonical, fingerprint, validate_report
import test_replay_contract as replay_fixture


TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_aggregation_case_worker_tests', TASK / 'scripts/task_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def manifest():
    return json.loads((TASK / 'cases.json').read_text())


@pytest.mark.parametrize('selection', [[], ['unknown'], [None], [1], 'one', ('one',)])
def test_invalid_case_selection_rejected(selection):
    with pytest.raises(ValueError, match='Case selection|Selected cases'):
        runner.select_cases(manifest(), selection)


def test_selection_is_exact_and_does_not_rewrite_full_manifest():
    full = manifest()
    before = canonical(full)
    selected = [full['cases'][index]['case_id'] for index in (0, 1, 3)]
    assert runner.select_cases(full) is full['cases']
    result = runner.select_cases(full, selected)
    assert [case['case_id'] for case in result] == selected
    assert all(case is full['cases'][index] for case, index in zip(result, (0, 1, 3)))
    assert canonical(full) == before and len(full['cases']) == 69
    for invalid in ([selected[0], selected[0]], list(reversed(selected))):
        with pytest.raises(ValueError):
            runner.select_cases(full, invalid)


@pytest.mark.parametrize('phase', ['compile', 'correctness', 'performance'])
def test_selected_cases_run_original_controls_and_complete_sample_vectors(phase):
    full = manifest()
    before = canonical(full)
    selected = [full['cases'][index]['case_id'] for index in (0, 1, 3)]
    request = {'schema_version': 1, 'phase': phase, 'challenge_seed': 17,
               'manifest_sha256': fingerprint(full),
               'source_sha256': {'source/attn_res_hip.py': 'synthetic-source'}}
    original_request = copy.deepcopy(request)
    records, progress, storage = [], [], []
    current = None

    class Prepared:
        def __init__(self, case):
            self.case = case
            self.callbacks, self.log, self.x, self.y = replay_fixture.FreshChecks().make()
            self.invoke = self.callbacks.replay
            self.log.clear()
            records.append((case['case_id'], self.log))
            storage.extend((weakref.ref(self), weakref.ref(self.x), weakref.ref(self.y)))

        def invoke_candidate(self):
            self.invoke()

        def observe(self):
            return self.case

        def corrupt_outputs(self):
            self.y.value = -999

        def native_engagement(self):
            return {'source_sha256': request['source_sha256'],
                    'candidate_invoked': 'replay' in self.log,
                    'independent_reference_invoked': 'reference' in self.log}

    class Graph:
        def __init__(self):
            self.invoke = current().invoke_candidate

        def replay(self):
            self.invoke()

    stream = SimpleNamespace(wait_stream=lambda other: None)
    torch = SimpleNamespace(cuda=SimpleNamespace(
        Stream=lambda: stream, current_stream=lambda: stream,
        stream=lambda value: nullcontext(), graph=lambda value: nullcontext(), CUDAGraph=Graph))

    def prepare(case):
        nonlocal current
        assert all(reference() is None for reference in storage), 'preceding case retained storage'
        prepared = Prepared(case)
        current = weakref.ref(prepared)
        return prepared

    result = runner.evaluate_cases(SimpleNamespace(prepare_case=prepare), full, phase, request, torch,
                                  case_ids=selected, progress=lambda value, emit: progress.append(copy.deepcopy(value)))
    assert all(reference() is None for reference in storage)
    assert canonical(full) == before and request == original_request
    assert [case['case_id'] for case in result['compiled_specializations']] == selected
    assert [case_id for case_id, _ in records] == selected
    assert all(event['full_case_count'] == 69 and event['case_count'] == 3 for event in progress)
    assert progress[-1]['cases_completed'] == 3
    assert not {'status', 'request', 'schema_version', 'test_cases', 'speedup', 'score'} & result.keys()
    with pytest.raises(ValueError, match='unsuccessful evaluation report'):
        validate_report(result, full, request)
    if phase == 'compile':
        assert result['cases'] == []
        assert all(log.count('replay') == log.count('reference') == 1 for _, log in records)
    elif phase == 'correctness':
        assert [row['case']['case_id'] for row in result['cases']] == selected
        assert all(row['seeds'] == [0, 1, 2] for row in result['cases'])
        assert all(row['negative_controls'] == {'no_op': True, 'wrong_output': True} for row in result['cases'])
        # Initial check, three setup checks, capture verification, three seeds,
        # and both negative controls retain the original independent oracle.
        assert all(log.count('reference') == 10 and log.count('event') == 0 for _, log in records)
    else:
        for row, (_, log) in zip(result['cases'], records):
            assert row['samples_ms'] == [0.25] * 100
            assert row['warmup_iterations'] == 10
            assert row['fresh_input_resets'] == row['output_initializations'] == row['oracle_checks'] == 100
            assert log.count('event') == 200 and log.count('event_sync') == 100
            assert log.count('reference') == 115  # Five setup/verification plus 110 policy replays.
        # A subset cannot pass the full-domain validator even with a forged
        # ordinary-success envelope; the shared reducer must supply all cases.
        with pytest.raises(ValueError, match='missing, duplicate or unexpected'):
            validate_report({'schema_version': 1, 'status': 'ok', 'request': request, **result}, full, request)


def test_case_failure_stops_the_selected_sequence_and_returns_no_payload():
    full = manifest()
    selected = [case['case_id'] for case in full['cases'][:3]]
    called = []

    def fail(runtime, case, phase, request, policy, torch):
        called.append(case['case_id'])
        raise AssertionError('oracle failure')

    with patch.object(runner, 'run_case', fail), pytest.raises(AssertionError, match='oracle failure'):
        runner.evaluate_cases(None, full, 'performance', {}, None, case_ids=selected)
    assert called == selected[:1]
