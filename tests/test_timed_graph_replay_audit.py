"""The source audit rejects a different graph, output, or callback binding."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import pytest
import yaml
from src.tools.audit_timed_graph_replay import audit_helper, write_fresh_result

HELPER = Path(__file__).resolve().parents[1] / 'src/tools/perf/aka_benchmark.py'


def test_canonical_timing_and_validation_share_graph_and_output():
    proof = audit_helper(HELPER.read_text())
    assert proof['measured_call_line'] < proof['validation_closure_line'] < proof['binding_line']


@pytest.mark.parametrize('original,replacement', [
    ('                    graph.replay()\n', '                    estimate_graph.replay()\n'),
    ('                return captured_output\n', '                return other_output\n'),
    ('timed_run._bind(_replay_once, captured_output)', 'timed_run._bind(other_replay, captured_output)'),
    ('timed_run._bind(_replay_once, captured_output)', 'timed_run._bind(_replay_once, other_output)'),
])
def test_wrong_graph_or_output_cannot_supply_positive_source_evidence(original, replacement):
    source = HELPER.read_text()
    assert original in source
    with pytest.raises(ValueError):
        audit_helper(source.replace(original, replacement, 1))


@pytest.mark.parametrize('rebind', [
    '        graph, = [estimate_graph]\n',
    '        graph: object = estimate_graph\n',
    '        (graph := estimate_graph)\n',
])
def test_rebinding_that_bypassed_structural_checks_is_unreviewed(rebind):
    source = HELPER.read_text()
    marker = '        if timed_run is not None:\n            captured_output ='
    assert marker in source
    source = source.replace(marker, rebind + marker, 1)
    with pytest.raises(ValueError, match='unreviewed canonical helper bytes'):
        audit_helper(source)


def test_unknown_even_semantically_equivalent_helper_requires_review():
    with pytest.raises(ValueError, match='unreviewed canonical helper bytes'):
        audit_helper(HELPER.read_bytes() + b'\n# unreviewed version\n')


@pytest.fixture
def retained_evidence(tmp_path):
    helper = tmp_path / 'helper.py'
    helper.write_bytes(HELPER.read_bytes())
    performance = tmp_path / 'performance.json'
    performance.write_text(json.dumps({'status': 'ok', 'cases': [
        {'validation': {'measured_graph_validated': True}}]}))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = tmp_path / 'report.yaml'
    report.write_text(yaml.safe_dump({'overall_status': 'WARN', 'checks': {
        'benchmark_integrity': {'benchmark_quality': {'status': 'pass',
            'performance_report_sha256': digest(performance)}}}}))
    integrity = tmp_path / 'integrity.json'
    pin = {'scripts/_aka_benchmark.py': digest(helper)}
    integrity.write_text(json.dumps({'status': 'PASS', 'raw_declared_sources_unchanged': True,
        'validation_report_sha256': digest(report),
        'harness': {'before_digests': pin, 'after_digests': pin}}))
    paths = {'helper': helper, 'report': report, 'integrity': integrity, 'performance': performance}
    return paths, {p: p.read_bytes() for p in paths.values()}


def run_cli(paths, output):
    tool = HELPER.parents[1] / 'audit_timed_graph_replay.py'
    args = [sys.executable, '-I', '-B', str(tool)]
    for name, path in paths.items():
        args += ['--' + name, str(path)]
    return subprocess.run(args + ['--out', str(output)], capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize('kind', ['same_report', 'symlink_report', 'hardlink_report',
                                  'existing_file', 'dangling_symlink'])
def test_actual_cli_cannot_overwrite_evidence_or_existing_paths(retained_evidence, tmp_path, kind):
    paths, before = retained_evidence
    output = tmp_path / 'out.json'
    if kind == 'same_report':
        output = paths['report']
    elif kind == 'symlink_report':
        output.symlink_to(paths['report'])
    elif kind == 'hardlink_report':
        output.hardlink_to(paths['report'])
    elif kind == 'existing_file':
        output.write_bytes(b'preserve existing output')
    else:
        output.symlink_to(tmp_path / 'absent')
    result = run_cli(paths, output)
    assert result.returncode != 0
    assert all(p.read_bytes() == raw for p, raw in before.items())
    if kind == 'existing_file':
        assert output.read_bytes() == b'preserve existing output'
    if kind.endswith('symlink') or kind == 'symlink_report':
        assert output.is_symlink()


def test_actual_cli_creates_new_output_once_and_keeps_warn(retained_evidence, tmp_path):
    paths, before = retained_evidence
    output = tmp_path / 'fresh.json'
    result = run_cli(paths, output)
    assert result.returncode == 0, result.stderr
    original_output = output.read_bytes()
    assert json.loads(original_output)['retained_overall_status'] == 'WARN'
    assert run_cli(paths, output).returncode != 0
    assert output.read_bytes() == original_output
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_exclusive_creation_rejects_racing_output(monkeypatch, tmp_path):
    output = tmp_path / 'race.json'
    real_open = os.open
    def race(path, flags, mode):
        assert flags & os.O_EXCL and flags & os.O_NOFOLLOW
        output.write_bytes(b'created by another writer')
        return real_open(path, flags, mode)
    monkeypatch.setattr(os, 'open', race)
    with pytest.raises(FileExistsError):
        write_fresh_result(output, {'status': 'audit'}, [])
    assert output.read_bytes() == b'created by another writer'
