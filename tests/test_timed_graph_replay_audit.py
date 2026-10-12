"""The source audit rejects a different graph, output, or callback binding."""
from pathlib import Path
import pytest
from src.tools.audit_timed_graph_replay import audit_helper

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
