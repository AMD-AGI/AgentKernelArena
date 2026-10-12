"""Read-only audit of the canonical timed graph's binding and retained evidence.

This audits the supported helper structure, not arbitrary Python or CUDA binaries.
It never changes a validator result or substitutes for fresh GPU qualification.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

# Admission is by exact reviewed bytes. The AST checks below explain that known
# version's binding; they are not a sound verifier for arbitrary Python rewrites.
REVIEWED_HELPER_SHA256 = frozenset({
    '40d97860647ee0a939b80c76f1de4a79758cff67fa6aa008670c42a202a82ed3',
})


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def attribute_calls(node, receiver, attribute):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attribute
            and isinstance(n.func.value, ast.Name) and n.func.value.id == receiver]


def audit_helper(source):
    raw = source.encode('utf-8') if isinstance(source, str) else source
    digest = hashlib.sha256(raw).hexdigest()
    require(digest in REVIEWED_HELPER_SHA256, 'unreviewed canonical helper bytes')
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    main = functions['benchmark_cuda_graph_or_events_samples']
    sample = functions['_graph_replay_samples']
    replay = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == '_replay_once')
    assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'graph' for t in n.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Call)
            and isinstance(assignments[0].value.func, ast.Name)
            and assignments[0].value.func.id == '_capture_graph', 'final graph binding is ambiguous')
    measured = [n for n in calls(main, '_graph_replay_samples')
                if n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == 'graph'
                and any(k.arg == 'samples' and isinstance(k.value, ast.Name)
                        and k.value.id == 'repetition' for k in n.keywords)]
    require(len(measured) == 1, 'reported samples do not use the bound final graph')
    require(len(attribute_calls(sample, 'graph', 'replay')) == 1, 'sample replay binding differs')
    require(len(attribute_calls(replay, 'graph', 'replay')) == 1
            and len([n for n in ast.walk(replay) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'replay']) == 1,
            'validation replay uses a different graph')
    require(not replay.args.args and not any(isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
            for n in ast.walk(replay)), 'validation closure rebinds captured state')
    returns = [n.value for n in ast.walk(replay) if isinstance(n, ast.Return)]
    require(len(returns) == 1 and isinstance(returns[0], ast.Name)
            and returns[0].id == 'captured_output', 'validation returns a different output')
    bindings = attribute_calls(main, 'timed_run', '_bind')
    require(len(bindings) == 1 and [ast.unparse(n) for n in bindings[0].args]
            == ['_replay_once', 'captured_output'], 'timed run binds a different replay or output')
    holder = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'captured_output' for t in n.targets)]
    require(len(holder) == 1 and ast.unparse(holder[0].value)
            == 'captured_outputs[0] if captured_outputs else None', 'captured output provenance differs')
    require(any(isinstance(n, ast.Assign) and any(ast.unparse(t) == "capture_kwargs['output_holder']"
            for t in n.targets) and ast.unparse(n.value) == 'captured_outputs'
            for n in ast.walk(main)), 'capture does not export its output holder')
    require(any(k.arg is None and isinstance(k.value, ast.Name) and k.value.id == 'capture_kwargs'
                for k in assignments[0].value.keywords), 'final capture does not receive output holder')
    return {'source_binding': 'same final graph variable and captured output holder',
            'reviewed_helper_sha256': digest,
            'measured_call_line': measured[0].lineno, 'validation_closure_line': replay.lineno,
            'binding_line': bindings[0].lineno}


def audit_evidence(helper, report_path, integrity_path, performance_path):
    import yaml
    # Check hashes and explain bindings from the same byte snapshots.
    helper_raw = Path(helper).read_bytes()
    report_raw = Path(report_path).read_bytes()
    integrity_raw = Path(integrity_path).read_bytes()
    performance_raw = Path(performance_path).read_bytes()
    helper_digest = hashlib.sha256(helper_raw).hexdigest()
    performance_digest = hashlib.sha256(performance_raw).hexdigest()
    helper_proof = audit_helper(helper_raw)
    report = yaml.safe_load(report_raw)
    integrity = json.loads(integrity_raw)
    performance = json.loads(performance_raw)
    require(integrity['status'] == 'PASS' and integrity['raw_declared_sources_unchanged'] is True,
            'retained source integrity did not pass')
    require(integrity['validation_report_sha256'] == hashlib.sha256(report_raw).hexdigest(), 'final report hash differs')
    require(integrity['harness']['before_digests']['scripts/_aka_benchmark.py'] == helper_digest
            == integrity['harness']['after_digests']['scripts/_aka_benchmark.py'], 'executed helper hash differs')
    quality = report['checks']['benchmark_integrity']['benchmark_quality']
    require(quality['status'] == 'pass' and quality['performance_report_sha256'] == performance_digest,
            'current quality/performance binding differs')
    require(performance['status'] == 'ok' and performance['cases'], 'performance report is incomplete')
    for row in performance['cases']:
        require(row['validation']['measured_graph_validated'] is True,
                'executed harness did not confirm measured replay validation')
    return {'status': 'EVIDENCE_CHAIN_AUDITED', **helper_proof,
            'helper_sha256': helper_digest, 'performance_sha256': performance_digest,
            'case_count': len(performance['cases']), 'retained_overall_status': report['overall_status'],
            'exported_graph_handles_or_per_sample_pointer_ids': False,
            'scope': 'frozen source binding plus executed assertion receipts; no result promotion'}


def write_fresh_result(output, result, evidence_paths):
    output = Path(output)
    resolved = output.resolve()
    for evidence in evidence_paths:
        evidence = Path(evidence)
        require(resolved != evidence.resolve(), 'audit output aliases an evidence input')
        if output.exists() and evidence.exists():
            require(not output.samefile(evidence), 'audit output aliases an evidence input')
    encoded = (json.dumps(result, indent=2) + '\n').encode('utf-8')
    # O_EXCL also rejects existing and dangling symlinks. O_NOFOLLOW makes the
    # final-component policy explicit; exclusivity protects a race after checks.
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('helper', 'report', 'integrity', 'performance', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = audit_evidence(args.helper, args.report, args.integrity, args.performance)
    write_fresh_result(args.out, result, (args.helper, args.report, args.integrity, args.performance))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
