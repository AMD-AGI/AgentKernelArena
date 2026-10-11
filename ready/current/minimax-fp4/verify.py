"""Check the complete pinned FP4 runtime before staging an isolated run."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(repo, *, compatible=False):
    repo = Path(repo).resolve()
    ready = json.loads((HERE / 'READY.json').read_text())
    policy = ready['compatible_inputs' if compatible else 'qualified_inputs']
    pin_file = HERE / policy['file']
    if hashlib.sha256(pin_file.read_bytes()).hexdigest() != policy['sha256']:
        raise ValueError('Input-pin manifest changed')
    pins = json.loads(pin_file.read_text())
    if pins['pinned_commit'] != policy['commit'] or pins['task_tree'] != ready['task_tree']:
        raise ValueError('Pinned qualification identities disagree')
    for name, digest in ready['entry_files_sha256'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Ready-entry helper or approval changed: ' + name)
    required = {'src/quality_measurements.py', 'src/benchmark_quality.py',
                'src/task_contract.py', 'src/tools/benchmark_quality_gate.py',
                'agents/task_validator/report_schema.py'}
    if not required <= pins['files'].keys():
        raise ValueError('Complete quality and finalization pins are required')
    for name, row in pins['files'].items():
        path = repo / name
        if row['kind'] == 'symlink':
            valid = path.is_symlink() and str(path.readlink()) == row['target']
        else:
            valid = (path.is_file() and not path.is_symlink()
                     and path.resolve().is_relative_to(repo)
                     and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'])
        if not valid:
            raise ValueError('Pinned input changed: ' + name)
    manifest = json.loads((repo / ready['fixtures']['manifest']).read_text())
    if len(manifest['assets']) != 74 or sum(row['bytes'] for row in manifest['assets']) != 2044395146:
        raise ValueError('FP4 fixture scope changed')
    return {'status': 'REVIEWED_COMPATIBLE_INPUTS_MATCH' if compatible else 'QUALIFIED_INPUTS_MATCH',
            'pinned_commit': policy['commit'], 'qualified_commit': ready['qualified_commit'],
            'task_tree': ready['task_tree'], 'files_verified': len(pins['files']),
            'new_GPU_qualification': False, 'accepted_gain': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--compatible', action='store_true',
                        help='check the separately reviewed 5854b2ac runtime pins')
    args = parser.parse_args()
    print(json.dumps(verify(args.repo, compatible=args.compatible), indent=2))
