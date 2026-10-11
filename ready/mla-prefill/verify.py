#!/usr/bin/env python3
"""Check the qualified starter and its one-task entry configs without GPU work."""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
pins = json.loads((HERE / 'INPUT-PINS.json').read_text())
ready = json.loads((HERE / 'READY.json').read_text())
for name, expected in {**pins['files'], **pins.get('post_evaluation_files', {})}.items():
    path = ROOT / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit('Missing or symlinked qualified input: ' + name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Qualified starter input changed: ' + name)
for name in ('entry_config', 'validator_config'):
    config = yaml.safe_load((ROOT / ready[name]).read_text())
    if config['tasks'] != ready['ready_tasks'] or config['target_gpu_model'] != 'MI355X':
        raise SystemExit('Ready entry scope changed: ' + ready[name])
manifest = json.loads((ROOT / ready['fixtures']['manifest']).read_text())
if manifest['oci_prefix'] != ready['fixtures']['oci_prefix']:
    raise SystemExit('Fixture prefix differs from the qualified pin')
if len(manifest['assets']) != 25 or sum(row['bytes'] for row in manifest['assets']) != 1413573052:
    raise SystemExit('Fixture inventory differs from the qualified pin')
print(json.dumps({'status': 'QUALIFIED_STARTER_PINS_MATCH',
                  'files_verified': len(pins['files']),
                  'post_evaluation_files_verified': len(pins.get('post_evaluation_files', {})),
                  'tasks': ready['ready_tasks'], 'cases': ready['case_ids'],
                  'qualified_commit': ready['qualified_commit'],
                  'GPU_actions': False}, indent=2))
