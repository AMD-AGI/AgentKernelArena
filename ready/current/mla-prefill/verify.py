"""Verify the exact qualified MLA checkout before staging or optimization."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def verify(repo):
    repo=Path(repo).resolve()
    ready=json.loads((HERE/'READY.json').read_text())
    path=HERE/'INPUT-PINS.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ready['input_pins_sha256']:
        raise ValueError('Qualified input-pin manifest changed')
    pins=json.loads(path.read_text())
    if pins['qualified_commit']!=ready['qualified_commit'] or pins['task_tree']!=ready['task_tree']:
        raise ValueError('Qualification and input identities disagree')
    for name,digest in ready['entry_files_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Current entry helper changed: '+name)
    for name,row in pins['files'].items():
        path=repo/name
        if row['kind']=='symlink':
            valid=path.is_symlink() and str(path.readlink())==row['target']
        else:
            valid=(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(repo)
                   and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'])
        if not valid:raise ValueError('Qualified input changed: '+name)
    manifest=json.loads((repo/ready['fixtures']['manifest']).read_text())
    if len(manifest['assets'])!=25 or sum(row['bytes'] for row in manifest['assets'])!=1413573052:
        raise ValueError('Qualified fixture inventory changed')
    return {'status':'CURRENT_SCOPED_INPUT_PINS_MATCH','task':ready['task'],
            'qualified_commit':ready['qualified_commit'],'files_verified':len(pins['files']),
            'later_runtime_compatibility_claimed':False,'accepted_gain':False,'GPU_actions':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,help='checkout/export of the qualified immutable commit')
    args=parser.parse_args()
    print(json.dumps(verify(args.repo),indent=2))
