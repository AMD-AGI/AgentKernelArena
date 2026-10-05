"""Reject historical evaluation until the current native dispatch is resolved."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['compile','correctness','performance']);args=parser.parse_args()
    mapping=json.loads((ROOT/'CURRENT-MAPPING.json').read_text())
    report={'status':'blocked','phase':args.phase,'current_mapping_status':mapping['status'],
        'compiled':False,'test_cases':[],'error':'Historical scale-copy source/harness cannot qualify current GLM dispatch. Resolve the completed native capture; retire the task if no materialized copy exists, or rebuild from actual copy fixtures.'}
    path=ROOT/'build'/(args.phase+'_report.json');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['error']);raise SystemExit(4)
if __name__=='__main__':main()
