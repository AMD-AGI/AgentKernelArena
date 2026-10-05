"""Unscored numerical/negative-source diagnostic before native operand import."""
import json
from pathlib import Path
import traceback
import task_runner as task


def main():
    import torch
    torch.set_num_threads(16)
    manifest=task.validate_manifest(task.strict_json((task.ROOT/'cases.json').read_text()));module=task.load_source();rows=[]
    for case in manifest['cases']:
        try:
            tensors,invoke,observe,initialize,verify,reset,reference=task.build_state(case,0,module)
            initialize();kernel=invoke();torch.cuda.synchronize();verify(reference)
            rows.append({'case':observe(),'status':'PASS','compiled_kernel':{'name':kernel.name,'hash':kernel.hash}})
        except Exception as error:
            rows.append({'case_id':case['case_id'],'status':'FAIL','error':str(error),'traceback':traceback.format_exc()});break
    result={'status':'PASS' if len(rows)==len(manifest['cases']) and all(r['status']=='PASS' for r in rows) else 'FAIL','diagnostic_only':True,'input_contract':'fp8_native_scale_v2' if manifest['cases'][0]['scalars']['fp8'] else 'bf16_generated','source_sha256':task.source_hash(),'qualification_complete':False,'cases':rows}
    path=task.ROOT/'build/generated_diagnostic.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
