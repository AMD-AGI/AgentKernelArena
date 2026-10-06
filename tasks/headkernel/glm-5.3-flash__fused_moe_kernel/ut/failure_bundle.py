"""Persist only CPU snapshots already owned by a failed verifier."""
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'trusted-tensor-failure-v1'
MAX_BUNDLE_BYTES = 2 << 30


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class LimitedWriter:
    def __init__(self,stream,limit):self.stream=stream;self.limit=limit
    def write(self,data):
        if self.stream.tell()+len(data)>self.limit:raise ValueError('Failure bundle exceeds byte limit')
        return self.stream.write(data)
    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()


def save_cpu_failure(directory,prefix,verify,truth,error,*,context,request,provenance,case,max_bytes=MAX_BUNDLE_BYTES):
    import torch
    directory=Path(directory)
    manifest_path=directory/(prefix+'.tensor_failure.json')
    bundle_path=directory/(prefix+'.tensor_failure.pt')
    record={'schema':SCHEMA,'status':'unavailable','context':context,'enclosing_request':request,
            'enclosing_request_sha256':json_sha(request),'provenance':provenance,
            'provenance_sha256':json_sha(provenance),'case':case,'case_sha256':json_sha(case),
            'error_type':type(error).__name__,'error':str(error),'maximum_bundle_bytes':max_bytes,
            'tensor_manifest':[],'bundle':None}
    try:
        frame_values=None;trace=error.__traceback__
        while trace is not None:
            if trace.tb_frame.f_code is verify.__code__:
                frame_values=trace.tb_frame.f_locals
                break
            trace=trace.tb_next
        if frame_values is None:raise ValueError('Exact verifier frame is unavailable')
        payload={'truth_inputs':truth.get('inputs'),'observed_inputs':frame_values.get('observed_inputs'),
                 'actual':frame_values.get('actual'),'expected':frame_values.get('expected')}
        record['missing_snapshots']=[name for name,value in payload.items() if value is None]
        tensors=[];storages={}
        def visit(value,path):
            if value is None:return None
            if type(value) is dict:
                if any(type(key) is not str for key in value):raise ValueError('Input snapshot keys must be strings')
                return {key:visit(value[key],path+[key]) for key in sorted(value)}
            elif type(value) is torch.Tensor:
                if value.device.type!='cpu' or value.layout is not torch.strided:
                    raise ValueError('Failure persistence accepts only existing dense CPU tensors')
                storage=value.untyped_storage();key=storage._cdata
                storages.setdefault(key,{'id':'s'+str(len(storages)),'bytes':storage.nbytes()})
                # Tensor attributes can themselves contain device tensors or
                # arbitrary objects. Keep JSON attributes in metadata and save
                # a detached CPU view sharing the same storage, never those objects.
                attributes=json.loads(json.dumps(value.__dict__,allow_nan=False))
                view=value.detach()
                if view.__dict__:raise ValueError('Detached tensor unexpectedly retains Python attributes')
                tensors.append((path,view,storages[key],value.requires_grad,attributes))
                return view
            else:raise ValueError('Unsupported object in CPU verifier snapshots')
        payload=visit(payload,[])
        if not tensors:raise ValueError('No already-owned CPU verifier tensors are available')
        record['unique_storage_bytes']=sum(item['bytes'] for item in storages.values())
        if record['unique_storage_bytes']>max_bytes:raise ValueError('Failure tensor storage exceeds byte limit')
        # Expanded/overlapping views can contain far more logical elements
        # than backing storage. Bound all hashing materialization first.
        record['logical_tensor_bytes']=sum(value.numel()*value.element_size() for _,value,_,_,_ in tensors)
        if record['logical_tensor_bytes']>max_bytes:raise ValueError('Failure tensor logical bytes exceed byte limit')
        for path,value,storage,requires_grad,attributes in tensors:
            raw=value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            record['tensor_manifest'].append({'path':path,'dtype':str(value.dtype),'shape':list(value.shape),
                'stride':list(value.stride()),'storage_offset':value.storage_offset(),'storage_id':storage['id'],
                'storage_bytes':storage['bytes'],'logical_bytes':value.numel()*value.element_size(),
                'requires_grad':requires_grad,'python_attributes':attributes,
                'logical_sha256':hashlib.sha256(memoryview(raw)).hexdigest()})
        with bundle_path.open('xb') as stream:
            torch.save(payload,LimitedWriter(stream,max_bytes));stream.flush();os.fsync(stream.fileno())
        record['status']='complete' if not record['missing_snapshots'] else 'partial'
    except Exception as capture_error:
        record['status']='capture_error'
        record['capture_error']={'type':type(capture_error).__name__,'message':str(capture_error)}
    if bundle_path.is_file():
        record['bundle']={'file':bundle_path.name,'bytes':bundle_path.stat().st_size,'sha256':file_sha(bundle_path),
                          'complete':record['status'] in ('complete','partial')}
    record['tensor_manifest_sha256']=json_sha(record['tensor_manifest'])
    with manifest_path.open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return {'status':record['status'],'manifest_file':manifest_path.name,'manifest_sha256':file_sha(manifest_path),
            'bundle':record['bundle']}
