"""Bound CPU validation parallelism without changing GPU work or check counts."""
import json
import os
from pathlib import Path

def configure_cpu_threads(torch,build):
    before=int(torch.get_num_threads());selected=min(before,8)
    if selected<1:raise RuntimeError('Invalid initial Torch CPU thread count')
    torch.set_num_threads(selected)
    after=int(torch.get_num_threads())
    if after!=selected:raise RuntimeError('Torch CPU validation thread cap did not apply')
    record={'intraop_before':before,'intraop_after':after,'maximum_intraop_threads':8,
        'interop_threads':int(torch.get_num_interop_threads()),
        'cpu_affinity_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'),
        'scope':'CPU snapshots, reductions and comparisons; GPU operator and mathematical oracle unchanged',
        'checks_or_samples_removed':False}
    build=Path(build);build.mkdir(exist_ok=True)
    (build/'cpu_thread_policy.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    return record
