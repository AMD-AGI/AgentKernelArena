#!/usr/bin/env bash
set -euo pipefail
# Use an independently allocated GPU and the repository's pinned Docker runtime.
# No task-side downloads, new credential mounts, or host Python GPU execution.
export PATH="$HOME/.local/bin:$PATH"
export MAX_JOBS=8
export AKA_DOCKER_PRIVILEGED=0
config=example_configs/validate_inference_tasks_mi355x.yaml
collect_reports() {
    python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import shutil

destination = Path('results/inference-validation')
destination.mkdir(parents=True, exist_ok=True)
records = {}
for root in Path('.').glob('workspace_inference_validation*'):
    for path in root.rglob('*'):
        if not path.is_file() or path.name not in {
            'validation_report.yaml', 'validation_summary.yaml',
            'validation_context.json', 'task_result.yaml', '.validation_complete',
        }:
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(destination / 'artifact_hashes.json').write_text(json.dumps(records, indent=2) + '\n')
print('Retained validation artifacts:', len(records), flush=True)
PY
}
trap collect_reports EXIT
test -n "${SPUR_JOB_ID:-${SLURM_JOB_ID:-}}"
make slurm-smoke SLURM_GPU_COUNT=1
make slurm-check-agents CONFIG="$config" SLURM_GPU_COUNT=1
make slurm-run CONFIG="$config" SLURM_GPU_COUNT=1 RUN_ARGS="--run-suffix inference_validation"
