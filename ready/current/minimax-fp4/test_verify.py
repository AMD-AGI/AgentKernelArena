"""Both quality admission and policy remain pinned in the compatible entry."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location('current_fp4_pin_verifier', HERE / 'verify.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.mark.parametrize('relative', ['src/quality_measurements.py', 'src/benchmark_quality.py'])
def test_quality_changes_reject_even_when_task_tree_is_unchanged(tmp_path, relative):
    pins = json.loads((HERE / 'COMPATIBLE-INPUT-PINS.json').read_text())
    for name, row in pins['files'].items():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row['kind'] == 'symlink':
            destination.symlink_to(row['target'])
        else:
            shutil.copyfile(REPO / name, destination)
    assert verifier.verify(tmp_path, compatible=True)['files_verified'] == 257
    changed = tmp_path / relative
    changed.write_bytes(changed.read_bytes() + b'\n# changed after compatibility review\n')
    with pytest.raises(ValueError, match='Pinned input changed: ' + relative):
        verifier.verify(tmp_path, compatible=True)
