"""Public v2 task coverage and explicit limits of the audited Forge engine."""
from pathlib import Path

import pytest

from agents.forge.adapter import ForgeRunError, require_supported_backend
from agents.forge.common import _infer_backend
from src.task_spec import TaskConfigError, TaskSpec, load_task_spec

ROOT = Path(__file__).resolve().parents[1]
# CPU fixture for audited Hyperloom 0425bde3, not the runtime's source of truth.
# Production launch uses the actual registry returned by the source-pinned probe.
KNOWN_BACKENDS = {'aiter', 'ck', 'flydsl', 'fusion', 'gluon', 'hip', 'hipblaslt', 'triton'}
EXPLICITLY_UNSUPPORTED = {'tilelang': 'No native backend in the pinned KernelForge release'}


def test_every_task_is_validated_and_its_forge_capability_is_explicit():
    paths = sorted((ROOT/'tasks').rglob('config.yaml'))
    assert paths, 'No task configs found'
    checked = []
    unsupported = set()
    for path in paths:
        task_id = path.parent.relative_to(ROOT/'tasks').as_posix()
        # No exception-catching skips: malformed or unmigrated configs fail here.
        spec = load_task_spec(path, task_id=task_id)
        language = _infer_backend(spec.to_mapping())
        assert language == spec.candidate.language
        if language in EXPLICITLY_UNSUPPORTED:
            unsupported.add(language)
            with pytest.raises(ForgeRunError, match=f'has no {language} backend'):
                require_supported_backend(spec, {'backends': sorted(KNOWN_BACKENDS)})
        else:
            assert require_supported_backend(spec, {'backends': sorted(KNOWN_BACKENDS)}) == language
        checked.append(task_id)
    assert len(checked) == len(paths)
    assert len(set(checked)) == len(paths)
    assert unsupported == set(EXPLICITLY_UNSUPPORTED), 'Review stale capability exclusions'
    assert not set(EXPLICITLY_UNSUPPORTED) & KNOWN_BACKENDS


def _config(language='triton'):
    return {'schema_version': 2, 'candidate': {'language': language, 'editable': ['kernel.py']},
            'evaluation': {'runner': ['python3', 'evaluate.py']}}


@pytest.mark.parametrize('language', ['hip', 'triton', 'flydsl', 'tilelang'])
def test_v2_backend_is_declared_language(language):
    assert _infer_backend(_config(language)) == language


@pytest.mark.parametrize('changes', [{'task_type': 'triton2triton'},
                                    {'kernel_identity': {'kernel_kind': 'flydsl'}},
                                    {'candidate': {'editable': ['kernel.py']}}])
def test_invalid_v2_contract_is_not_recovered_using_legacy_hints(changes):
    with pytest.raises(TaskConfigError):
        _infer_backend({**_config(), **changes})


def test_capability_check_uses_reported_registry_without_aliases():
    spec = TaskSpec.from_mapping(_config('tilelang'), task_id='arbitrary/operator')
    with pytest.raises(ForgeRunError, match='has no tilelang backend'):
        require_supported_backend(spec, {'backends': ['flydsl', 'triton']})
    # If a future audited upstream actually supplies it, preserve its name.
    assert require_supported_backend(spec, {'backends': ['tilelang']}) == 'tilelang'
