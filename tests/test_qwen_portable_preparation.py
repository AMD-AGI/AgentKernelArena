"""Original Qwen captures use the registered whole-object fixture protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('qwen_portable_prepare', ROOT/'tools/prepare_headkernel_run.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
IMAGE = 'docker.io/example/runtime@sha256:' + 'a' * 64
REMOTE = 'oci:bucket/original-captures'


@pytest.fixture
def package(tmp_path):
    arena = tmp_path/'arena'
    task = arena/'tasks/headkernel/example'
    (task/'ut').mkdir(parents=True)
    (arena/'tools').mkdir()
    (task/'config.yaml').write_text(yaml.safe_dump({'headkernel':{'docker':IMAGE}}))
    content = b'original captured values and routing'
    digest = hashlib.sha256(content).hexdigest()
    (task/'ut/meta.json').write_text(json.dumps({'reference_io_sha256':digest}))
    manifest = {'schema_version':1, 'artifacts':[{
        'task':'example', 'path':'example/ut/reference_io.pt',
        'size_bytes':len(content), 'sha256':digest}],
        'tasks_without_persistent_fixtures':[], 'oci_storage':{'remote_root':REMOTE}}
    path = arena/prepare.LEGACY_MANIFEST
    path.write_text(json.dumps(manifest))
    row = {'task':'headkernel/example', 'scope':'refreshed_sg520', 'fixture_manifest':None,
           'preparation_protocol':prepare.LEGACY_PROTOCOL,
           'legacy_artifact_manifest':prepare.LEGACY_MANIFEST,
           'legacy_artifact_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    mirror = tmp_path/'mirror'
    (mirror/'example/ut').mkdir(parents=True)
    (mirror/'example/ut/reference_io.pt').write_bytes(content)
    return arena, task, row, mirror, content


def stage(package, **kwargs):
    arena, _task, row, mirror, _content = package
    return prepare.prepare_legacy_task(arena, row, runtime_image=IMAGE,
        use_manifest_oci_prefixes=False, fixture_local_mirror=mirror, **kwargs)


def test_original_fixture_is_installed_and_verified_without_rewriting_code(package):
    arena, task, row, mirror, content = package
    original_config = (task/'config.yaml').read_bytes()
    result = stage(package)
    assert (task/'ut/reference_io.pt').read_bytes() == content
    assert (task/'config.yaml').read_bytes() == original_config
    assert result['artifacts'] == [{'path':'example/ut/reference_io.pt',
        'bytes':len(content), 'sha256':hashlib.sha256(content).hexdigest()}]
    assert result['workload_scope'] == 'preserved_committed_task_contract'
    assert result['qualification'] is False


def test_scope_change_alone_cannot_omit_legacy_fixtures(package):
    arena, task, row, _mirror, _content = package
    del row['preparation_protocol']
    assert prepare.requires_legacy_protocol(arena, row)
    with pytest.raises(ValueError, match='explicit headkernel_artifacts_v1'):
        stage(package)
    assert not (task/'ut/reference_io.pt').exists()


@pytest.mark.parametrize('change', ['manifest_pin', 'runtime', 'external_manifest', 'manifest_path'])
def test_protocol_identity_mismatches_reject_before_install(package, change):
    arena, task, row, _mirror, _content = package
    if change == 'manifest_pin':
        row['legacy_artifact_manifest_sha256'] = '0'*64
    elif change == 'runtime':
        (task/'config.yaml').write_text(yaml.safe_dump({'headkernel':{'docker':'other-image'}}))
    elif change == 'external_manifest':
        row['fixture_manifest'] = 'tasks/headkernel/example/EXTERNAL-MANIFEST.json'
    else:
        row['legacy_artifact_manifest'] = '../outside.json'
    with pytest.raises(ValueError):
        stage(package)
    assert not (task/'ut/reference_io.pt').exists()


def test_wrong_original_bytes_are_rejected(package):
    _arena, task, _row, mirror, content = package
    (mirror/'example/ut/reference_io.pt').write_bytes(b'x'*len(content))
    with pytest.raises(prepare.legacy_artifacts.ArtifactError, match='SHA-256 mismatch'):
        stage(package)
    assert not (task/'ut/reference_io.pt').exists()


def test_task_metadata_must_match_registered_original_hash(package):
    _arena, task, _row, _mirror, _content = package
    (task/'ut/meta.json').write_text(json.dumps({'reference_io_sha256':'0'*64}))
    with pytest.raises(prepare.legacy_artifacts.ArtifactError, match='manifest SHA-256 disagrees'):
        stage(package)
    assert not (task/'ut/reference_io.pt').exists()


def test_oci_path_delegates_to_the_existing_original_artifact_downloader(package):
    arena, task, row, mirror, content = package
    def download(artifacts, suite, remote):
        assert remote == REMOTE
        return prepare.legacy_artifacts.prepare_artifacts(artifacts, suite, mirror=mirror)
    with patch.object(prepare.legacy_artifacts, 'download_artifacts', side_effect=download) as operation:
        result = prepare.prepare_legacy_task(arena, row, runtime_image=IMAGE,
            use_manifest_oci_prefixes=True)
    operation.assert_called_once()
    assert (task/'ut/reference_io.pt').read_bytes() == content
    assert result['fixture_oci_prefix'] == REMOTE


def test_explicit_registered_fixture_free_task_needs_no_tensor(package):
    arena, task, row, _mirror, _content = package
    path = arena/prepare.LEGACY_MANIFEST
    path.write_text(json.dumps({'schema_version':1, 'artifacts':[],
        'tasks_without_persistent_fixtures':[{'task':'example','reason':'generated legal inputs'}],
        'oci_storage':{'remote_root':REMOTE}}))
    row['legacy_artifact_manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = stage(package)
    assert result['artifacts'] == []
    assert result['inputs'] == 'explicitly_registered_fixture_free_task'
    assert not (task/'ut/reference_io.pt').exists()


def test_trusted_contract_is_not_reclassified_as_legacy(package):
    arena, task, row, _mirror, _content = package
    (task/'config.yaml').write_text(yaml.safe_dump({
        'headkernel':{'docker':IMAGE}, 'trusted_evaluation':{'schema_version':1}}))
    assert not prepare.requires_legacy_protocol(arena, row)
    with pytest.raises(ValueError, match='own staging protocol'):
        stage(package)
