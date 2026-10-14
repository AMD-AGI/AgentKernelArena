"""Host snapshot races use private markers, never real host secrets."""
import os
from pathlib import Path

import pytest

from src.serving_snapshot import read_file_at, read_tree_at, workspace_at


def fixture_tree(tmp_path):
    root = tmp_path / 'checkout'
    source = root / 'run/candidate/source'
    source.mkdir(parents=True)
    (source / 'kernel.py').write_text('safe implementation')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'kernel.py').write_text('private marker')
    return root, source, outside


@pytest.mark.parametrize('swap', ['leaf', 'bundle', 'ancestor'])
def test_symlink_swap_at_open_cannot_copy_outside_bytes(tmp_path, monkeypatch, swap):
    root, source, outside = fixture_tree(tmp_path)
    original_open = os.open
    changed = False
    def racing_open(path, flags, *args, **kwargs):
        nonlocal changed
        trigger = {'leaf': 'kernel.py', 'bundle': 'source', 'ancestor': 'candidate'}[swap]
        if not changed and str(path) == trigger:
            changed = True
            target = {'leaf': source / 'kernel.py', 'bundle': source, 'ancestor': source.parent}[swap]
            target.rename(target.with_name(target.name + '.saved'))
            target.symlink_to(outside / 'kernel.py' if swap == 'leaf' else outside)
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', racing_open)
    with pytest.raises(ValueError, match='symlink'):
        with workspace_at(root, Path('run/candidate')) as workspace:
            read_tree_at(workspace, 'source', max_bytes=1000)
    assert changed


def test_renamed_ancestor_does_not_redirect_an_open_workspace(tmp_path):
    root, source, outside = fixture_tree(tmp_path)
    with workspace_at(root, Path('run/candidate')) as workspace:
        source.parent.rename(source.parent.with_name('saved'))
        source.parent.symlink_to(outside)
        assert read_file_at(workspace, 'source/kernel.py', max_bytes=1000) == b'safe implementation'


def test_mutating_open_file_is_rejected(tmp_path, monkeypatch):
    root, source, _ = fixture_tree(tmp_path)
    original_read = os.read
    def racing_read(fd, size):
        data = original_read(fd, size)
        if data:
            (source / 'kernel.py').write_text('different implementation')
        return data
    monkeypatch.setattr(os, 'read', racing_read)
    with workspace_at(root, Path('run/candidate')) as workspace:
        with pytest.raises(ValueError, match='changed'):
            read_file_at(workspace, 'source/kernel.py', max_bytes=1000)


def test_native_multifile_snapshot_and_limits(tmp_path):
    root, source, _ = fixture_tree(tmp_path)
    (source / 'native').mkdir()
    (source / 'native/kernel.hip').write_bytes(b'hip source')
    with workspace_at(root, Path('run/candidate')) as workspace:
        assert read_tree_at(workspace, 'source', max_bytes=1000) == {
            'kernel.py': b'safe implementation', 'native/kernel.hip': b'hip source'}
        with pytest.raises(ValueError, match='budget|bounded'):
            read_tree_at(workspace, 'source', max_bytes=5)
        with pytest.raises(ValueError, match='budget'):
            read_tree_at(workspace, 'source', max_bytes=1000, max_files=1)


def test_special_files_and_hardlinks_are_not_read(tmp_path):
    root, source, outside = fixture_tree(tmp_path)
    (source / 'kernel.py').unlink()
    os.mkfifo(source / 'kernel.py')
    with workspace_at(root, Path('run/candidate')) as workspace:
        with pytest.raises(ValueError, match='regular'):
            read_tree_at(workspace, 'source', max_bytes=1000)
        (source / 'kernel.py').unlink()
        os.link(outside / 'kernel.py', source / 'kernel.py')
        with pytest.raises(ValueError, match='regular'):
            read_tree_at(workspace, 'source', max_bytes=1000)
