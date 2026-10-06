"""Finite payloads and fail-closed native enum relocation admission."""
from enum import IntEnum
import importlib.util
from importlib.machinery import ExtensionFileLoader, ModuleSpec, SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

TASK=Path(__file__).resolve().parents[1]/'tasks/headkernel/glm-5.3-flash__fused_moe_kernel'
spec=importlib.util.spec_from_file_location('moe_native_enums',TASK/'ut/native_enums.py')
codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)


def payload(kind,name,number):
    return {'kind':'enum','codec':'aiter-core-enum-v1','type':'aiter.jit.module_aiter_core:'+kind,'name':name,'value':number}


@pytest.mark.parametrize('kind,name,number',[(kind,name,number) for kind,values in codec.ENUM_VALUES.items() for name,number in values.items()])
def test_all_canonical_payload_values_remain_exact(kind,name,number):
    assert codec.enum_number(payload(kind,name,number),kind)==number


@pytest.mark.parametrize('change',[{'type':'module_aiter_core:ActivationType'},{'value':False},{'value':'0'},{'value':999},{'name':'Unknown'},{'codec':'unreviewed'},{'extra':1}])
def test_runtime_alias_does_not_expand_the_serialized_payload(change):
    value=payload('ActivationType','Silu',0);value.update(change)
    with pytest.raises(ValueError):codec.enum_number(value,'ActivationType')


@pytest.mark.parametrize('module_name',['module_aiter_core','aiter.jit.module_aiter_core','foreign.module'])
def test_python_class_claiming_native_name_is_rejected_without_module_spoofing(module_name):
    foreign=IntEnum('ActivationType',codec.ENUM_VALUES['ActivationType'],module=module_name)
    with pytest.raises(ValueError):codec.restore_native_enum(payload('ActivationType','Silu',0),'ActivationType',foreign)
    if module_name in codec.ENUM_MODULES:
        with pytest.raises(ValueError):codec.native_enum_payload(foreign.Silu)
    else:assert codec.native_enum_payload(foreign.Silu) is None


@pytest.mark.parametrize('kind',codec.ENUM_VALUES)
def test_complete_member_table_is_checked_even_after_the_requested_member(kind):
    # These Python enums exercise only the table checker; public native-class
    # admission rejects them, as the separate test above demonstrates.
    values=dict(codec.ENUM_VALUES[kind]);table=IntEnum(kind,values)
    codec._validate_enum_members(table,kind)
    last=next(reversed(values));values[last]=999
    altered=IntEnum(kind,values)
    with pytest.raises(ValueError,match='member/value'):codec._validate_enum_members(altered,kind)


@pytest.mark.parametrize('change',['extra','missing','wrong_type'])
def test_extra_missing_and_foreign_enum_members_are_rejected(change):
    values=dict(codec.ENUM_VALUES['ActivationType'])
    if change=='extra':values['Extra']=50
    elif change=='missing':values.pop('GeluTanh')
    if change=='wrong_type':
        class Wrong:
            __members__={name:number for name,number in values.items()}
        candidate=Wrong
    else:candidate=IntEnum('ActivationType',values)
    with pytest.raises(ValueError):codec._validate_enum_members(candidate,'ActivationType')


def extension_record(path,*,name='module_aiter_core'):
    # A module record is sufficient to test rejection; it is never installed
    # into sys.modules or used as a successful native-class substitute.
    module=ModuleType(name);module.__file__=str(path)
    module.__spec__=ModuleSpec(name,ExtensionFileLoader(name,str(path)),origin=str(path))
    return module


@pytest.mark.parametrize('mutation',['binary','symlink','wrong_location','source_loader','spec_name','loader_name','origin','loader_path'])
def test_foreign_binary_location_or_loader_is_rejected(tmp_path,mutation):
    path=tmp_path/'module_aiter_core.so';path.write_bytes(b'not the pinned image binary')
    module=extension_record(path);expected=path
    if mutation=='symlink':
        target=tmp_path/'other.so';path.rename(target);path.symlink_to(target)
    elif mutation=='wrong_location':expected=tmp_path/'another/module_aiter_core.so'
    elif mutation=='source_loader':module.__spec__.loader=SourceFileLoader('module_aiter_core',str(path))
    elif mutation=='spec_name':module.__spec__.name='foreign'
    elif mutation=='loader_name':module.__spec__.loader.name='foreign'
    elif mutation=='origin':module.__spec__.origin=str(tmp_path/'foreign.so')
    elif mutation=='loader_path':module.__spec__.loader.path=str(tmp_path/'foreign.so')
    with pytest.raises(ValueError):codec._validate_extension(module,'module_aiter_core',expected)


def test_unrelated_values_remain_outside_the_codec():
    assert codec.native_enum_payload(object()) is None
