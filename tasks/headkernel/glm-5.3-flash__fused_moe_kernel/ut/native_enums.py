"""Finite codec for the two actual AITER pybind enum types used by whole MoE."""
from collections.abc import Mapping
import hashlib
from importlib.machinery import ExtensionFileLoader, ModuleSpec
import os
from pathlib import Path
import sys
from types import ModuleType

ENUM_CODEC='aiter-core-enum-v1'
ENUM_MODULE='aiter.jit.module_aiter_core'
ENUM_MODULES=(ENUM_MODULE,'module_aiter_core')
# Same image binary under both import names; see provenance/NATIVE-ENUM-RELOCATION.json.
ENUM_BINARY_SHA256='58c4a636ab04c32d774218946e4c6087b3443fe4a4697130b7e826d7e935a536'
ENUM_BINARY_BYTES=661008
ENUM_VALUES={
    'ActivationType':{'No':-1,'Silu':0,'Gelu':1,'Swiglu':2,'Situv2':3,'GeluTanh':4},
    'QuantType':{'No':0,'per_Tensor':1,'per_Token':2,'per_1x32':3,'per_1x128':4,'per_128x128':5,'per_256x128':6,'per_1024x128':7},
}


def _validate_enum_members(native_class,expected_type):
    members=getattr(native_class,'__members__',None)
    if not isinstance(members,Mapping) or set(members)!=set(ENUM_VALUES[expected_type]):raise ValueError('Native enum member table differs from pinned declaration')
    for name,number in ENUM_VALUES[expected_type].items():
        member=members[name]
        if type(member) is not native_class or getattr(native_class,name,None) is not member or int(member)!=number:
            raise ValueError('Native AITER enum member/value differs from pinned declaration: '+expected_type+'.'+name)


def _validate_extension(module,module_name,expected_path):
    spec=getattr(module,'__spec__',None)
    if (not isinstance(module,ModuleType) or module.__name__!=module_name or not isinstance(spec,ModuleSpec)
            or spec.name!=module_name or not isinstance(spec.loader,ExtensionFileLoader) or spec.loader.name!=module_name):
        raise ValueError('Native enum module is not the expected extension loader')
    filename=getattr(module,'__file__',None)
    if not isinstance(filename,str):raise ValueError('Native enum extension path missing')
    path=Path(filename)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve()!=expected_path.resolve()
            or spec.origin!=filename or spec.loader.path!=filename):
        raise ValueError('Native enum extension path differs from its declared JIT location')
    raw=path.read_bytes()
    if len(raw)!=ENUM_BINARY_BYTES or hashlib.sha256(raw).hexdigest()!=ENUM_BINARY_SHA256:
        raise ValueError('Native enum extension binary differs from pinned image')


def _validate_native_class(native_class,expected_type):
    module_name=getattr(native_class,'__module__',None)
    if (expected_type not in ENUM_VALUES or module_name not in ENUM_MODULES
            or getattr(native_class,'__qualname__',None)!=expected_type or getattr(native_class,'__name__',None)!=expected_type):
        raise ValueError('Wrong native enum restore class')
    meta=type(native_class)
    if meta.__module__!='pybind11_builtins' or meta.__qualname__!='pybind11_type':raise ValueError('Expected native pybind enum class')
    module=sys.modules.get(module_name);aiter=sys.modules.get('aiter')
    if (not isinstance(aiter,ModuleType) or getattr(aiter,expected_type,None) is not native_class
            or getattr(module,expected_type,None) is not native_class):
        raise ValueError('AITER enum class identity mismatch')
    if module_name==ENUM_MODULE:
        package_file=getattr(aiter,'__file__',None)
        if not isinstance(package_file,str):raise ValueError('AITER package path missing')
        expected_path=Path(package_file).resolve().parent/'jit/module_aiter_core.so'
    else:
        location=os.environ.get('AITER_JIT_DIR')
        if not location or not Path(location).is_absolute():raise ValueError('Relocated enum requires an explicit absolute AITER_JIT_DIR')
        expected_path=Path(location)/'module_aiter_core.so'
    _validate_extension(module,module_name,expected_path)
    _validate_enum_members(native_class,expected_type)


def native_enum_payload(value):
    cls=type(value);name=cls.__qualname__
    if cls.__module__ not in ENUM_MODULES or name not in ENUM_VALUES:return None
    _validate_native_class(cls,name)
    for member_name,number in ENUM_VALUES[name].items():
        member=getattr(cls,member_name,None)
        if value==member:
            if int(value)!=number:raise ValueError('AITER enum payload disagrees with member')
            return {'kind':'enum','codec':ENUM_CODEC,'type':ENUM_MODULE+':'+name,'name':member_name,'value':number}
    raise ValueError('Unknown native AITER enum value')


def enum_number(payload,expected_type):
    if expected_type not in ENUM_VALUES:raise ValueError('Unsupported AITER enum type')
    if not isinstance(payload,dict) or set(payload)!={'kind','codec','type','name','value'}:raise ValueError('Expected finite native AITER enum payload')
    if payload['kind']!='enum' or payload['codec']!=ENUM_CODEC or payload['type']!=ENUM_MODULE+':'+expected_type:raise ValueError('AITER enum codec/type mismatch')
    if type(payload['value']) is not int or ENUM_VALUES[expected_type].get(payload['name'])!=payload['value']:raise ValueError('AITER enum name/value mismatch')
    return payload['value']


def restore_native_enum(payload,expected_type,native_class):
    number=enum_number(payload,expected_type)
    _validate_native_class(native_class,expected_type)
    result=getattr(native_class,payload['name'])
    if int(result)!=number:raise ValueError('Native enum restore value mismatch')
    return result
