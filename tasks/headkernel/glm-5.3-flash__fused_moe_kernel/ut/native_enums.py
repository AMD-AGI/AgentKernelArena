"""Finite codec for the two actual AITER pybind enum types used by whole MoE."""
import sys

ENUM_CODEC='aiter-core-enum-v1'
ENUM_MODULE='aiter.jit.module_aiter_core'
ENUM_VALUES={
    'ActivationType':{'No':-1,'Silu':0,'Gelu':1,'Swiglu':2,'Situv2':3,'GeluTanh':4},
    'QuantType':{'No':0,'per_Tensor':1,'per_Token':2,'per_1x32':3,'per_1x128':4,'per_128x128':5,'per_256x128':6,'per_1024x128':7},
}


def native_enum_payload(value):
    cls=type(value);name=cls.__qualname__
    if cls.__module__!=ENUM_MODULE or name not in ENUM_VALUES:return None
    module=sys.modules.get(ENUM_MODULE)
    if module is None or getattr(module,name,None) is not cls:raise ValueError('AITER enum class identity mismatch')
    for member_name,number in ENUM_VALUES[name].items():
        member=getattr(cls,member_name,None)
        if member is None or type(int(member)) is not int or int(member)!=number:raise ValueError('Native AITER enum member/value differs from pinned declaration: '+name+'.'+member_name)
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
    if native_class.__module__!=ENUM_MODULE or native_class.__qualname__!=expected_type:raise ValueError('Wrong native enum restore class')
    result=getattr(native_class,payload['name'])
    if int(result)!=number:raise ValueError('Native enum restore value mismatch')
    return result
