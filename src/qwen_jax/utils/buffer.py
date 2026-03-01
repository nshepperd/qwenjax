import dataclasses
import types
from jax import Array
import typing
from dataclasses import dataclass
from typing import Annotated

import jaxtyping


@dataclass
class Buffer:
    persistent: bool = True

def collect_annotations(tp):
    if typing.get_origin(tp) is Annotated:
        args = typing.get_args(tp)
        yield from args[1:]
        yield from collect_annotations(args[0])
    elif hasattr(tp, '__args__'):
        for arg in tp.__args__:
            yield from collect_annotations(arg)

def is_array_type(tp):
    if tp is Array:
        return True
    elif typing.get_origin(tp) is Annotated:
        args = typing.get_args(tp)
        return is_array_type(args[0])
    elif typing.get_origin(tp) in (types.UnionType, typing.Union):
        # Mainly want for things like Array | None which are common.
        args = typing.get_args(tp)
        return any(is_array_type(arg) for arg in args)
    elif isinstance(tp, jaxtyping._array_types._MetaAbstractArray):
        return True
    return False

def is_param_or_persistent_buffer(field: dataclasses.Field) -> bool:
    """Check if a field is a parameter or persistent buffer."""
    return (
        is_array_type(field.type)
        # Not actually sure if we should exclude static fields? probably should?
        and not field.metadata.get('static', False) 
        and not any(
            isinstance(ann, Buffer) and not ann.persistent
            for ann in collect_annotations(field.type)
        )
    )

