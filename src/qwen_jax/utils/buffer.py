from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Any

import jaxtyping
from jax import Array


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

def resolved_fields(cls) -> Iterator[tuple[dataclasses.Field, Any]]:
    """Yield (field, resolved_annotation) pairs for a dataclass.

    Every module here uses `from __future__ import annotations` (PEP 563), so
    `field.type` is the annotation's *source text*, not the type itself. Resolve
    it against the defining module's namespace before inspecting it.
    """
    hints = typing.get_type_hints(cls, include_extras=True)
    for field in dataclasses.fields(cls):
        yield field, hints[field.name]

def is_param_or_persistent_buffer(field: dataclasses.Field, tp: Any) -> bool:
    """Check if a field is a parameter or persistent buffer.

    tp is the field's resolved annotation, as yielded by resolved_fields.
    """
    return (
        is_array_type(tp)
        # Not actually sure if we should exclude static fields? probably should?
        and not field.metadata.get('static', False)
        and not any(
            isinstance(ann, Buffer) and not ann.persistent
            for ann in collect_annotations(tp)
        )
    )

