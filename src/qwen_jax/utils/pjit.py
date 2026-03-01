import jax
from typing import ParamSpec, TypeVar, Callable, overload

P = ParamSpec("P")
R = TypeVar("R")


@overload
def pjit(fn: Callable[P, R], **kwargs) -> Callable[P, R]: ...


@overload
def pjit(fn: None = None, **kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def pjit(
    fn: Callable[P, R] | None = None,
    **kwargs,
):
    def wrapper(f: Callable[P, R]) -> Callable[P, R]:
        return jax.jit(f, **kwargs)

    if fn is None:
        return wrapper
    else:
        return wrapper(fn)
