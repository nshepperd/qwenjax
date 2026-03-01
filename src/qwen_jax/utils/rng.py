from typing import overload
import jax
from jaxtyping import PRNGKeyArray

@overload
def split(key: PRNGKeyArray, num: int = 2) -> PRNGKeyArray: ...

@overload
def split(key: None, num: int = 2) -> tuple[None, ...]: ...

def split(key: PRNGKeyArray | None, num: int = 2) -> PRNGKeyArray | tuple[None, ...]:
    """Split a PRNG key into num keys."""
    if key is None:
        return (None,) * num
    return jax.random.split(key, num)