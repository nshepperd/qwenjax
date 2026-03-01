import re

import jax
import jax.numpy as jnp
from jaxtyping import Array

def gather(pattern: str, array: Array, indices: Array) -> Array:
    """Gather elements from array according to pattern and indices.

    Args:
        pattern: A string pattern like "b s h d, b [s] -> b h d". If the second part contains [s] it
                indicates a one-dimensional index to gather along that dimension. The [s] dimension
                in the index is implicit (since it would be length 1).
        array: The input array to gather from, shape should match the first argument (b s h d).
        indices: The indices to gather, shape should match the second argument in the pattern (b [s]).

    Returns:
        Gathered array with the given output shape.
    """
    # Parse pattern: "b s h d, b [s] -> b h d"
    inputs_str, output_str = pattern.split('->')
    input_parts = inputs_str.split(',')
    input_dims = input_parts[0].strip().split()
    output_dims = output_str.strip().split()

    # Parse index dims, extracting the [bracketed] indexed dimension
    indexed_dim = None
    index_dims = []
    for token in input_parts[1].strip().split():
        m = re.match(r'\[(\w+)\]', token)
        if m:
            indexed_dim = m.group(1)
        else:
            index_dims.append(token)

    assert indexed_dim is not None
    assert indexed_dim in input_dims

    # Classify dimensions
    input_set, index_set = set(input_dims), set(index_dims)
    batch_dims = [d for d in output_dims if d in input_set and d in index_set]
    input_broadcast_dims = [d for d in output_dims if d in input_set and d not in index_set]
    index_broadcast_dims = [d for d in output_dims if d in index_set and d not in input_set]

    # Transpose input array to: [batch...] + [indexed] + [input_broadcast...]
    input_target = batch_dims + [indexed_dim] + input_broadcast_dims
    input_perm = [input_dims.index(d) for d in input_target]
    arr = jnp.transpose(array, input_perm) if input_perm != list(range(len(input_perm))) else array

    # Transpose index array to: [batch...] + [index_broadcast...]
    index_target = batch_dims + index_broadcast_dims
    index_perm = [index_dims.index(d) for d in index_target]
    idx = jnp.transpose(indices, index_perm) if index_perm != list(range(len(index_perm))) else indices

    # Core: scalar index into the indexed dimension (axis 0 after batch dims are vmapped away)
    def core(a, i):
        return a[i]

    fn = core

    # vmap index-broadcast dims (inner), then batch dims (outer)
    # Reversed so outermost vmap corresponds to leading axis
    for _ in reversed(index_broadcast_dims):
        fn = jax.vmap(fn, in_axes=(None, 0))
    for _ in reversed(batch_dims):
        fn = jax.vmap(fn, in_axes=(0, 0))

    result = fn(arr, idx)

    # Result is [batch...] + [index_broadcast...] + [input_broadcast...]; transpose to output order
    result_dims = batch_dims + index_broadcast_dims + input_broadcast_dims
    if result_dims != output_dims:
        output_perm = [result_dims.index(d) for d in output_dims]
        result = jnp.transpose(result, output_perm)

    return result