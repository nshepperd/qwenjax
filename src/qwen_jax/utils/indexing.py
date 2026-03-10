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
    if '->' not in pattern:
        raise ValueError(
            f"Pattern must contain '->' to separate inputs from output, got: {pattern!r}"
        )
    inputs_str, output_str = pattern.split('->')

    input_parts = inputs_str.split(',')
    if len(input_parts) != 2:
        raise ValueError(
            f"Pattern must have exactly two comma-separated inputs (array dims and index dims), "
            f"got {len(input_parts)}: {pattern!r}"
        )

    input_dims = input_parts[0].strip().split()
    output_dims = output_str.strip().split()

    if not input_dims:
        raise ValueError(f"Array dimensions (first input) cannot be empty in pattern: {pattern!r}")

    # Parse index dims, extracting the [bracketed] indexed dimension
    indexed_dim = None
    index_dims = []
    index_tokens = input_parts[1].strip().split()
    if not index_tokens:
        raise ValueError(f"Index dimensions (second input) cannot be empty in pattern: {pattern!r}")

    for token in index_tokens:
        m = re.match(r'\[(\w+)\]', token)
        if m:
            if indexed_dim is not None:
                raise ValueError(
                    f"Pattern must have exactly one [bracketed] index dimension, "
                    f"found both [{indexed_dim}] and {token} in: {pattern!r}"
                )
            indexed_dim = m.group(1)
        else:
            if not re.match(r'\w+$', token):
                raise ValueError(f"Invalid dimension name {token!r} in pattern: {pattern!r}")
            index_dims.append(token)

    if indexed_dim is None:
        raise ValueError(
            f"Index dimensions must contain one [bracketed] dimension to gather along, "
            f"e.g. 'b [s]', got: {input_parts[1].strip()!r}"
        )
    if indexed_dim not in input_dims:
        raise ValueError(
            f"Indexed dimension [{indexed_dim}] must appear in the array dimensions {input_dims}, "
            f"in pattern: {pattern!r}"
        )

    # Validate output dims reference known dimensions
    all_known = set(input_dims) | set(index_dims)
    unknown_output = [d for d in output_dims if d not in all_known]
    if unknown_output:
        raise ValueError(
            f"Output dimensions {unknown_output} not found in input dimensions {input_dims} "
            f"or index dimensions {index_dims}, in pattern: {pattern!r}"
        )
    if indexed_dim in output_dims:
        raise ValueError(
            f"Indexed dimension [{indexed_dim}] should not appear in the output dimensions "
            f"(it is consumed by gathering), in pattern: {pattern!r}"
        )

    # Validate array ranks match pattern
    if array.ndim != len(input_dims):
        raise ValueError(
            f"Array has {array.ndim} dimensions but pattern specifies {len(input_dims)} "
            f"({' '.join(input_dims)}), in pattern: {pattern!r}"
        )
    if indices.ndim != len(index_dims):
        raise ValueError(
            f"Indices array has {indices.ndim} dimensions but pattern specifies {len(index_dims)} "
            f"({' '.join(index_dims)}), in pattern: {pattern!r}"
        )

    # Validate that shared dimension names have matching sizes
    dim_sizes = {d: array.shape[i] for i, d in enumerate(input_dims)}
    for i, d in enumerate(index_dims):
        if d in dim_sizes and indices.shape[i] != dim_sizes[d]:
            raise ValueError(
                f"Dimension '{d}' has size {dim_sizes[d]} in array but size {indices.shape[i]} "
                f"in indices, in pattern: {pattern!r}"
            )

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