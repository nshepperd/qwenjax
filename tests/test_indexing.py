import pytest
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from qwen_jax.utils.indexing import gather


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

def test_missing_arrow():
    with pytest.raises(ValueError, match="must contain '->'"):
        gather("b s h d, b [s]", jnp.zeros((2, 3, 4, 5)), jnp.zeros((2,), dtype=jnp.int32))


def test_wrong_number_of_inputs_one():
    with pytest.raises(ValueError, match="exactly two comma-separated"):
        gather("b s h d [s] -> b h d", jnp.zeros((2, 3, 4, 5)), jnp.zeros((2,), dtype=jnp.int32))


def test_wrong_number_of_inputs_three():
    with pytest.raises(ValueError, match="exactly two comma-separated"):
        gather("b s, h d, b [s] -> b h d", jnp.zeros((2, 3)), jnp.zeros((2,), dtype=jnp.int32))


def test_empty_array_dims():
    with pytest.raises(ValueError, match="cannot be empty"):
        gather(" , b [s] -> b", jnp.zeros((2,)), jnp.zeros((2,), dtype=jnp.int32))


def test_empty_index_dims():
    with pytest.raises(ValueError, match="cannot be empty"):
        gather("b s,  -> b", jnp.zeros((2, 3)), jnp.zeros((2,), dtype=jnp.int32))


def test_no_bracketed_dim():
    with pytest.raises(ValueError, match="must contain one \\[bracketed\\]"):
        gather("b s, b -> b", jnp.zeros((2, 3)), jnp.zeros((2,), dtype=jnp.int32))


def test_multiple_bracketed_dims():
    with pytest.raises(ValueError, match="exactly one \\[bracketed\\]"):
        gather("b s h, [b] [s] -> h", jnp.zeros((2, 3, 4)), jnp.zeros((2, 3), dtype=jnp.int32))


def test_indexed_dim_not_in_array():
    with pytest.raises(ValueError, match="must appear in the array dimensions"):
        gather("b s, b [x] -> b", jnp.zeros((2, 3)), jnp.zeros((2,), dtype=jnp.int32))


def test_unknown_output_dim():
    with pytest.raises(ValueError, match="not found in input dimensions"):
        gather("b s h d, b [s] -> b z h d", jnp.zeros((2, 3, 4, 5)), jnp.zeros((2,), dtype=jnp.int32))


def test_indexed_dim_in_output():
    with pytest.raises(ValueError, match="should not appear in the output"):
        gather("b s h d, b [s] -> b s h d", jnp.zeros((2, 3, 4, 5)), jnp.zeros((2,), dtype=jnp.int32))


def test_array_rank_mismatch():
    with pytest.raises(ValueError, match="Array has 3 dimensions but pattern specifies 4"):
        gather("b s h d, b [s] -> b h d", jnp.zeros((2, 3, 4)), jnp.zeros((2,), dtype=jnp.int32))


def test_indices_rank_mismatch():
    with pytest.raises(ValueError, match="Indices array has 2 dimensions but pattern specifies 1"):
        gather("b s h d, b [s] -> b h d", jnp.zeros((2, 3, 4, 5)), jnp.zeros((2, 3), dtype=jnp.int32))


def test_shared_dim_size_mismatch():
    with pytest.raises(ValueError, match="Dimension 'b' has size 2 .* but size 7"):
        gather("b s h d, b [s] -> b h d", jnp.zeros((2, 3, 4, 5)), jnp.zeros((7,), dtype=jnp.int32))


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_basic_gather():
    """b s h d, b [s] -> b h d — select one sequence position per batch."""
    array = jnp.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    indices = jnp.array([1, 2])  # batch=2, pick seq position 1 and 2
    result = gather("b s h d, b [s] -> b h d", array, indices)
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result[0], array[0, 1])
    np.testing.assert_array_equal(result[1], array[1, 2])


def test_gather_1d():
    """s, [s] -> (scalar output)."""
    array = jnp.array([10, 20, 30, 40])
    indices = jnp.array(2)
    result = gather("s, [s] -> ", array, indices)
    assert result.shape == ()
    assert int(result) == 30


def test_gather_with_index_broadcast():
    """s d, n [s] -> n d — gather multiple positions from a shared array."""
    array = jnp.arange(4 * 3).reshape(4, 3)
    indices = jnp.array([0, 2, 3])
    result = gather("s d, n [s] -> n d", array, indices)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result[0], array[0])
    np.testing.assert_array_equal(result[1], array[2])
    np.testing.assert_array_equal(result[2], array[3])


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

@given(
    b=st.integers(1, 4),
    s=st.integers(2, 6),
    h=st.integers(1, 4),
    d=st.integers(1, 4),
)
@settings(max_examples=30)
def test_gather_shape_bshd(b, s, h, d):
    """Output shape matches pattern for 'b s h d, b [s] -> b h d'."""
    array = jnp.zeros((b, s, h, d))
    indices = jnp.zeros((b,), dtype=jnp.int32)
    result = gather("b s h d, b [s] -> b h d", array, indices)
    assert result.shape == (b, h, d)


@given(
    b=st.integers(1, 4),
    s=st.integers(2, 6),
    h=st.integers(1, 4),
    d=st.integers(1, 4),
)
@settings(max_examples=30)
def test_gather_values_bshd(b, s, h, d):
    """Gathered values match direct numpy indexing."""
    rng = np.random.RandomState(b * 1000 + s * 100 + h * 10 + d)
    arr_np = rng.randn(b, s, h, d).astype(np.float32)
    idx_np = rng.randint(0, s, size=(b,))

    array = jnp.array(arr_np)
    indices = jnp.array(idx_np, dtype=jnp.int32)
    result = gather("b s h d, b [s] -> b h d", array, indices)

    expected = arr_np[np.arange(b), idx_np]  # shape (b, h, d)
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6)


@given(
    s=st.integers(2, 8),
    d=st.integers(1, 5),
    n=st.integers(1, 6),
)
@settings(max_examples=30)
def test_gather_index_broadcast_shape(s, d, n):
    """Output shape for 's d, n [s] -> n d' (index broadcast)."""
    array = jnp.zeros((s, d))
    indices = jnp.zeros((n,), dtype=jnp.int32)
    result = gather("s d, n [s] -> n d", array, indices)
    assert result.shape == (n, d)


@given(
    s=st.integers(2, 8),
    d=st.integers(1, 5),
    n=st.integers(1, 6),
)
@settings(max_examples=30)
def test_gather_index_broadcast_values(s, d, n):
    """Gathered values for index broadcast pattern match numpy."""
    rng = np.random.RandomState(s * 100 + d * 10 + n)
    arr_np = rng.randn(s, d).astype(np.float32)
    idx_np = rng.randint(0, s, size=(n,))

    array = jnp.array(arr_np)
    indices = jnp.array(idx_np, dtype=jnp.int32)
    result = gather("s d, n [s] -> n d", array, indices)

    expected = arr_np[idx_np]  # shape (n, d)
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6)


@given(
    b=st.integers(1, 3),
    s=st.integers(2, 5),
    n=st.integers(1, 4),
    d=st.integers(1, 4),
)
@settings(max_examples=30)
def test_gather_batch_and_index_broadcast(b, s, n, d):
    """'b s d, b n [s] -> b n d' — batch + index broadcast."""
    rng = np.random.RandomState(b * 1000 + s * 100 + n * 10 + d)
    arr_np = rng.randn(b, s, d).astype(np.float32)
    idx_np = rng.randint(0, s, size=(b, n))

    array = jnp.array(arr_np)
    indices = jnp.array(idx_np, dtype=jnp.int32)
    result = gather("b s d, b n [s] -> b n d", array, indices)
    assert result.shape == (b, n, d)

    expected = np.stack([arr_np[i][idx_np[i]] for i in range(b)])
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6)


@given(
    b=st.integers(1, 3),
    s=st.integers(2, 5),
    h=st.integers(1, 3),
    d=st.integers(1, 3),
)
@settings(max_examples=20)
def test_gather_reordered_output(b, s, h, d):
    """Output dim reordering: 'b s h d, b [s] -> h b d' works correctly."""
    rng = np.random.RandomState(b * 1000 + s * 100 + h * 10 + d)
    arr_np = rng.randn(b, s, h, d).astype(np.float32)
    idx_np = rng.randint(0, s, size=(b,))

    array = jnp.array(arr_np)
    indices = jnp.array(idx_np, dtype=jnp.int32)
    result = gather("b s h d, b [s] -> h b d", array, indices)
    assert result.shape == (h, b, d)

    # Build expected: first gather b h d, then transpose to h b d
    gathered = arr_np[np.arange(b), idx_np]  # (b, h, d)
    expected = np.transpose(gathered, (1, 0, 2))  # (h, b, d)
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6)
