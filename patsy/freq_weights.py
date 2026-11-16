# This file is part of Patsy
# See file LICENSE.txt for license information.

# Support for fitting design matrices from data that has been compressed by
# storing a "frequency" (replication count) for each distinct row, rather
# than physically repeating the rows.
#
# The public entry points are the ``freq_weights=`` argument of
# :func:`patsy.dmatrix` / :func:`patsy.dmatrices`, and the
# :func:`patsy.freq_dmatrix` / :func:`patsy.freq_dmatrices` convenience
# wrappers. The rest of this module is internal machinery.

# This module provides no public names of its own; the public API lives in
# patsy.highlevel.
__all__ = []

import numpy as np

from patsy import PatsyError
from patsy.util import have_pandas, safe_issubdtype

if have_pandas:
    import pandas


# Data chunks that carry frequency weights alongside the actual data expose
# them under this key. It is deliberately improbable as a user column name,
# and it is only ever added internally (see _FreqWeightData below).
_FREQ_WEIGHTS_KEY = "__patsy_freq_weights__"

# Attribute used to hand the current chunk's weights to a stateful transform
# while it is memorizing. See EvalFactor.memorize_chunk.
_TRANSFORM_WEIGHTS_ATTR = "_patsy_freq_weights"


def _validate_freq_weights(weights, n_rows=None, origin=None):
    """Validate frequency weights and return them as a 1-d integer array.

    Frequency weights are non-negative integers; a weight of zero means that
    the corresponding row does not occur in the data set at all.

    Anything that cannot possibly be interpreted as a vector of replicate
    counts raises :class:`PatsyError` with an explicit message.
    """
    if have_pandas and isinstance(weights, pandas.Series):
        # Positional alignment; index agreement is checked by
        # _resolve_freq_weights when both sides are pandas objects.
        weights = weights.to_numpy()
    elif have_pandas and isinstance(weights, (pandas.DataFrame, pandas.Categorical)):
        raise PatsyError(
            "frequency weights must be one-dimensional, not a %s"
            % (type(weights).__name__,),
            origin,
        )
    try:
        arr = np.asarray(weights)
    except Exception:
        raise PatsyError("frequency weights must be array-like", origin)
    if arr.dtype == np.dtype(object):
        try:
            arr = np.asarray(arr, dtype=float)
        except (TypeError, ValueError):
            raise PatsyError(
                "frequency weights must be numeric, not %r" % (weights,), origin
            )
    if safe_issubdtype(arr.dtype, np.bool_):
        raise PatsyError(
            "frequency weights must be integers; boolean values are not "
            "valid frequency weights",
            origin,
        )
    if not (
        safe_issubdtype(arr.dtype, np.integer)
        or safe_issubdtype(arr.dtype, np.floating)
    ):
        raise PatsyError(
            "frequency weights must be real-valued numbers (got dtype %r)"
            % (arr.dtype,),
            origin,
        )
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise PatsyError(
            "frequency weights must be one-dimensional (got an array with "
            "%s dimensions)" % (arr.ndim,),
            origin,
        )
    if safe_issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise PatsyError(
                "frequency weights must be finite (no NaN or infinity allowed)",
                origin,
            )
        if not np.all(arr == np.floor(arr)):
            bad = arr[arr != np.floor(arr)]
            raise PatsyError(
                "frequency weights must be integers (got non-integer value %r)"
                % (bad.ravel()[0],),
                origin,
            )
    if np.any(arr < 0):
        bad = arr[arr < 0]
        raise PatsyError(
            "frequency weights must be non-negative (got %r)" % (bad.ravel()[0],),
            origin,
        )
    # Refuse values that would overflow when converted to the integer index
    # type used internally (np.asarray(1e30).astype(np.intp) silently wraps).
    if np.any(arr > np.iinfo(np.intp).max):
        bad = arr[arr > np.iinfo(np.intp).max]
        raise PatsyError(
            "frequency weights are too large (got %r; the maximum is %s)"
            % (bad.ravel()[0], np.iinfo(np.intp).max),
            origin,
        )
    # The total number of replicated rows is used as an integer, so it must
    # not overflow either. (Checked in Python integers, and only when a quick
    # bound says it might be possible.)
    max_weight = int(np.max(arr)) if arr.size else 0
    if max_weight and np.iinfo(np.intp).max // max_weight < arr.size:
        if sum(int(weight) for weight in arr) > np.iinfo(np.intp).max:
            raise PatsyError(
                "frequency weights are too large: their total exceeds %s"
                % (np.iinfo(np.intp).max,),
                origin,
            )
    if n_rows is not None and arr.shape[0] != n_rows:
        raise PatsyError(
            "frequency weights must have one entry per data row: expected "
            "%s entries but got %s" % (n_rows, arr.shape[0]),
            origin,
        )
    return arr.astype(np.intp)


def _resolve_freq_weights(freq_weights, data):
    """Turn the public ``freq_weights=`` argument into an array.

    A string is interpreted as the name of a column in ``data``. If both the
    weights and the data are pandas objects, their indexes must agree.
    """
    if isinstance(freq_weights, str):
        try:
            resolved = data[freq_weights]
        except (KeyError, TypeError):
            raise PatsyError(
                "frequency weights column %r was not found in the data"
                % (freq_weights,)
            )
        if (
            have_pandas
            and isinstance(data, pandas.DataFrame)
            and isinstance(resolved, (pandas.Series, pandas.DataFrame))
            and not resolved.index.equals(data.index)
        ):
            raise PatsyError(
                "index mismatch between the data and the frequency weights "
                "column %r" % (freq_weights,)
            )
        return resolved
    if (
        have_pandas
        and isinstance(data, (pandas.Series, pandas.DataFrame))
        and isinstance(freq_weights, (pandas.Series, pandas.DataFrame))
        and not freq_weights.index.equals(data.index)
    ):
        raise PatsyError("index mismatch between the data and the frequency weights")
    return freq_weights


def _num_rows(data):
    """Best-effort row count for a data mapping, or None if unknowable."""
    if have_pandas and isinstance(data, pandas.DataFrame):
        return data.shape[0]
    try:
        keys = list(data.keys())
    except (AttributeError, TypeError):
        return None
    for key in keys:
        value = data[key]
        if have_pandas and isinstance(value, (pandas.Series, pandas.DataFrame)):
            return value.shape[0]
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return value.shape[0]
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def _subset_parallel(value, mask):
    """Subset a single parallel data vector, leaving scalars alone."""
    if have_pandas:
        if isinstance(value, (pandas.Series, pandas.DataFrame)):
            return value.iloc[np.flatnonzero(mask)]
        if isinstance(value, pandas.Categorical):
            return value.take(np.flatnonzero(mask))
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == mask.shape[0]:
            return value[mask, ...]
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == mask.shape[0]:
            return type(value)(item for (item, keep) in zip(value, mask) if keep)
        return value
    return value


def _subset_rows(data, mask):
    """Subset the rows of a data mapping with a boolean mask."""
    if have_pandas and isinstance(data, pandas.DataFrame):
        return data.iloc[np.flatnonzero(mask)]
    if isinstance(data, dict):
        return dict(
            (key, _subset_parallel(value, mask)) for (key, value) in data.items()
        )
    # Generic mapping: try to rebuild the same type, else fall back to dict.
    try:
        keys = list(data.keys())
    except (AttributeError, TypeError):
        raise PatsyError(
            "cannot subset the data to remove zero-frequency rows; pass a "
            "dict or a pandas.DataFrame"
        )
    subsetted = {key: _subset_parallel(data[key], mask) for key in keys}
    try:
        return type(data)(subsetted)
    except Exception:
        return subsetted


class _FreqWeightData(object):
    """Read-only view of a data mapping that also exposes frequency weights.

    This is handed to the memorization machinery; a stateful transform that
    understands frequency weights receives the weights array under the
    reserved key, while everything else sees exactly the original data.
    """

    def __init__(self, data, freq_weights):
        self._data = data
        self.freq_weights = freq_weights

    def __getitem__(self, key):
        if key == _FREQ_WEIGHTS_KEY:
            return self.freq_weights
        return self._data[key]

    def __contains__(self, key):
        if key == _FREQ_WEIGHTS_KEY:
            return True
        return key in self._data

    def keys(self):
        return list(self._data.keys()) + [_FREQ_WEIGHTS_KEY]

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._data)


def _wrap_data_iter_maker(data_iter_maker, freq_weights):
    """Wrap a chunk iterator so each chunk carries its slice of the weights."""
    if freq_weights is None:
        return data_iter_maker
    freq_weights = np.asarray(freq_weights)

    def wrapped_iter_maker():
        offset = 0
        total = len(freq_weights)
        for chunk in data_iter_maker():
            try:
                clashes = _FREQ_WEIGHTS_KEY in chunk
            except TypeError:  # pragma: no cover
                clashes = False
            if clashes:
                raise PatsyError(
                    "the column name %r is reserved for internal use and "
                    "cannot be combined with frequency weights" % (_FREQ_WEIGHTS_KEY,)
                )
            n_rows = _num_rows(chunk)
            if n_rows is None:
                if offset == 0 and total != 0:
                    chunk_weights = freq_weights
                else:  # pragma: no cover
                    raise PatsyError(
                        "cannot determine how many rows are in this data "
                        "chunk, so frequency weights cannot be aligned with it"
                    )
            else:
                chunk_weights = freq_weights[offset : offset + n_rows]
                if chunk_weights.shape[0] != n_rows:  # pragma: no cover
                    raise PatsyError(
                        "frequency weights do not cover all of the data: "
                        "ran out after %s of %s rows" % (offset, total)
                    )
                offset += n_rows
            yield _FreqWeightData(chunk, chunk_weights)
        if offset < total:  # pragma: no cover
            raise PatsyError(
                "frequency weights do not cover all of the data: %s of %s "
                "rows were never seen" % (offset, total)
            )

    return wrapped_iter_maker


def _patsy_chunk_weights(transform, n_rows):
    """Frequency weights that apply to the current chunk of transform data.

    Returns ``None`` when the transform is being used without frequency
    weights, so that the ordinary code path stays bit-for-bit unchanged.
    """
    raw = getattr(transform, _TRANSFORM_WEIGHTS_ATTR, None)
    if raw is None:
        return None
    weights = np.asarray(raw)
    if weights.ndim != 1:
        return None
    if weights.shape[0] == n_rows:
        return weights.astype(float)
    # If the transform is being applied to something that is not row-parallel
    # with the data (a literal array, say), the weights do not apply. This is
    # the same situation in which patsy will complain about mismatched row
    # counts later on.
    return None


def _weighted_percentile(values, weights, percents):
    """Weighted equivalent of ``np.percentile`` on replicated data.

    ``values`` and ``weights`` are 1-d arrays of equal length; ``weights``
    holds non-negative integer replicate counts. The result is exactly what
    ``np.percentile(np.repeat(values, weights), percents)`` computes with
    numpy's default ('linear') interpolation, but without materializing the
    replicated array.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must be 1-d arrays of equal length")
    keep = weights > 0
    values = values[keep]
    weights = weights[keep]
    if values.size == 0:
        raise PatsyError("frequency weights must contain at least one positive entry")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order].astype(np.int64)
    probs = np.asarray(percents, dtype=float) / 100.0
    flat = probs.ravel()
    # numpy's quantile code reports NaN for every requested quantile as soon
    # as the (sorted) data contains a NaN, because then the last element is
    # NaN. Match that behavior exactly.
    if np.issubdtype(values.dtype, np.inexact) and np.any(np.isnan(values)):
        return np.full(probs.shape, np.nan, dtype=float)
    totals = np.cumsum(weights)
    n = int(totals[-1])
    # np.percentile's 'linear' method uses virtual index (n - 1) * q.
    virtual = flat * (n - 1)
    vfloor = np.floor(virtual)
    prev_indexes = vfloor.astype(np.intp)
    next_indexes = prev_indexes + 1
    above = virtual >= (n - 1)
    if np.any(above):
        prev_indexes[above] = n - 1
        next_indexes[above] = n - 1
    below = virtual < 0
    if np.any(below):
        prev_indexes[below] = 0
        next_indexes[below] = 0
    t = virtual - vfloor
    # Map replicated-array positions back to the compressed values.
    prev_values = values[np.searchsorted(totals, prev_indexes, side="right")]
    next_values = values[np.searchsorted(totals, next_indexes, side="right")]
    # numpy's own percentile implementation can emit these warnings for
    # infinite inputs; do not turn them into patsy warnings of our own.
    with np.errstate(invalid="ignore", over="ignore"):
        diff = next_values - prev_values
        result = prev_values + diff * t
        # Mirror numpy's _lerp exactly, to keep bit-for-bit compatibility.
        upper_half = t >= 0.5
        if np.any(upper_half):
            result[upper_half] = next_values[upper_half] - diff[upper_half] * (
                1.0 - t[upper_half]
            )
    return result.reshape(probs.shape, order="C")


def _weighted_R_compat_quantile(x, weights, probs):
    """Weighted version of :func:`patsy.splines._R_compat_quantile`."""
    probs = np.asarray(probs)
    quantiles = _weighted_percentile(x, weights, 100 * probs.ravel(order="C"))
    return np.asarray(quantiles).reshape(probs.shape, order="C")


def test__validate_freq_weights():
    import pytest

    assert np.array_equal(_validate_freq_weights([1, 0, 2]), [1, 0, 2])
    assert np.array_equal(_validate_freq_weights([1.0, 2.0]), [1, 2])
    assert np.array_equal(_validate_freq_weights(3, n_rows=1), [3])
    assert _validate_freq_weights([1, 2]).dtype == np.dtype(np.intp)

    pytest.raises(PatsyError, _validate_freq_weights, [-1, 2])
    pytest.raises(PatsyError, _validate_freq_weights, [1.5, 2])
    pytest.raises(PatsyError, _validate_freq_weights, [np.nan, 2])
    pytest.raises(PatsyError, _validate_freq_weights, [np.inf, 2])
    pytest.raises(PatsyError, _validate_freq_weights, [[1, 2], [3, 4]])
    pytest.raises(PatsyError, _validate_freq_weights, [True, False])
    pytest.raises(PatsyError, _validate_freq_weights, ["a", "b"])
    pytest.raises(PatsyError, _validate_freq_weights, [1, 2], n_rows=3)
    # Values that would overflow the internal integer representation.
    pytest.raises(PatsyError, _validate_freq_weights, [1e30, 1])
    pytest.raises(PatsyError, _validate_freq_weights, [2**62, 2**62])


def test__weighted_percentile_matches_replication():
    import pytest

    rng = np.random.RandomState(0)
    for _ in range(200):
        n = rng.randint(1, 12)
        x = rng.normal(size=n)
        w = rng.randint(0, 5, size=n)
        if w.sum() == 0:
            w[0] = 1
        probs = rng.uniform(size=5)
        replicated = np.repeat(x, w)
        assert np.array_equal(
            _weighted_percentile(x, w, 100 * probs),
            np.percentile(replicated, 100 * probs),
        )
    pytest.raises(PatsyError, _weighted_percentile, [1, 2], [0, 0], 50)


def test__subset_rows():
    data = {"a": [1, 2, 3], "b": np.array([4, 5, 6]), "c": 10}
    out = _subset_rows(data, np.array([True, False, True]))
    assert out["a"] == [1, 3]
    assert np.array_equal(out["b"], [4, 6])
    assert out["c"] == 10
    if have_pandas:
        df = pandas.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
        out_df = _subset_rows(df, np.array([False, True, True]))
        assert list(out_df.index) == [20, 30]


def test__freq_weight_data():
    data = _FreqWeightData({"a": [1, 2]}, np.array([3, 4]))
    assert list(data["a"]) == [1, 2]
    assert np.array_equal(data[_FREQ_WEIGHTS_KEY], [3, 4])
    assert "a" in data
    assert _FREQ_WEIGHTS_KEY in data
    assert "__other__" not in data
    assert isinstance(repr(data), str)


def test__wrap_data_iter_maker():
    import pytest

    def iter_maker():
        yield {"a": [1, 2]}
        yield {"a": [3, 4, 5]}

    wrapped = _wrap_data_iter_maker(iter_maker, np.array([1, 0, 2, 3, 4]))
    chunks = list(wrapped())
    assert [list(chunk["a"]) for chunk in chunks] == [[1, 2], [3, 4, 5]]
    assert [list(chunk[_FREQ_WEIGHTS_KEY]) for chunk in chunks] == [[1, 0], [2, 3, 4]]
    assert all(chunk.keys() == ["a", _FREQ_WEIGHTS_KEY] for chunk in chunks)

    # Weights that do not cover the whole data set are an error.
    short = _wrap_data_iter_maker(iter_maker, np.array([1, 2, 3]))
    pytest.raises(PatsyError, list, short())

    # A column with the reserved name is rejected rather than shadowed.
    def clashing_iter_maker():
        yield {"a": [1], _FREQ_WEIGHTS_KEY: [2]}

    bad = _wrap_data_iter_maker(clashing_iter_maker, np.array([1]))
    pytest.raises(PatsyError, list, bad())

    # Without weights the iterator is passed through untouched.
    assert _wrap_data_iter_maker(iter_maker, None) is iter_maker


def test__patsy_chunk_weights():
    class Transform(object):
        pass

    t = Transform()
    assert _patsy_chunk_weights(t, 3) is None
    setattr(t, _TRANSFORM_WEIGHTS_ATTR, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(_patsy_chunk_weights(t, 3), [1.0, 2.0, 3.0])
    # A transform applied to something that is not row-parallel with the
    # data does not get weighted.
    assert _patsy_chunk_weights(t, 4) is None
