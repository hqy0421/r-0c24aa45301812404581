# This file is part of Patsy
# Copyright (C) 2011-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# These are made available in the patsy.* namespace:
__all__ = [
    "dmatrix",
    "dmatrices",
    "freq_dmatrix",
    "freq_dmatrices",
    "incr_dbuilder",
    "incr_dbuilders",
]

# problems:
#   statsmodels reluctant to pass around separate eval environment, suggesting
#     that design_and_matrices-equivalent should return a formula_like
#   is ModelDesc really the high-level thing?
#   ModelDesign doesn't work -- need to work with the builder set
#   want to be able to return either a matrix or a pandas dataframe

import numpy as np

from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc
from patsy.build import (
    design_matrix_builders,
    build_design_matrices,
    _build_design_matrices_impl,
)
from patsy.freq_weights import (
    _num_rows,
    _resolve_freq_weights,
    _subset_rows,
    _validate_freq_weights,
    _wrap_data_iter_maker,
)
from patsy.util import have_pandas, asarray_or_pandas, atleast_2d_column_default

if have_pandas:
    import pandas


# Tries to build a (lhs, rhs) design given a formula_like and an incremental
# data source. If formula_like is not capable of doing this, then returns
# None.
def _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action):
    if isinstance(formula_like, DesignInfo):
        return (
            design_matrix_builders([[]], data_iter_maker, eval_env, NA_action)[0],
            formula_like,
        )
    if (
        isinstance(formula_like, tuple)
        and len(formula_like) == 2
        and isinstance(formula_like[0], DesignInfo)
        and isinstance(formula_like[1], DesignInfo)
    ):
        return formula_like
    if hasattr(formula_like, "__patsy_get_model_desc__"):
        formula_like = formula_like.__patsy_get_model_desc__(eval_env)
        if not isinstance(formula_like, ModelDesc):
            raise PatsyError(
                "bad value from %r.__patsy_get_model_desc__" % (formula_like,)
            )
        # fallthrough
    if isinstance(formula_like, str):
        formula_like = ModelDesc.from_formula(formula_like)
        # fallthrough
    if isinstance(formula_like, ModelDesc):
        assert isinstance(eval_env, EvalEnvironment)
        return design_matrix_builders(
            [formula_like.lhs_termlist, formula_like.rhs_termlist],
            data_iter_maker,
            eval_env,
            NA_action,
        )
    else:
        return None


def incr_dbuilder(formula_like, data_iter_maker, eval_env=0, NA_action="drop"):
    """Construct a design matrix builder incrementally from a large data set.

    :arg formula_like: Similar to :func:`dmatrix`, except that explicit
      matrices are not allowed. Must be a formula string, a
      :class:`ModelDesc`, a :class:`DesignInfo`, or an object with a
      ``__patsy_get_model_desc__`` method.
    :arg data_iter_maker: A zero-argument callable which returns an iterator
      over dict-like data objects. This must be a callable rather than a
      simple iterator because sufficiently complex formulas may require
      multiple passes over the data (e.g. if there are nested stateful
      transforms).
    :arg eval_env: Either a :class:`EvalEnvironment` which will be used to
      look up any variables referenced in `formula_like` that cannot be
      found in `data`, or else a depth represented as an
      integer which will be passed to :meth:`EvalEnvironment.capture`.
      ``eval_env=0`` means to use the context of the function calling
      :func:`incr_dbuilder` for lookups. If calling this function from a
      library, you probably want ``eval_env=1``, which means that variables
      should be resolved in *your* caller's namespace.
    :arg NA_action: An :class:`NAAction` object or string, used to determine
      what values count as 'missing' for purposes of determining the levels of
      categorical factors.
    :returns: A :class:`DesignInfo`

    Tip: for `data_iter_maker`, write a generator like::

      def iter_maker():
          for data_chunk in my_data_store:
              yield data_chunk

    and pass `iter_maker` (*not* `iter_maker()`).

    .. versionadded:: 0.2.0
       The ``NA_action`` argument.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    design_infos = _try_incr_builders(
        formula_like, data_iter_maker, eval_env, NA_action
    )
    if design_infos is None:
        raise PatsyError("bad formula-like object")
    if len(design_infos[0].column_names) > 0:
        raise PatsyError(
            "encountered outcome variables for a model that does not expect them"
        )
    return design_infos[1]


def incr_dbuilders(formula_like, data_iter_maker, eval_env=0, NA_action="drop"):
    """Construct two design matrix builders incrementally from a large data
    set.

    :func:`incr_dbuilders` is to :func:`incr_dbuilder` as :func:`dmatrices` is
    to :func:`dmatrix`. See :func:`incr_dbuilder` for details.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    design_infos = _try_incr_builders(
        formula_like, data_iter_maker, eval_env, NA_action
    )
    if design_infos is None:
        raise PatsyError("bad formula-like object")
    if len(design_infos[0].column_names) == 0:
        raise PatsyError("model is missing required outcome variables")
    return design_infos


# This always returns a length-two tuple,
#   response, predictors
# where
#   response is a DesignMatrix (possibly with 0 columns)
#   predictors is a DesignMatrix
# The input 'formula_like' could be like:
#   (np.ndarray, np.ndarray)
#   (DesignMatrix, DesignMatrix)
#   (None, DesignMatrix)
#   np.ndarray  # for predictor-only models
#   DesignMatrix
#   (None, np.ndarray)
#   "y ~ x"
#   ModelDesc(...)
#   DesignInfo
#   (DesignInfo, DesignInfo)
#   any object with a special method __patsy_get_model_desc__
def _attach_freq_weights(matrices, freq_weights):
    """Expose the surviving frequency weights on each returned matrix."""
    for matrix in matrices:
        # Bypass pandas' __setattr__, which warns about list-like values and
        # would otherwise turn this into a data column.
        object.__setattr__(matrix, "freq_weights", freq_weights)


def _do_highlevel_design(
    formula_like, data, eval_env, NA_action, return_type, freq_weights=None
):
    if return_type == "dataframe" and not have_pandas:
        raise PatsyError("pandas.DataFrame was requested, but pandas is not installed")
    if return_type not in ("matrix", "dataframe"):
        raise PatsyError(
            "unrecognized output type %r, should be "
            "'matrix' or 'dataframe'" % (return_type,)
        )

    surviving_freq_weights = None
    if freq_weights is not None:
        # Resolve a column name / pandas object, validate, and remove the
        # rows that have zero frequency: they simply do not occur in the
        # replicated data set, so they must not influence memorized
        # statistics, categorical levels, or the design matrix itself.
        freq_weights = _resolve_freq_weights(freq_weights, data)
        freq_weights = _validate_freq_weights(
            freq_weights, n_rows=_num_rows(data), origin=formula_like
        )
        if not np.any(freq_weights > 0):
            raise PatsyError(
                "frequency weights must contain at least one positive entry",
                formula_like,
            )
        nonzero = freq_weights > 0
        if not np.all(nonzero):
            data = _subset_rows(data, nonzero)
            freq_weights = freq_weights[nonzero]

    def data_iter_maker():
        return iter([data])

    memorization_iter_maker = _wrap_data_iter_maker(data_iter_maker, freq_weights)

    design_infos = _try_incr_builders(
        formula_like, memorization_iter_maker, eval_env, NA_action
    )
    if design_infos is not None:
        matrices, surviving_freq_weights = _build_design_matrices_impl(
            design_infos,
            data,
            NA_action=NA_action,
            return_type=return_type,
            freq_weights=freq_weights,
        )
        if surviving_freq_weights is not None:
            _attach_freq_weights(matrices, surviving_freq_weights)
        return (matrices, surviving_freq_weights)
    else:
        if freq_weights is not None:
            raise PatsyError(
                "frequency weights are not supported when the design matrix "
                "is given explicitly rather than through a formula",
                formula_like,
            )
        # No builders, but maybe we can still get matrices
        if isinstance(formula_like, tuple):
            if len(formula_like) != 2:
                raise PatsyError(
                    "don't know what to do with a length %s "
                    "matrices tuple" % (len(formula_like),)
                )
            (lhs, rhs) = formula_like
        else:
            # subok=True is necessary here to allow DesignMatrixes to pass
            # through
            (lhs, rhs) = (None, asarray_or_pandas(formula_like, subok=True))

        # some sort of explicit matrix or matrices were given. Currently we
        # have them in one of these forms:
        #   -- an ndarray or subclass
        #   -- a DesignMatrix
        #   -- a pandas.Series
        #   -- a pandas.DataFrame
        # and we have to produce a standard output format.
        def _regularize_matrix(m, default_column_prefix):
            di = DesignInfo.from_array(m, default_column_prefix)
            if have_pandas and isinstance(m, (pandas.Series, pandas.DataFrame)):
                orig_index = m.index
            else:
                orig_index = None
            if return_type == "dataframe":
                m = atleast_2d_column_default(m, preserve_pandas=True)
                m = pandas.DataFrame(m)
                m.columns = di.column_names
                m.design_info = di
                return (m, orig_index)
            else:
                return (DesignMatrix(m, di), orig_index)

        rhs, rhs_orig_index = _regularize_matrix(rhs, "x")
        if lhs is None:
            lhs = np.zeros((rhs.shape[0], 0), dtype=float)
        lhs, lhs_orig_index = _regularize_matrix(lhs, "y")

        assert isinstance(getattr(lhs, "design_info", None), DesignInfo)
        assert isinstance(getattr(rhs, "design_info", None), DesignInfo)
        if lhs.shape[0] != rhs.shape[0]:
            raise PatsyError(
                "shape mismatch: outcome matrix has %s rows, "
                "predictor matrix has %s rows" % (lhs.shape[0], rhs.shape[0])
            )
        if rhs_orig_index is not None and lhs_orig_index is not None:
            if not rhs_orig_index.equals(lhs_orig_index):
                raise PatsyError(
                    "index mismatch: outcome and predictor have incompatible indexes"
                )
        if return_type == "dataframe":
            if rhs_orig_index is not None and lhs_orig_index is None:
                lhs.index = rhs.index
            if rhs_orig_index is None and lhs_orig_index is not None:
                rhs.index = lhs.index
        return ((lhs, rhs), None)


def dmatrix(
    formula_like,
    data={},
    eval_env=0,
    NA_action="drop",
    return_type="matrix",
    freq_weights=None,
):
    """Construct a single design matrix given a formula_like and data.

    :arg formula_like: An object that can be used to construct a design
      matrix. See below.
    :arg data: A dict-like object that can be used to look up variables
      referenced in `formula_like`.
    :arg eval_env: Either a :class:`EvalEnvironment` which will be used to
      look up any variables referenced in `formula_like` that cannot be
      found in `data`, or else a depth represented as an
      integer which will be passed to :meth:`EvalEnvironment.capture`.
      ``eval_env=0`` means to use the context of the function calling
      :func:`dmatrix` for lookups. If calling this function from a library,
      you probably want ``eval_env=1``, which means that variables should be
      resolved in *your* caller's namespace.
    :arg NA_action: What to do with rows that contain missing values. You can
      ``"drop"`` them, ``"raise"`` an error, or for customization, pass an
      :class:`NAAction` object. See :class:`NAAction` for details on what
      values count as 'missing' (and how to alter this).
    :arg return_type: Either ``"matrix"`` or ``"dataframe"``. See below.
    :arg freq_weights: Optional frequency weights, used when the data has
      been compressed by storing each distinct row once together with a
      non-negative integer count of how many times it occurs. See
      :ref:`freq-weights` for details. When given, the returned matrix has a
      ``.freq_weights`` attribute holding the weights of the rows that
      survived missing-value handling.

    The `formula_like` can take a variety of forms. You can use any of the
    following:

    * (The most common option) A formula string like ``"x1 + x2"`` (for
      :func:`dmatrix`) or ``"y ~ x1 + x2"`` (for :func:`dmatrices`). For
      details see :ref:`formulas`.
    * A :class:`ModelDesc`, which is a Python object representation of a
      formula. See :ref:`formulas` and :ref:`expert-model-specification` for
      details.
    * A :class:`DesignInfo`.
    * An object that has a method called :meth:`__patsy_get_model_desc__`.
      For details see :ref:`expert-model-specification`.
    * A numpy array_like (for :func:`dmatrix`) or a tuple
      (array_like, array_like) (for :func:`dmatrices`). These will have
      metadata added, representation normalized, and then be returned
      directly. In this case `data` and `eval_env` are
      ignored. There is special handling for two cases:

      * :class:`DesignMatrix` objects will have their :class:`DesignInfo`
        preserved. This allows you to set up custom column names and term
        information even if you aren't using the rest of the patsy
        machinery.
      * :class:`pandas.DataFrame` or :class:`pandas.Series` objects will have
        their (row) indexes checked. If two are passed in, their indexes must
        be aligned. If ``return_type="dataframe"``, then their indexes will be
        preserved on the output.

    Regardless of the input, the return type is always either:

    * A :class:`DesignMatrix`, if ``return_type="matrix"`` (the default)
    * A :class:`pandas.DataFrame`, if ``return_type="dataframe"``.

    The actual contents of the design matrix is identical in both cases, and
    in both cases a :class:`DesignInfo` object will be available in a
    ``.design_info`` attribute on the return value. However, for
    ``return_type="dataframe"``, any pandas indexes on the input (either in
    `data` or directly passed through `formula_like`) will be preserved, which
    may be useful for e.g. time-series models.

    .. versionadded:: 0.2.0
       The ``NA_action`` argument.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    (matrices, _) = _do_highlevel_design(
        formula_like, data, eval_env, NA_action, return_type, freq_weights
    )
    (lhs, rhs) = matrices
    if lhs.shape[1] != 0:
        raise PatsyError(
            "encountered outcome variables for a model that does not expect them"
        )
    return rhs


def dmatrices(
    formula_like,
    data={},
    eval_env=0,
    NA_action="drop",
    return_type="matrix",
    freq_weights=None,
):
    """Construct two design matrices given a formula_like and data.

    This function is identical to :func:`dmatrix`, except that it requires
    (and returns) two matrices instead of one. By convention, the first matrix
    is the "outcome" or "y" data, and the second is the "predictor" or "x"
    data.

    See :func:`dmatrix` for details.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    (matrices, _) = _do_highlevel_design(
        formula_like, data, eval_env, NA_action, return_type, freq_weights
    )
    (lhs, rhs) = matrices
    if lhs.shape[1] == 0:
        raise PatsyError("model is missing required outcome variables")
    return (lhs, rhs)


def freq_dmatrix(
    formula_like,
    data={},
    freq_weights=None,
    eval_env=0,
    NA_action="drop",
    return_type="matrix",
):
    """Construct a design matrix from frequency-compressed data.

    This is a convenience wrapper around :func:`dmatrix` for the common case
    where the data has already been compressed by storing each distinct row
    once, along with a non-negative integer count of how many times it occurs.

    ``freq_weights`` is either a 1-d array-like of such counts (one per row of
    ``data``) or the name of a column in ``data`` holding them. Rows with
    weight zero are dropped, and every stateful transform is memorized exactly
    as if each row had been physically repeated ``weight`` times.

    Returns ``(design_matrix, row_freq_weights)``, where ``design_matrix`` is
    an ordinary design matrix (so ``design_matrix.design_info`` is a real
    :class:`DesignInfo`, usable directly with :func:`build_design_matrices`)
    and ``row_freq_weights`` holds the weight of each returned row after
    missing-value handling. Prediction on new data needs neither the weights
    nor any re-learning: pass ``design_matrix.design_info`` to
    :func:`build_design_matrices` as usual.

    .. versionadded:: 1.1.0
    """
    if freq_weights is None:
        raise PatsyError(
            "freq_dmatrix requires the freq_weights argument; use dmatrix "
            "for ordinary design matrices"
        )
    design_matrix = dmatrix(
        formula_like,
        data,
        eval_env=eval_env,
        NA_action=NA_action,
        return_type=return_type,
        freq_weights=freq_weights,
    )
    return (design_matrix, design_matrix.freq_weights)


def freq_dmatrices(
    formula_like,
    data={},
    freq_weights=None,
    eval_env=0,
    NA_action="drop",
    return_type="matrix",
):
    """Construct two design matrices from frequency-compressed data.

    This is to :func:`dmatrices` as :func:`freq_dmatrix` is to
    :func:`dmatrix`. See :func:`freq_dmatrix` for details.

    Returns ``((lhs, rhs), row_freq_weights)``.

    .. versionadded:: 1.1.0
    """
    if freq_weights is None:
        raise PatsyError(
            "freq_dmatrices requires the freq_weights argument; use dmatrices "
            "for ordinary design matrices"
        )
    matrices = dmatrices(
        formula_like,
        data,
        eval_env=eval_env,
        NA_action=NA_action,
        return_type=return_type,
        freq_weights=freq_weights,
    )
    return (matrices, matrices[0].freq_weights)


def _expand_freq_weights(data, freq_weights):
    """Reference helper: physically replicate compressed rows."""
    freq_weights = np.asarray(freq_weights)
    keep = freq_weights > 0
    starts = np.concatenate(([0], np.cumsum(freq_weights)[:-1]))
    expanded = {}
    for key, value in data.items():
        value = np.asarray(value)
        if value.ndim >= 1 and value.shape[0] == len(freq_weights):
            expanded[key] = np.repeat(value, freq_weights, axis=0)
        else:
            expanded[key] = value
    return expanded, starts[keep], keep


def _assert_freq_weights_equivalent(formula, data, freq_weights, **kwargs):
    expanded, starts, keep = _expand_freq_weights(data, freq_weights)
    reference = dmatrix(formula, expanded, **kwargs)
    got = dmatrix(formula, data, freq_weights=freq_weights, **kwargs)
    assert reference.shape[1] == got.shape[1]
    assert reference.design_info.column_names == got.design_info.column_names
    assert np.allclose(np.asarray(reference)[starts], np.asarray(got))
    return got


def test_freq_weights_center_standardize():
    data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
    weights = [3, 0, 2, 1, 4]
    _assert_freq_weights_equivalent("center(x)", data, weights)
    _assert_freq_weights_equivalent("standardize(x)", data, weights)
    _assert_freq_weights_equivalent("standardize(x, ddof=1)", data, weights)
    _assert_freq_weights_equivalent("standardize(x, center=False)", data, weights)
    _assert_freq_weights_equivalent("standardize(x, rescale=False)", data, weights)
    _assert_freq_weights_equivalent(
        "standardize(x, center=False, rescale=False)", data, weights
    )
    # Multi-column inputs center/standardize column by column.
    data2 = {
        "x": np.array([1.0, 2.0, 3.0, 4.0]),
        "y": np.array([4.0, 1.0, 8.0, 2.0]),
    }
    weights2 = [2, 0, 1, 3]
    _assert_freq_weights_equivalent("center(x + y)", data2, weights2)
    _assert_freq_weights_equivalent("standardize(x + y)", data2, weights2)


def test_freq_weights_bs():
    data = {"x": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]}
    weights = [1, 0, 3, 2, 1, 4, 2]
    _assert_freq_weights_equivalent("bs(x, df=4)", data, weights)
    _assert_freq_weights_equivalent("bs(x, df=4, degree=1)", data, weights)
    _assert_freq_weights_equivalent("bs(x, df=4, degree=2)", data, weights)
    _assert_freq_weights_equivalent(
        "bs(x, df=4, include_intercept=True)", data, weights
    )
    _assert_freq_weights_equivalent("bs(x, knots=[1.5, 3.5])", data, weights)
    _assert_freq_weights_equivalent(
        "bs(x, df=4, lower_bound=0, upper_bound=6)", data, weights
    )


def test_freq_weights_nested_transforms():
    data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    weights = [2, 0, 1, 3, 1, 2]
    for formula in [
        "bs(center(x), df=4)",
        "bs(standardize(x), df=4)",
        "center(bs(x, df=4))",
        "standardize(center(x))",
        "bs(center(standardize(x)), df=4, degree=2)",
    ]:
        _assert_freq_weights_equivalent(formula, data, weights)


def test_freq_weights_categorical_and_interactions():
    data = {"x": [1.0, 2.0, 3.0, 4.0], "g": ["b", "a", "c", "a"]}
    weights = [1, 2, 0, 1]
    for formula in ["C(g)", "C(g) + x", "C(g):x", "C(g) * x", "center(x) + C(g)"]:
        _assert_freq_weights_equivalent(formula, data, weights)


def test_freq_weights_zero_rows_dropped():
    data = {"x": [1.0, 2.0, 3.0]}
    weights = [2, 0, 3]
    got = dmatrix("x", data, freq_weights=weights)
    assert got.shape == (2, 2)
    assert np.array_equal(got.freq_weights, [2, 3])
    # A category that only occurs with weight 0 does not exist.
    data = {"g": ["b", "a", "c", "a"]}
    got = dmatrix("C(g)", data, freq_weights=[1, 2, 0, 1])
    assert got.design_info.column_names == ["Intercept", "C(g)[T.b]"]


def test_freq_weights_na_handling():
    # Rows with NAs are dropped, and the weights follow them.
    data = {"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, 3.0]}
    got = dmatrix("x + y", data, freq_weights=[2, 5, 3])
    assert got.shape == (2, 3)
    assert np.array_equal(got.freq_weights, [2, 3])
    # Handle NA still applies to the weighted (compressed) data.
    import pytest

    pytest.raises(
        PatsyError,
        dmatrix,
        "x",
        data,
        NA_action="raise",
        freq_weights=[2, 5, 3],
    )


def test_freq_weights_design_info_reuse():
    data = {"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "b", "a", "b"]}
    weights = [2, 0, 1, 3]
    (design_matrix, row_weights) = freq_dmatrix(
        "bs(x, df=4) + C(g):x", data, freq_weights=weights
    )
    assert np.array_equal(row_weights, [2, 1, 3])
    assert np.array_equal(design_matrix.freq_weights, row_weights)
    assert isinstance(design_matrix.design_info, DesignInfo)
    # New data can be used directly, without weights and without relearning.
    new_data = {"x": [1.5, 2.5], "g": ["a", "b"]}
    (new_matrix,) = build_design_matrices([design_matrix.design_info], new_data)
    assert new_matrix.shape == (2, 7)
    # ...and it agrees with what the expanded-data native call would predict.
    expanded, _, _ = _expand_freq_weights(data, weights)
    (reference,) = build_design_matrices(
        [dmatrix("bs(x, df=4) + C(g):x", expanded).design_info], new_data
    )
    assert np.array_equal(new_matrix, reference)


def test_freq_weights_dataframe_index():
    from patsy.util import have_pandas

    if not have_pandas:
        return
    import pandas

    df = pandas.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", None, "b", "a"]},
        index=[10, 20, 30, 40],
    )
    weights = pandas.Series([2, 1, 0, 3], index=[10, 20, 30, 40])
    matrix = dmatrix(
        "center(x) + C(g)", df, freq_weights=weights, return_type="dataframe"
    )
    assert list(matrix.index) == [10, 40]
    assert np.array_equal(matrix.freq_weights, [2, 3])
    # The column name form works too.
    df["frequency"] = [2, 1, 0, 3]
    matrix2 = dmatrix(
        "center(x) + C(g)", df, freq_weights="frequency", return_type="dataframe"
    )
    assert np.array_equal(matrix, matrix2)
    # Mismatched pandas index is an error.
    import pytest

    pytest.raises(
        PatsyError,
        dmatrix,
        "x",
        df,
        freq_weights=pandas.Series([1, 1, 1, 1], index=[1, 2, 3, 4]),
    )


def test_freq_weights_validation_errors():
    import pytest

    data = {"x": [1.0, 2.0, 3.0]}
    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights=[-1, 1, 1])
    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights=[1.5, 1, 1])
    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights=[1, 2])
    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights=[1, 2, 3, 4])
    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights=[0, 0, 0])
    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights=[np.nan, 1, 1])
    pytest.raises(
        PatsyError, dmatrix, "x", data, freq_weights=np.array([[1, 2], [3, 4]])
    )
    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights="nope")
    # Explicit matrices do not go through the formula machinery.
    pytest.raises(
        PatsyError,
        dmatrix,
        np.ones((3, 1)),
        freq_weights=[1, 1, 1],
    )
    # freq_dmatrix/freq_dmatrices require the argument.
    pytest.raises(PatsyError, freq_dmatrix, "x", data)
    pytest.raises(PatsyError, freq_dmatrices, "y ~ x", {"x": [1.0], "y": [1.0]})


def test_freq_dmatrices():
    data = {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}
    weights = [2, 0, 3]
    ((lhs, rhs), row_weights) = freq_dmatrices("y ~ x", data, freq_weights=weights)
    assert lhs.shape == (2, 1)
    assert rhs.shape == (2, 2)
    assert np.array_equal(row_weights, [2, 3])
    assert np.array_equal(lhs.freq_weights, [2, 3])
    # Equivalence with expanded data.
    expanded, starts, _ = _expand_freq_weights(data, weights)
    (ref_lhs, ref_rhs) = dmatrices("y ~ x", expanded)
    assert np.array_equal(np.asarray(ref_lhs)[starts], np.asarray(lhs))
    assert np.array_equal(np.asarray(ref_rhs)[starts], np.asarray(rhs))


def test_freq_weights_non_weighted_unchanged():
    data = {"x": [1.0, 2.0, 3.0]}
    plain = dmatrix("center(x)", data)
    assert not hasattr(plain, "freq_weights")
    assert np.allclose(plain, dmatrix("center(x)", {"x": [1.0, 2.0, 3.0]}))


def test_freq_weights_without_data_factors():
    # "~ 1" refers to no data column at all, so the row count has to come
    # from the weights.
    for formula in ["~ 1", "1", "~ 0"]:
        got = dmatrix(formula, {}, freq_weights=[1, 2, 3], return_type="dataframe")
        assert got.shape == (3, got.shape[1])
        assert list(got.index) == [0, 1, 2]
        assert np.array_equal(got.freq_weights, [1, 2, 3])
    # Zero-weight rows are removed even here.
    got = dmatrix("~ 1", {}, freq_weights=[0, 2, 0], return_type="dataframe")
    assert got.shape == (1, 1)
    assert list(got.index) == [0]
    assert np.array_equal(got.freq_weights, [2])
    # Without weights the old error is unchanged.
    import pytest

    pytest.raises(PatsyError, dmatrix, "~ 1", {}, return_type="dataframe")


def test_freq_weights_reserved_name():
    from patsy.freq_weights import _FREQ_WEIGHTS_KEY

    data = {"x": [1.0, 2.0, 3.0], _FREQ_WEIGHTS_KEY: [9, 9, 9]}
    import pytest

    pytest.raises(PatsyError, dmatrix, "x", data, freq_weights=[1, 2, 3])
    # Without frequency weights the column is just an odd name.
    assert dmatrix("x", data).shape == (3, 2)


def test_freq_weights_literal_transform():
    # A transform applied to a literal is not row-parallel with the data, so
    # it must memorize exactly as it always would, weights or no weights.
    data = {"x": [1.0, 2.0, 3.0]}
    weights = [1, 2, 3]
    literal = "center([1.0, 2.0, 3.0])"
    assert np.array_equal(
        dmatrix(literal, data, freq_weights=weights), dmatrix(literal, data)
    )
    # A transform of a data column *is* weighted, and here it really matters:
    # the weighted mean of x is (1*1 + 2*2 + 3*3) / 6, not 2.
    got = dmatrix("center(x)", data, freq_weights=weights)
    assert not np.allclose(got, dmatrix("center(x)", data))
    assert np.allclose(got[:, 1], np.array([1.0, 2.0, 3.0]) - 14.0 / 6.0)


def test_freq_weights_cubic_splines():
    data = {
        "x": np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "z": np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0]),
    }
    weights = [1, 0, 3, 2, 1, 4]
    for formula in [
        "cr(x, df=4)",
        "cc(x, df=4)",
        "cr(x, df=4, constraints='center')",
        "cc(x, df=5, constraints='center')",
        "te(cr(x, df=4), cc(z, df=4), constraints='center')",
    ]:
        _assert_freq_weights_equivalent(formula, data, weights)
