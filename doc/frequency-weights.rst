.. _freq-weights:

Frequency weights
=================

Sometimes a data set has already been compressed before it reaches you: each
distinct row appears once, together with a ``frequency`` column recording how
many times that row occurred in the original data. Expanding such a table to
the full row set can be very expensive, but the model does not actually need
the expanded rows -- it only needs the *identical* answer that expanding would
have produced.

Patsy therefore accepts an optional ``freq_weights=`` argument on
:func:`~patsy.dmatrix` and :func:`~patsy.dmatrices`, plus the
:func:`~patsy.freq_dmatrix` and :func:`~patsy.freq_dmatrices` convenience
wrappers::

  (mat, weights) = freq_dmatrix("y ~ bs(x, df=4) + C(g)",
                                compressed_df, freq_weights="frequency")

The weights must be non-negative integers, with one entry per row of the data
(or the name of a column holding them). They may also be zero: a row with
frequency zero does not occur in the expanded data at all, so it is removed
before anything is learned from the data.

What is guaranteed
------------------

The learned state -- and therefore the design matrix -- is exactly what the
native API would have produced from the physically expanded data:

* ``center`` memorizes the weighted mean, and ``standardize`` the weighted
  mean and variance (honoring ``center``, ``rescale``, and ``ddof``).
* ``bs`` uses the same interior knots, boundary knots, and boundary-checking
  behavior as if each row were repeated.
* Categorical coding and interactions see the same levels in the same order,
  because rows with weight zero are removed before the levels are sniffed.
* Nested stateful transforms such as ``bs(center(x))`` obey the weights in
  every memorization pass, since the weights are attached to each data chunk
  rather than only to one transform.
* Missing values are handled exactly as in the non-weighted case; the returned
  weights are subset to the rows that survive NA handling, so they always line
  up with the returned matrix.

Prediction
----------

The returned matrix carries a real :class:`~patsy.DesignInfo` in its
``.design_info`` attribute, and the surviving weights in its ``.freq_weights``
attribute. The design info can be handed straight to
:func:`~patsy.build_design_matrices` to build matrices for new data:

.. ipython:: python
   :suppress:

   import numpy as np
   from patsy import *

.. ipython:: python

   compressed = {"x": [1.0, 2.0, 3.0], "frequency": [2, 0, 3]}
   mat = dmatrix("center(x)", compressed, freq_weights=compressed["frequency"])
   mat
   mat.freq_weights
   build_design_matrices([mat.design_info], {"x": [0.0, 10.0]})[0]

New data needs neither weights nor relearning: everything that depended on the
whole data set is already stored in the design info.

Errors
------

Weights that cannot possibly be replicate counts -- negative values,
non-integers, NaN or infinite values, the wrong length, or more than one
dimension -- raise :class:`~patsy.PatsyError` with a message describing the
problem.

Existing behavior without ``freq_weights`` is unchanged.
