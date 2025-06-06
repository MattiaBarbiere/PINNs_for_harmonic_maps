"""
The author of this code is Jochen Hinz.
"""



from typing import Sequence

import numpy as np
from numba import njit, prange
from numba.typed import List


""" itertools-equivalent Numba implementations. """


@njit(cache=True)
def _product(arr0: np.ndarray, list_of_linspaces: Sequence[np.ndarray]):
  """
  Given :class:`np.ndarray` `arr0` and :class:`list` of :class:`np.ndarray`s
  ``list_of_linspaces``,  create a column tensor product with ``arr0`` and
  all arrays in ``list_of_linspaces``.
  The input ``arr0`` is assumed to be two-dimensional, i.e., in the case of a
  single array, the shape needs to be ``(npoints, 1)``.
  """
  while True:
    lin0, list_of_linspaces = list_of_linspaces[0], list_of_linspaces[1:]
    n, m = len(lin0), len(arr0)
    m, ndims = arr0.shape[0], arr0.shape[1]

    ret = np.empty((len(arr0) * n, ndims + 1), dtype=arr0.dtype)
    for i in range(n):
      ret[i * m: (i+1) * m, -ndims:] = arr0

    counter = 0
    for i in range(n):
      myval = lin0[i]
      for j in range(arr0.shape[0]):
        ret[counter, 0] = myval
        counter += 1
    if len(list_of_linspaces) == 0:
      return ret
    arr0 = ret


@njit(cache=True)
def product(list_of_arrays):
  """
  Numba equivalent of the ``itertools.product`` iterator with the difference
  that it can be used inside of Numba, works only with array inputs and
  creates all products at once.
  >>> linspaces = [np.linspace(0, 1, i) for i in (4, 5, 6)]
  >>> X = np.stack(list(map(np.ravel, np,meshgrid(*linspaces))), axis=1)
  >>> Y = _product(linspaces)
  >>> np.allclose(X, Y)
      True
  """
  # XXX: This is a workaround for the fact that Numba does not support
  #      itertools.product. This implementation creates all products at once.
  #      This is not a problem for small arrays, but can be for
  #      large ones. In the long run, we should implement a more efficient
  #      version of this function. For now, this is a good enough solution.

  assert len(list_of_arrays) >= 1
  # If there is only one array, we need to add a dimension to it
  if len(list_of_arrays) == 1:
    return list_of_arrays[0][:, None]
  # Reverse the list of arrays to get the correct order
  list_of_arrays = list_of_arrays[::-1]
  # Call the recursive function
  return _product(list_of_arrays[0][:, None], list_of_arrays[1:])


@njit(cache=True)
def linspace_product(array_of_steps):
  """
  Convenience function for creating a product of linspaces
  over [0, 1] from an array of integers representing the steps.
  """
  list_of_arrays = []
  for i in array_of_steps:
    list_of_arrays.append(np.linspace(0, 1, i))
  return product(list_of_arrays)


@njit(cache=True)
def arange_product(array_of_integers):
  """
  Convenience function for creating a product of aranges
  from [0, i) from an array of integers containing the `i`.
  """
  list_of_arrays = []
  for i in array_of_integers:
    list_of_arrays.append(np.arange(i).astype(np.int64))
  return product(list_of_arrays)


"""
Various custom implementations of numpy functions not yet supported in Numba
"""


@njit(cache=True)
def ravel_multi_index(multi_index, dims):
  """
  Numba implementation of np.ravel_multi_index.
  """
  flat_index = 0
  stride = 1

  # Loop through dimensions in reverse order to calculate the flat index
  for i in range(len(dims) - 1, -1, -1):
    flat_index += multi_index[i] * stride
    stride *= dims[i]

  return flat_index


@njit(cache=True)
def unravel_multi_index(flat_index, dims):
  """
  Numba implementation of np.unravel_multi_index.

  Parameters
  ----------
  flat_index : int
      The flattened index to convert.
  dims : np.ndarray
      The shape of the multi-dimensional array.

  Returns
  -------
  multi_index : tuple of ints
      The multi-dimensional indices corresponding to the flat index.
  """
  multi_index = np.empty(len(dims), dtype=np.int64)

  for i in range(len(dims) - 1, -1, -1):
      multi_index[i] = flat_index % dims[i]  # Get the remainder
      flat_index //= dims[i]  # Update the flat index for the next dimension

  return multi_index


@njit(cache=True)
def position_in_knotvector(t, x):
  """
  Return the position of ``x`` in the knotvector ``t``.
  If x equals t[-1], return the position before the first
  occurence of x in t.

  Parameters
  ----------
  t : :class:`np.ndarray`
      The knotvector with repeated knots.
  x : :class:`np.ndarray`
      The vector of positions.

  Returns
  -------
  ret : :class:`np.ndarray` comprised of integers
      The positions in the knotvector. Has the same length as `x`.
      If entry is not found, defaults to -1.
  """
  ret = np.empty(len(x), dtype=np.int64)
  for i in range(len(x)):
    myx = x[i]
    if myx < t[0] or myx > t[-1]:
      ret[i] = -1
    elif myx == t[-1]:
      ret[i] = np.searchsorted(t, myx) - 1
    else:
      ret[i] = np.searchsorted(t, myx, side='right') - 1
  return ret


@njit(cache=True)
def nonzero_bsplines(mu, x, t, d):
  """
  Return the value of the d+1 nonzero basis
  functions at position ``x``.

  Parameters
  ----------
  mu : :class:`int`
      The position in `t` that contains `x`,
  x: :class:`float`
      The position,
  t: :class:`np.ndarray`
      The knotvector.
  d: :class:`int`
      The degree of the B-spline basis.

  Returns
  -------
  b : :class:`np.ndarray`
      The nonzero bsplines evalated in `x`
  """

  b = np.zeros(d + 1, dtype=np.float64)
  b[-1] = 1

  if x == t[-1]:
    return b

  for r in range(1, d + 1):

    k = mu - r + 1
    w2 = (t[k + r] - x) / (t[k + r] - t[k])
    b[d - r] = w2 * b[d - r + 1]

    for i in range(d - r + 1, d):
      k = k + 1
      w1 = w2
      w2 = (t[k + r] - x) / (t[k + r] - t[k])
      b[i] = (1 - w1) * b[i] + w2 * b[i + 1]

    b[d] = (1 - w2) * b[d]

  return b


@njit(cache=True)
def nonzero_bsplines_deriv(kv, p, x, dx):
  """
  Return the value of the d+1 nonzero basis
  functions and their derivatives up to order `dx` at position `x`.

  Parameters
  ----------
  kv : :class:`np.ndarray`
      The knotvector.
  p : :class:`int`
      The degree of the B-spline basis.
  x : :class:`float`
      The position.
  dx : :class:`int`
      The highest-order derivative.

  Returns
  -------
  ders : :class:`np.ndarray`
      The nonzero bsplines evalated in `x` and their derivatives up to order `dx`.
  """
  # Initialize variables
  span = position_in_knotvector(kv, np.array([x], dtype=np.float64))[0]
  left = np.ones((p + 1,), dtype=np.float64)
  right = np.ones((p + 1,), dtype=np.float64)
  ndu = np.ones((p + 1, p + 1), dtype=np.float64)

  for j in range(1, p + 1):
    left[j] = x - kv[span + 1 - j]
    right[j] = kv[span + j] - x
    saved = 0.0
    r = 0
    for r in range(r, j):
      # Lower triangle
      ndu[j, r] = right[r + 1] + left[j - r]
      temp = ndu[r, j - 1] / ndu[j, r]
      # Upper triangle
      ndu[r, j] = saved + (right[r + 1] * temp)
      saved = left[j - r] * temp
    ndu[j, j] = saved

  # Load the basis functions
  # ders = [[0.0 for _ in range(p + 1)] for _ in range((min(p, dx) + 1))]
  ders = np.zeros((min(p, dx) + 1, p + 1), dtype=np.float64)
  for j in range(0, p + 1):
    ders[0, j] = ndu[j, p]

  # Start calculating derivatives
  # a = [[1.0 for _ in range(p + 1)] for _ in range(2)]
  a = np.ones((2, p + 1), dtype=np.float64)
  # Loop over function index
  for r in range(0, p + 1):
    # Alternate rows in array a
    s1 = 0
    s2 = 1
    a[0, 0] = 1.0
    # Loop to compute k-th derivative
    for k in range(1, dx + 1):
      d = 0.0
      rk = r - k
      pk = p - k
      if r >= k:
        a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
        d = a[s2, 0] * ndu[rk, pk]
      if rk >= -1:
        j1 = 1
      else:
        j1 = -rk
      if (r - 1) <= pk:
        j2 = k - 1
      else:
        j2 = p - r
      for j in range(j1, j2 + 1):
        a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
        d += (a[s2, j] * ndu[rk + j, pk])
      if r <= pk:
        a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
        d += (a[s2, k] * ndu[r, pk])
      ders[k, r] = d

      # Switch rows
      j = s1
      s1 = s2
      s2 = j

  # Multiply through by the the correct factors
  r = float(p)
  for k in range(1, dx + 1):
    for j in range(0, p + 1):
      ders[k, j] *= r
    r *= (p - k)

  # Return the basis function derivatives list
  return ders


@njit(cache=True)
def nonzero_bsplines_deriv_vectorized(kv, p, x, dx):
  """
  Vectorized (in x) version of `nonzero_bsplines_deriv`
  only returns the dx-th derivative though.

  Parameters are the same as in `nonzero_bsplines_deriv` but `x` is a vector.
  """
  ret = np.empty((len(x), p+1), dtype=np.float64)
  for i in prange(len(ret)):
    ret[i] = nonzero_bsplines_deriv(kv, p, x[i], dx)[dx]
  return ret


@njit(cache=True)
def der_ith_basis_fun( kv, p, i, x, dx ):  # based on algorithm A2.5 from the NURBS-book
  """
  Return the N_ip(x) and its derivatives up to ``dx``,
  where N denotes the ``i``-th basis function of order
  ``p`` resulting from knotvector ``kv`` and x the position.

  Parameters
  ----------
  kv : :class:`np.ndarray`
      The knotvector.
  p : :class:`int`
      The degree of the B-spline basis.
  i : :class:`int`
      The index of the basis function.
  x : :class:`float`
      The position.
  dx : :class:`int`
      The highest-order derivative.

  Returns
  -------
  ders : :class:`np.ndarray`
      The basis function and its derivatives up to order `dx`.
  """

  if x == kv[-1]:
    x -= 1e-15

  basis_len = len(kv) - p - 1

  if x < kv[i] or x >= kv[i + p + 1]:
    if i != basis_len - 1 or x > kv[-1]:
      ''' x lies outside of the support of basis function or domain '''
      return np.zeros( (dx + 1, ), dtype=np.float64 )
    if i == basis_len - 1 and x == kv[-1]:
      '''
      special case: evaluation of the last basis function
      in the last point of the interval. Return a sequence
      (p / a_0) ** 0, (p / a_1) ** 1, ... (p / a_dx) ** dx
      '''
      # a = 1
      # ret = np.empty( (dx + 1, ), dtype=float64 )
      # for i in range( ret.shape[0] ):
      #     ret[i] = a
      #     if i != ret.shape[0] - 1:
      #         a *= p / ( kv[basis_len - 1 + p - i] - kv[basis_len - 1 - i] )
      # return ret
      x -= 1e-15

  ders = np.empty( (dx + 1, ), dtype=np.float64 )
  N = np.zeros( (p + 1, p + 1), dtype=np.float64 )

  for j in range(p + 1):
    if ( x >= kv[i + j] and x < kv[i + j + 1] ):
      N[j, 0] = 1.0
    else:
      N[j, 0] = 0.0

  for k in range(1, p + 1):
    saved = 0.0 if N[0, k - 1] == 0.0 else \
        (x - kv[i]) * N[0, k - 1] / (kv[i + k] - kv[i])
    for j in range(p - k + 1):
      Uleft, Uright = kv[i + j + 1], kv[i + j + k + 1]
      if N[j + 1, k - 1] == 0:
        N[j, k], saved = saved, 0
      else:
        temp = N[j + 1, k - 1] / (Uright - Uleft)
        N[j, k] = saved + (Uright - x) * temp
        saved = (x - Uleft) * temp

  ders[0] = N[0, p]
  ND = np.zeros( (k + 1, ), dtype=np.float64 )
  for k in range(1, dx + 1):
    for j in range(k + 1):
      ND[j] = N[j, p - k]
    for jj in range(1, k + 1):
      saved = 0.0 if ND[0] == 0.0 else ND[0] / (kv[i + p - k + jj] - kv[i])
      for j in range(k - jj + 1):
        # wrong in the NURBS book, -k is missing in Uright
        Uleft, Uright = kv[i + j + 1], kv[i + j + p - k + jj + 1]
        if ND[j + 1] == 0.0:
          ND[j], saved = (p - k + jj) * saved, 0.0
        else:
          temp = ND[j + 1] / (Uright - Uleft)
          ND[j] = (p - k + jj) * (saved - temp)
          saved = temp

    ders[k] = ND[0]
  return ders


@njit(cache=True)
def _call1D(xi, kv0, p0, x, dx):
  """
  Return function evaluations at positions xi.

  Parameters
  ----------
  xi : :class:`np.ndarray`
      The positions.
  kv0 : :class:`np.ndarray`
      The knotvector.
  p0 : :class:`int`
      The degree of the B-spline basis.
  x : :class:`np.ndarray`
      The control points. Unlike `_callND`, this is a 1D array.
  dx : :class:`int`
      The highest-order derivative.

  Returns
  -------
  ret : :class:`np.ndarray`
      The function evaluations at positions xi.

  This version is fully sequential.
  """

  # XXX:

  ret = np.zeros(xi.shape, dtype=np.float64)
  assert ret.ndim == 1
  element_indices0 = position_in_knotvector(kv0, xi)

  for i in range(len(xi)):
    xi_calls = nonzero_bsplines_deriv(kv0, p0, xi[i], dx)[dx]
    for j in range(p0 + 1):
      a = xi_calls[j]
      global_index = element_indices0[i] - p0 + j
      ret[i] += x[global_index] * a

  return ret


@njit(cache=True, parallel=True)
def _callND(Xi, list_of_knotvectors, degrees, controlpoints, derivatives, into):
  """
  Return function evaluations (or their derivatives) of a nD tensor product
  spline at positions Xi.

  Parameters
  ----------
  Xi : :class:`np.ndarray`
      The positions of shape (nentries, ncoords). Different coordinates in the columns.
  list_of_knotvectors : :class:`List`
      List containing the knotvectors of the ncoords directions.
  degrees : :class:`np.ndarray`
      The polynomial degrees in each direction.
  controlpoints : :class:`np.ndarray`
      The control points. Of shape (naxes, ndofs).
  derivatives : :class:`np.ndarray`
      The highest-order derivatives in each direction.
  into : :class:`np.ndarray`
      The array to store the results. Must be of shape (nentries, naxes).

  This version is parallelized along the `naxes` coordinate.
  """

  assert Xi.shape[0] == into.shape[0]
  nentries, naxes = into.shape
  assert controlpoints.shape[:1] == (naxes,)

  # make len(list_of_knotvectors) - length homogeneous container containing
  # temporary `into` arrays
  container = [np.zeros((nentries,), dtype=np.float64) for _ in range(naxes)]

  # make len(list_of_knotvectors) - shaped integer array with the ndofs per direction
  dims = np.empty(len(list_of_knotvectors), dtype=np.int64)
  for i, (kv, degree) in enumerate(zip(list_of_knotvectors, degrees)):
    dims[i] = kv.shape[0] - degree - 1

  # make an outer product flat meshgrid with aranges from 0 to p + 1
  inner_loop_indices = arange_product(degrees + 1)

  # make integer array containing the positions in the knotvectors of the univariate
  # contributions in `Xi`
  element_indices = np.empty((nentries, len(list_of_knotvectors)), dtype=np.int64)
  for i, (mykv, xi) in enumerate(zip(list_of_knotvectors, Xi.T)):
    element_indices[:, i] = position_in_knotvector(mykv, xi)

  for iaxis in prange(naxes):
    x = controlpoints[iaxis]
    myinto = container[iaxis]

    for i in range(nentries):

      # get all univariate local calls
      mycalls = [nonzero_bsplines_deriv(kv, p, xi, dx)[dx, :] for kv, p, xi, dx
                 in zip(list_of_knotvectors, degrees, Xi[i], derivatives)]

      for multi_index in inner_loop_indices:
        # global index in x results from the multi_index + the element_indices minus the degrees
        # and the `dims` vector
        global_index = ravel_multi_index(element_indices[i] + multi_index - degrees, dims)

        # add product of all evaluations times the weight to the corresponding
        # position
        myval = x[global_index]
        for j, myindex in enumerate(multi_index):
          myval = myval * mycalls[j][myindex]

        myinto[i] += myval

      into[:, iaxis] = myinto


def call(Xi,
         list_of_knotvectors,
         list_of_degrees,
         controlpoints,
         dx=None,
         into=None):
  """
  Return function evaluations at positions Xi.

  Optionally put them into a preallocated array ``into``.

  For a detailed docstring, see `_callND`.
  """

  nvars, = Xi.shape[1:]

  assert controlpoints.ndim == 2, "Please provide a 2D array of control points."

  assert 1 <= nvars <= 3
  if dx is None:
    dx = 0
  if np.isscalar(dx):
    dx = (dx,) * nvars

  degrees = np.asarray(list_of_degrees, dtype=int)
  dx = np.asarray(dx, dtype=int)

  assert len(list_of_knotvectors) == len(list_of_degrees) \
                                  == len(dx)

  if into is None:
    into = np.zeros(Xi.shape[:1] + controlpoints.shape[:1], dtype=np.float64)

  assert into.shape == Xi.shape[:1] + controlpoints.shape[:1]

  _callND(Xi,
          List(list_of_knotvectors),
          degrees,
          controlpoints, dx, into)

  return into
