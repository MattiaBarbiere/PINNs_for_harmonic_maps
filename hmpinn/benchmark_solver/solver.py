"""
The author of this code is Jochen Hinz.
"""

import hmpinn.benchmark_solver.jitBSpline as jitBSpline

from typing import Optional, Callable, Tuple, Any, Union
from itertools import repeat
from functools import cached_property
from collections.abc import Hashable
from nutils.function import _dtypes


import numpy as np
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')  # For non-GUI rendering

from nutils import function, mesh, solver

from numpy.typing import NDArray, ArrayLike


DEBUG_SHAPE = False

if DEBUG_SHAPE:
  print("_dytes from nutils.function is ", _dtypes)

SIDES = 'left', 'right', 'bottom', 'top'


def frozen(arr: ArrayLike, dtype=None):
  """
  Coerce and freeze inplace.
  """
  arr = np.asarray(arr, dtype=dtype)
  arr.flags.writeable = False
  return arr


def multi_indices(dx: int):
  """
  dx = 0: ((0, 0),)
  dx = 1: ((1, 0), (0, 1))
  dx = 2: ((2, 0), (1, 1), (1, 1), (0, 2))
  ...
  """
  assert dx >= 0

  if dx == 0:
    yield (0, 0)
    return

  for mindex in multi_indices(dx-1):
    for j in range(2):
      yield mindex[:j] + (mindex[j] + 1,) + mindex[j+1:]


class SplineSolution:
  """
  Immutable object representing a spline solution to a PDE.
  """

  def __init__(self, nx: int, ny: int, p: int, controlpoints: ArrayLike) -> None:
    self.nx = int(nx)
    self.ny = int(ny)
    self.p = int(p)
    assert all( n > 0 for n in (self.nx, self.ny, self.p) )
    self.controlpoints = frozen(controlpoints, dtype=float)
    assert self.controlpoints.shape == (self.ndofs,), \
      "Number of DOFs and dimension do not match. \n" \
      f"Expected controlpoints.shape == {(self.ndofs,)}, " \
      f"found {self.controlpoints.shape}."

  def __repr__(self) -> str:
    return f'SplineSolution(nx={self.nx}, ny={self.ny}, p={self.p})'

  @property
  def ndofs(self):
    return (self.nx + self.p) * (self.ny + self.p)

  @cached_property
  def _args(self) -> Tuple[Hashable]:
    return self.nx, self.ny, self.p, \
           (self.controlpoints.tobytes(), self.controlpoints.shape)

  @cached_property
  def _hash(self) -> int:
    return hash(self._args)

  @cached_property
  def knotvectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    kvs = (np.repeat(np.linspace(0, 1, i+1), [self.p+1] + [1] * (i-1) + [self.p+1])
                                     for i in [self.nx, self.ny])
    return tuple(map(frozen, kvs))

  def __hash__(self) -> int:
    return self._hash

  def __eq__(self, other: Any) -> bool:
    if self is other: return True
    return self.__class__ is other.__class__ and \
           hash(self) is hash(other) and \
           self._args == other._args

  def __call__(self, X: ArrayLike, Y: ArrayLike, dx: int = 0) -> NDArray[np.float64]:
    """
    Call the spline solution with the given arguments.
    Output shape: X.shape + (2,) * dx.
    Note that if dx == 0, then (2,) * dx == () and thus, output_shape == X.shape.

    Parameters
    ----------
    X : :class:`ArrayLike`
      The x-coordinates of the points to evaluate the spline at.
    Y : :class:`ArrayLike`
      The y-coordinates of the points to evaluate the spline at.
      After coercion to :class:`np.ndarray`, X.shape == Y.shape has to hold.
    dx : :class:`int`
      The order of the derivative to evaluate at. Defaults to 0.
      For dx > 0, we take all partial derivatives of the spline solution
      whose order sums to `dx`.
    """

    X, Y = map(np.asarray, (X, Y))
    assert X.shape == Y.shape

    Xi = np.stack(list(map(np.ravel, (X, Y))), axis=1)

    ret = np.empty((np.prod(X.shape), 2 ** dx), dtype=float)
    seen = {}

    for i, mindex in enumerate(multi_indices(dx)):
      try:
        ret[:, i] = seen[mindex]
      except KeyError:
        ret[:, i] = seen.setdefault(mindex, jitBSpline.call(Xi,
                                                 self.knotvectors,
                                                 [self.p]*2,
                                                 self.controlpoints[:, None].T,
                                                 dx=mindex).ravel())

    return ret.reshape(X.shape + (2,) * dx)

  def plot(self, nx: int = 101,
                 ny: Optional[int] = None,
                 title: Optional[str] = None,
                 block=True) -> None:
    if ny is None:
      ny = nx

    X, Y = np.meshgrid(*map(lambda x: np.linspace(0, 1, x), (nx, ny)))
    Z = self(X, Y)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=Z.min(), vmax=Z.max())
    ax.set_title(title or 'Solution')
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)

    plt.show(block=block)


def solve(A: Callable, f: Optional[Callable] = None,
                       nx: int = 21,
                       ny: Optional[int] = None,
                       p: int = 3,
                       bc: Optional[Union[Callable, dict]] = None,
                       nondiv=False) -> SplineSolution:
  """
  Solve a problem of the form:
    \nabla \cdot (A \nabla u) - f = 0  if nondiv is False or else
                     A : H(u) - f = 0

  and return as :class:`SplineSolution`.

  Parameters
  ----------
  A : :class:`Callable`
    Coefficient function A(x, y) that creates a 2x2 tensor representing the
    diffusion coefficients or the nondiv analogue.
  f : :class:`Callable`
    Source term f(x, y) that is added to the right-hand side.
  nx : :class:`int`
    Number of elements in the x-direction.
  ny : :class:`int` or None
    Number of elements in the y-direction. If None, then nx == ny.
  p : :class:`int`
    Degree of the spline basis. Defaults to `p = 3`.
  bc : :class:`Callable` or dict or None
    Boundary condition function g(x, y) or a dictionary of the form
    {side: g(x, y)} where side is one of 'left', 'right', 'bottom', 'top'.
    If passed as a function g(x, y), it gets coerced into the dict
    {side: g(x, y)} for all sides.
    If None, then it gets coerced into g(x, y) = 0 for all sides.
  nondiv : :class:`bool`
    If True, then the nondiv analogue of the PDE is solved.

  Returns
  -------
  :class:`SplineSolution`
    The solution to the PDE as a :class:`SplineSolution` object.
  """

  # TODO: Find a better solution for the `nondiv` keyword argument.

  if ny is None:
    ny = nx

  if bc is None:
    bc = lambda x, y: 0

  if isinstance(bc, Callable):
    bc = dict(zip(SIDES, repeat(bc)))

  assert set(bc.keys()).issubset(set(SIDES)) and len(bc), \
    "Need the BC to be set on at least one edge {} explicitly.".format(SIDES)

  if f is None:
    f = lambda x: 0

  domain, geom = mesh.rectilinear([np.linspace(0, 1, i + 1) for i in [nx, ny]])
  basis = domain.basis('spline', degree=p)

  A = A(*geom)
  assert A.shape == (2, 2), f"A must be a 2x2 but got {A.shape}."

  f = f(*geom)

  bc = {key: func(*geom) for key, func in bc.items()}

  ns = function.Namespace()
  ns.x = geom
  ns.A = A
  ns.f = f
  ns.phi = basis
  ns.dphi = function.laplace(basis, geom)


  if DEBUG_SHAPE:
    print("Shape of ns.x is ", ns.x.shape, " and has shape ", ns.x.dtype)
    print("Shape of ns.A is ", ns.A.shape," and has shape ", ns.A.dtype)
    print("Shape of ns.f is ", ns.f.shape," and has shape ", ns.f.dtype)
    print("Shape of ns.phi is ", ns.phi.shape," and has shape ", ns.phi.dtype)
    print("Shape of ns.dphi is ", ns.dphi.shape," and has shape ", ns.dphi.dtype)

  cons = None
  for side, func in bc.items():
    if DEBUG_SHAPE:
      print("Shape of bc on ", side, " side is ", func.shape," and has shape ", func.dtype)
    cons = domain.boundary[side].project(func, onto=basis,
                                               geometry=geom,
                                               ischeme=f'gauss{p * 2}',
                                               constrain=cons)

  if nondiv:
    if DEBUG_SHAPE:
      print("nondiv is True")
    resstr = "dphi_k (A_ij (phi_n ?lhs_n)_,ij - f) d:x"
  else:
    if DEBUG_SHAPE:
      print("nondiv is False")
    resstr = "(-phi_k,i A_ij (phi_n ?lhs_n)_,j - phi_k f) d:x"
  
  res = domain.integral(resstr @ ns, degree=2*p)

  controlpoints = solver.solve_linear('lhs', res, constrain=cons)

  return SplineSolution(nx, ny, p, controlpoints)


if __name__ == '__main__':
  A = lambda x, y: np.eye(2) * (1 + np.sin(x) ** 2)
  f = lambda *geom: 10 * np.exp(-.5 * sum( (_x - .5)**2 for _x in geom ))

  bc = lambda x, y: np.float64(0.0)  # Dirichlet BC on each side is just g(x, y) = x

  res = solve(A, f, bc=bc, nondiv=False)

  res.plot()

  X, Y = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))

  print(res(X, Y, dx=2).shape)
