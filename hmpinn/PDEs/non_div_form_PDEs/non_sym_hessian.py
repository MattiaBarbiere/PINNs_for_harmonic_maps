"""
Non-Symmetric Hessian PDE.

This module provides a PDE implementation with non-symmetric Hessian operator
and known analytical solution that has singularities at the origin.

Classes:
    NonSymHessian: PDE with non-symmetric Hessian operator and analytical solution.
"""

import torch
from functools import partial

import nutils.function as fn

from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str, frobenius_prod, zeros, ones, all


def u(x, backend=torch):
    """
    Analytical solution with singularity at origin.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Analytical solution values, zero at origin and defined elsewhere.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1

    # Check if coordinates are at origin
    zeros_tensor = zeros(x, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    # Compute solution for non-zero points
    x1 = x[:, 0]
    x2 = x[:, 1]
    non_zero_values = x1 * x2 * (x1**2 - x2**2) / (x1**2 + x2**2)

    # Return zero at origin, computed values elsewhere
    result = backend.where(is_zero, zeros_tensor[:, 0], non_zero_values)
    return result


def grad_u(x, backend=torch):
    """
    Gradient of the analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Gradient of the analytical solution.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    # Transform input to range [-1, 1]
    x = 2 * x - 1

    # Check if coordinates are at origin
    zeros_tensor = zeros(x, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    # Compute gradient for non-zero points
    x1 = x[:, 0]
    x2 = x[:, 1]
    grad_x = (x2 * (x1**4 + 4 * x1**2 * x2**2 - x2**4)) / (x1**2 + x2**2)**2
    grad_y = (x1 * (x1**4 - 4 * x1**2 * x2**2 - x2**4)) / (x1**2 + x2**2)**2
    
    # Apply zero condition at origin
    grad_x = backend.where(is_zero, zeros_tensor[:, 0], grad_x)
    grad_y = backend.where(is_zero, zeros_tensor[:, 0], grad_y)
    
    return stack([grad_x, grad_y], dim=1, backend=backend)


def hessian_u(x, backend=torch):
    """
    Hessian matrix of the analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Hessian matrix with special handling at the origin.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1

    # Check if coordinates are at origin
    zeros_tensor = zeros(x, backend=backend)
    ones_tensor = ones(x, add_extra_dim=True, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    x1 = x[:, 0]
    x2 = x[:, 1]

    # Compute Hessian components for non-zero points
    hessian_xx = -4 * x1 * x2**3 * (x1**2 - 3 * x2**2) / (x1**2 + x2**2)**3
    hessian_xy = (x1**6 + 9 * x1**4 * x2**2 - 9 * x2**4 * x1**2 - x2**6) / (x1**2 + x2**2)**3
    hessian_yx = hessian_xy
    hessian_yy = 4 * x1**3 * x2 * (-3 * x1**2 + x2**2) / (x1**2 + x2**2)**3

    # Construct Hessian matrix with special values at origin
    batch_size = x.shape[0]
    hessian = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    hessian[:, 0, 0] = backend.where(is_zero, zeros_tensor[:, 0], hessian_xx)
    hessian[:, 0, 1] = backend.where(is_zero, -ones_tensor[:, 0], hessian_xy)
    hessian[:, 1, 0] = backend.where(is_zero, ones_tensor[:, 0], hessian_yx)
    hessian[:, 1, 1] = backend.where(is_zero, zeros_tensor[:, 0], hessian_yy)

    return hessian


def diffusion_matrix(x, model=None, backend=torch):
    """
    Non-symmetric diffusion matrix with fractional power coupling.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        model: torch.nn.Module, optional
            Model instance (unused in this implementation).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor or fn.Array
            Non-symmetric diffusion matrix with off-diagonal coupling.
    """
    # Transform input to range [-1, 1]
    x = 2 * x - 1

    if isinstance(x, fn.Array):
        # Symbolic case using nutils.function
        b = (x[0, 0]**2 * x[0, 1]**2)**(1/3)
        
        # Create the symbolic diffusion matrix
        diffusion = fn.asarray([[
            [1, b],
            [b, 2],
        ]])

        return diffusion

    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = 1
    diffusion[:, 0, 1] = (x[:, 0]**2 * x[:, 1]**2)**(1/3)
    diffusion[:, 1, 0] = (x[:, 0]**2 * x[:, 1]**2)**(1/3)
    diffusion[:, 1, 1] = 2
    return diffusion


def f(x, backend=torch):
    """
    Source term computed as Frobenius product of diffusion and Hessian.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Source term values at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return frobenius_prod(diffusion_matrix(x, backend=backend), hessian_u(x, backend=backend))


def boundary_condition(x, backend=torch):
    """
    Boundary condition matching the analytical solution.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Boundary condition values.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1

    # Check if coordinates are at origin
    zeros_tensor = zeros(x, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    # Compute boundary values for non-zero points
    x1 = x[:, 0]
    x2 = x[:, 1]
    non_zero_values = x1 * x2 * (x1**2 - x2**2) / (x1**2 + x2**2)

    # Return zero at origin, computed values elsewhere
    result = backend.where(is_zero, zeros_tensor[:, 0], non_zero_values)
    return result
  
class NonSymHessian(DivFormResidual, DirichletBC, WithSolution):
    """
    PDE with non-symmetric Hessian operator and analytical solution.
    
    This class implements a PDE with a diffusion matrix that creates non-symmetric
    behavior and an analytical solution with singularities at the origin.
    """
    
    def __init__(self, backend=torch):
        """
        Initialize the non-symmetric Hessian PDE.

        Parameters:
            backend: torch or np, optional
                Backend library to use for computations. Defaults to torch.
        """
        # Create partial functions with bound backend
        f_partial = partial(f, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)
        diffusion_matrix_partial = partial(diffusion_matrix, backend=backend)
        boundary_condition_partial = partial(boundary_condition, backend=backend)

        # Initialize parent classes
        DivFormResidual.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        """
        String representation of the NonSymHessian class.

        Returns:
            str
                String representation including backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"NonSymHessian(backend={backend_str})"