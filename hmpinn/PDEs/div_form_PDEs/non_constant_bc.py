"""
Non-Constant Boundary Condition PDE.

This module provides a PDE implementation with non-constant Dirichlet boundary conditions
in divergence form, featuring a symmetric diffusion matrix and known analytical solution.

Classes:
    NonConstantBC: PDE with spatially varying boundary conditions.
"""

import torch
from functools import partial

import nutils.function as fn

from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str

def f(x, backend=torch):
    """
    Source term function for the non-constant boundary condition equation.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Source term values at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Extract coordinate components
    x_val = x[:, 0]
    y_val = x[:, 1]

    # Compute source term expression
    result = 3 * y_val**4 + 9 * y_val**2 + 4 * x_val**2 * y_val + 18 * y_val + 4 * x_val + 4
    return result

def u(x, backend=torch):
    """
    Analytical solution for the non-constant boundary condition equation.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Analytical solution values at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return x[:,0] **2 + x[:,1] ** 3

def boundary_condition(x, backend=torch):
    """
    Non-constant Dirichlet boundary condition function.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Boundary condition values matching the analytical solution.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return (x[:,0] **2 + x[:,1] ** 3)

def diffusion_matrix(x, model=None, backend=torch):
    """
    Symmetric diffusion matrix function with off-diagonal coupling.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        model: torch.nn.Module, optional
            Model instance (unused in this implementation).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor or fn.Array
            Symmetric diffusion matrix of shape (batch_size, 2, 2) or symbolic array.
    """
    if isinstance(x, fn.Array):
        # Symbolic diffusion matrix construction
        diffusion = fn.asarray([[
            [x[0, 0] + 2, x[0, 0] * x[0, 1]**2],  # First row
            [x[0, 0] * x[0, 1]**2, x[0, 1] + 3],  # Second row
        ]])

        return diffusion

    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = x[:, 0] + 2
    diffusion[:, 0, 1] = x[:, 0] * x[:, 1]**2
    diffusion[:, 1, 0] = x[:, 0] * x[:, 1]**2
    diffusion[:, 1, 1] = x[:, 1] + 3
    return diffusion

def grad_u(x, backend=torch):
    """
    Gradient of the analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Gradient of the analytical solution at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    grad_x = 2 * x[:, 0]
    grad_y = 3 * x[:, 1] ** 2
    return stack([grad_x, grad_y], dim=1, backend=backend)
    
class NonConstantBC(DivFormResidual, DirichletBC, WithSolution):
    """
    Divergence form PDE with non-constant boundary conditions.
    
    This class implements a PDE with symmetric diffusion matrix and spatially
    varying Dirichlet boundary conditions that match the analytical solution.
    """
    
    def __init__(self, backend=torch):
        """
        Initialize the non-constant boundary condition PDE.

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
        String representation of the NonConstantBC class.

        Returns:
            str
                String representation including backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"NonConstantBC(backend={backend_str})"
