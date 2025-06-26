"""
Symmetric Diffusion PDE.

This module provides a symmetric diffusion PDE implementation with known analytical solution.
The PDE is in divergence form with a symmetric diffusion matrix.

Classes:
    SymDiffusion: Symmetric diffusion PDE with analytical solution.
"""

import torch
from functools import partial

import nutils.function as fn

from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str

def f(x, backend=torch):
    """
    Source term function for the symmetric diffusion equation.

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
    result = (y_val - y_val**2) * (1 - 4 * x_val) + \
            (2 * x_val - 3 * x_val**2) * (y_val - 2 * y_val**2) + \
            (2 * y_val - 3 * y_val**2) * (x_val - 2 * x_val**2) + \
            (x_val - x_val**2) * (1 - 4 * y_val)
    return result

def u(x, backend=torch):
    """
    Analytical solution for the symmetric diffusion equation.

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

    return x[:,1] * (1 - x[:,1]) * x[:,0] * (1 - x[:,0])

def diffusion_matrix(x, model=None, backend=torch):
    """
    Symmetric diffusion matrix function.

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
        # Symbolic case using nutils.function
        diffusion = fn.asarray([[
            [x[0, 0], x[0, 0] * x[0, 1]],
            [x[0, 1] * x[0, 0], x[0, 1]],
        ]])
        return diffusion

    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = x[:, 0]
    diffusion[:, 0, 1] = x[:, 0] * x[:, 1]
    diffusion[:, 1, 0] = x[:, 1] * x[:, 0]
    diffusion[:, 1, 1] = x[:, 1]
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

    grad_x = (1 - 2 * x[:, 0]) * (1 - x[:, 1]) * x[:, 1]
    grad_y = (1 - 2 * x[:, 1]) * (1 - x[:, 0]) * x[:, 0]
    return stack([grad_x, grad_y], dim=1, backend=backend)
    
class SymDiffusion(DivFormResidual, ConstantBC, WithSolution):
    """
    Symmetric diffusion PDE with constant boundary conditions.
    
    This class implements a PDE with symmetric diffusion matrix in divergence form,
    featuring a known analytical solution and constant boundary conditions.
    """
    
    def __init__(self, backend=torch):
        """
        Initialize the symmetric diffusion PDE.

        Parameters:
            backend: torch or np, optional
                Backend library to use for computations. Defaults to torch.
        """
        # Create partial functions with bound backend
        f_partial = partial(f, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)
        diffusion_matrix_partial = partial(diffusion_matrix, backend=backend)

        # Initialize parent classes
        DivFormResidual.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
        ConstantBC.__init__(self, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        """
        String representation of the SymDiffusion class.

        Returns:
            str
                String representation including backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"SymDiffusion(backend={backend_str})"