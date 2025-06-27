"""
Inhomogeneous Boundary Condition PDE.

This module provides a PDE implementation with inhomogeneous (non-zero) constant
boundary conditions in divergence form, featuring a known analytical solution.

Classes:
    InhomoBCDF: PDE with inhomogeneous constant boundary conditions.
"""

import torch
from functools import partial

from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str

# The source term
def f(x, backend=torch):
    """
    Source term function for the inhomogeneous boundary condition equation.

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

    return x[:,1] * (1 - x[:,1]) * (-2) + x[:,0] * (1 - x[:,0]) * (-2)

# The analytical solution
def u(x, backend=torch):
    """
    Analytical solution for the inhomogeneous boundary condition equation.

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

    return x[:,1] * (1 - x[:,1]) * x[:,0] * (1 - x[:,0]) + 5

# The gradient of the analytical solution
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

# Constant boundary value
boundary_value = 5
    
class InhomoBCDF(DivFormResidual, ConstantBC, WithSolution):
    """
    Divergence form PDE with inhomogeneous constant boundary conditions.
    
    This class implements a Poisson-type equation with a non-zero constant
    boundary condition, demonstrating boundary condition handling.
    """
    
    def __init__(self, backend=torch):
        """
        Initialize the inhomogeneous boundary condition PDE.

        Parameters:
            backend: torch or np, optional
                Backend library to use for computations. Defaults to torch.
        """
        # Create partial functions with bound backend
        f_partial = partial(f, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)

        # Initialize parent classes
        DivFormResidual.__init__(self, f_partial, backend=backend)
        ConstantBC.__init__(self, boundary_value, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)
    
    def __repr__(self):
        """
        String representation of the InhomoBCDF class.

        Returns:
            str
                String representation including backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"InhomoBCDF(backend={backend_str})"
