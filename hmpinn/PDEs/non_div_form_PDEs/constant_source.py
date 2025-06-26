"""
Constant Source PDE (Non-Divergence Form).

This module provides a non-divergence form PDE implementation with constant source term,
featuring a quadratic analytical solution and Dirichlet boundary conditions.

Classes:
    ConstantSourceNonDF: PDE with constant source term in non-divergence form.
"""

import torch
from functools import partial

from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import check_backend, ensure_backend, stack, ones, backend_to_str

def f(x, const_value=4.0, backend=torch):
    """
    Constant source term function.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        const_value: float, optional
            Constant value for the source term. Defaults to 4.0.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Constant source term values at all coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return const_value * ones(x, backend=backend)
    
def boundary_condition(x, const_value=4.0, backend=torch):
    """
    Quadratic Dirichlet boundary condition function.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2).
        const_value: float, optional
            Constant value affecting boundary condition. Defaults to 4.0.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Boundary condition values based on quadratic function.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return const_value / 4.0 * (x[:, 0]**2 + x[:, 1]**2)
    
def u(x, const_value=4.0, backend=torch):
    """
    Analytical solution for the constant source equation.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        const_value: float, optional
            Constant value for the analytical solution. Defaults to 4.0.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Analytical solution values (quadratic function).
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    return const_value / 4.0 * (x[:, 0]**2 + x[:, 1]**2)
    
def grad_u(x, const_value=4.0, backend=torch):
    """
    Gradient of the analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        const_value: float, optional
            Constant value affecting the gradient. Defaults to 4.0.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Gradient of the analytical solution (linear function).
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return const_value / 2.0 * stack((x[:, 0], x[:, 1]), dim=1, backend=backend)

class ConstantSourceNonDF(NonDivFormResidual, DirichletBC, WithSolution):
    """
    Non-divergence form PDE with constant source term.
    
    This class implements a Poisson equation with constant source term in non-divergence form,
    resulting in a quadratic analytical solution with Dirichlet boundary conditions.
    """
    
    def __init__(self, const_value=4.0, backend=torch):
        """
        Initialize the constant source PDE in non-divergence form.

        Parameters:
            const_value: float, optional
                Constant value for source term and solution scaling. Defaults to 4.0.
            backend: torch or np, optional
                Backend library to use. Defaults to torch.
        """
        self.const_value = const_value

        # Create partial functions with bound constant value
        f_partial = partial(f, const_value=const_value, backend=backend)
        boundary_condition_partial = partial(boundary_condition, const_value=const_value, backend=backend)
        u_partial = partial(u, const_value=const_value, backend=backend)
        grad_u_partial = partial(grad_u, const_value=const_value, backend=backend)

        # Initialize parent classes
        NonDivFormResidual.__init__(self, f_partial, backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        """
        String representation of the ConstantSourceNonDF class.

        Returns:
            str
                String representation including constant value and backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"ConstantSourceNonDF(const_value={self.const_value}, backend={backend_str})"