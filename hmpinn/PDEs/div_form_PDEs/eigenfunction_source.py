"""
Eigenfunction Source PDE in Divergence Form.

This module provides a PDE implementation with eigenfunction source term,
featuring a known analytical solution based on trigonometric functions.

Classes:
    EigenfunctionSourceDF: PDE with eigenfunction source term in divergence form.
"""

import torch
from functools import partial

from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import check_backend, ensure_backend, stack, backend_to_str

# The source term
def f(x, a=1, b=1, amplitude=1, backend=torch):
    """
    Eigenfunction source term for the PDE.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        a: int, optional
            Domain length in x-direction. Defaults to 1.
        b: int, optional
            Domain length in y-direction. Defaults to 1.
        amplitude: int, optional
            Amplitude of the source term. Defaults to 1.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Source term values based on negative sine product.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return -amplitude * backend.sin(backend.pi * x[:, 0] / a) * backend.sin(backend.pi * x[:, 1] / b)

# The analytical solution
def u(x, a=1, b=1, amplitude=1, backend=torch):
    """
    Analytical solution for the eigenfunction source equation.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        a: float, optional
            Domain length in x-direction. Defaults to 1.
        b: float, optional
            Domain length in y-direction. Defaults to 1.
        amplitude: float, optional
            Amplitude of the solution. Defaults to 1.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Analytical solution values at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Compute eigenvalue coefficient
    eigenvalue_coeff = amplitude * (1 / ((backend.pi / a)**2 + (backend.pi / b)**2))
    return eigenvalue_coeff * backend.sin(backend.pi * x[:, 0] / a) * backend.sin(backend.pi * x[:, 1] / b)

# The gradient of the analytical solution
def grad_u(x, a=1, b=1, amplitude=1, backend=torch):
    """
    Gradient of the analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        a: float, optional
            Domain length in x-direction. Defaults to 1.
        b: float, optional
            Domain length in y-direction. Defaults to 1.
        amplitude: float, optional
            Amplitude of the solution. Defaults to 1.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Gradient of the analytical solution at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Compute eigenvalue coefficient
    C = amplitude * (1 / ((backend.pi / a)**2 + (backend.pi / b)**2))
    
    # Compute gradient components
    grad_x = C * backend.cos(backend.pi * x[:, 0] / a) * backend.sin(backend.pi * x[:, 1] / b) * backend.pi / a
    grad_y = C * backend.sin(backend.pi * x[:, 0] / a) * backend.cos(backend.pi * x[:, 1] / b) * backend.pi / b
    
    return stack([grad_x, grad_y], dim=1, backend=backend)

class EigenfunctionSourceDF(DivFormResidual, ConstantBC, WithSolution):
    """
    PDE with eigenfunction source term in divergence form.
    
    This class implements a Poisson-type equation where the source term is
    an eigenfunction, resulting in a trigonometric analytical solution.
    """
    
    def __init__(self, a=1, b=1, amplitude=1, backend=torch):
        """
        Initialize the eigenfunction source PDE.

        Parameters:
            a: float, optional
                Domain length in x-direction. Defaults to 1.
            b: float, optional
                Domain length in y-direction. Defaults to 1.
            amplitude: float, optional
                Amplitude of the solution. Defaults to 1.
            backend: torch or np, optional
                Backend library to use. Defaults to torch.
        """
        self.a = a
        self.b = b
        self.amplitude = amplitude

        # Create partial functions with bound parameters
        f_partial = partial(f, a=a, b=b, amplitude=amplitude, backend=backend)
        u_partial = partial(u, a=a, b=b, amplitude=amplitude, backend=backend)
        grad_u_partial = partial(grad_u, a=a, b=b, amplitude=amplitude, backend=backend)

        # Initialize parent classes
        DivFormResidual.__init__(self, f_partial, backend=backend)
        ConstantBC.__init__(self, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)
    
    def __repr__(self):
        """
        String representation of the EigenfunctionSourceDF class.

        Returns:
            str
                String representation including all parameters and backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"EigenfunctionSourceDF(a={self.a}, b={self.b}, amplitude={self.amplitude}, backend={backend_str})"