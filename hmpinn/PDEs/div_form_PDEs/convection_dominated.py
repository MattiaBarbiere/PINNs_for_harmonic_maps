"""
Convection Dominated PDE in Divergence Form.

This module provides a convection-dominated PDE implementation with known analytical solution.
The PDE features a diffusion matrix that varies based on the arctangent function.

Classes:
    ConvectionDominatedDF: Convection-dominated PDE with analytical solution.
"""

import torch
from functools import partial

import nutils.function as fn

from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str, norm, frobenius_prod

def u(x, backend=torch):
    """
    Analytical solution for the convection-dominated equation.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Analytical solution values at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1

    return backend.sin(backend.pi * x[:, 0]) * backend.sin(backend.pi * x[:, 1])

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
            Gradient of the analytical solution at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    # Transform input to range [-1, 1]
    x = 2 * x - 1

    grad_x = backend.pi * backend.cos(backend.pi * x[:, 0]) * backend.sin(backend.pi * x[:, 1])
    grad_y = backend.pi * backend.sin(backend.pi * x[:, 0]) * backend.cos(backend.pi * x[:, 1])
    return stack([grad_x, grad_y], dim=1, backend=backend)

def hessian_u(x, backend=torch):
    """
    Hessian of the analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor or fn.Array
            Hessian matrix at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1

    if isinstance(x, fn.Array):
        # Symbolic case using nutils.function
        x_0 = x[0, 0]
        x_1 = x[0, 1]
        
        # Create symbolic Hessian matrix
        hessian = fn.asarray([[
            [
                -backend.pi**2 * backend.sin(backend.pi * x_0) * backend.sin(backend.pi * x_1), 
                backend.pi**2 * backend.cos(backend.pi * x_0) * backend.cos(backend.pi * x_1)
            ],
            [
                backend.pi**2 * backend.cos(backend.pi * x_0) * backend.cos(backend.pi * x_1), 
                -backend.pi**2 * backend.sin(backend.pi * x_0) * backend.sin(backend.pi * x_1)
            ]
        ]])
        return hessian

    x_0 = x[:, 0]
    x_1 = x[:, 1]

    batch_size = x.shape[0]
    hessian = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    hessian[:, 0, 0] = -backend.pi**2 * backend.sin(backend.pi * x_0) * backend.sin(backend.pi * x_1)
    hessian[:, 0, 1] = backend.pi**2 * backend.cos(backend.pi * x_0) * backend.cos(backend.pi * x_1)
    hessian[:, 1, 0] = hessian[:, 0, 1]
    hessian[:, 1, 1] = -backend.pi**2 * backend.sin(backend.pi * x_0) * backend.sin(backend.pi * x_1)
    
    return hessian

def diffusion_matrix(x, K=1, model=None, backend=torch):
    """
    Convection-dominated diffusion matrix function.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        K: float, optional
            Multiplicative constant for the diffusion matrix. Defaults to 1.
        model: torch.nn.Module, optional
            Model instance (unused in this implementation).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor or fn.Array
            Diffusion matrix featuring arctangent nonlinearity.
    """
    # Transform input to range [-1, 1]
    x = 2 * x - 1

    if isinstance(x, fn.Array):
        # Symbolic case using nutils.function
        diffusion = fn.asarray([[
            [1, 0],
            [0, backend.arctan(K * (norm(x, backend=backend)[0]**2 - 1)) + 2],
        ]])

        return diffusion

    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = 1
    diffusion[:, 0, 1] = 0
    diffusion[:, 1, 0] = 0
    diffusion[:, 1, 1] = backend.arctan(K * (norm(x, backend=backend)**2 - 1)) + 2
    return diffusion

def f(x, K=1, backend=torch):
    """
    Source term function for the convection-dominated equation.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        K: float, optional
            Multiplicative constant for the diffusion matrix. Defaults to 1.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Source term values computed as Frobenius product of diffusion and Hessian.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return frobenius_prod(diffusion_matrix(x, K=K, backend=backend), hessian_u(x, backend=backend), backend=backend)

def boundary_condition(x, backend=torch):
    """
    Dirichlet boundary condition function.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Boundary condition values matching the analytical solution.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1

    return backend.sin(backend.pi * x[:, 0]) * backend.sin(backend.pi * x[:, 1])
  
class ConvectionDominatedDF(DivFormResidual, DirichletBC, WithSolution):
    """
    Convection-dominated PDE in divergence form with analytical solution.
    
    This class implements a PDE with arctangent-based diffusion matrix that creates
    convection-dominated behavior in certain regions of the domain.
    """
    
    def __init__(self, K=1, backend=torch):
        """
        Initialize the convection-dominated PDE.

        Parameters:
            K: float, optional
                Multiplicative constant for the diffusion matrix. Defaults to 1.
            backend: torch or np, optional
                Backend library to use for computations. Defaults to torch.
        """
        self.K = K

        # Create partial functions with bound parameters
        f_partial = partial(f, K=K, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)
        diffusion_matrix_partial = partial(diffusion_matrix, K=K, backend=backend)
        boundary_condition_partial = partial(boundary_condition, backend=backend)

        # Initialize parent classes
        DivFormResidual.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        """
        String representation of the ConvectionDominatedDF class.

        Returns:
            str
                String representation including K parameter and backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"ConvectionDominatedDF(K={self.K}, backend={backend_str})"