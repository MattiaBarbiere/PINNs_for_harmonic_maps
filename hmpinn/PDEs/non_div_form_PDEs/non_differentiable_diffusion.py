"""
Non-Differentiable Diffusion PDE.

This module provides a PDE implementation with a diffusion matrix containing
non-differentiable components (fractional powers) and Gaussian analytical solution.

Classes:
    NonDifferentiableDiffusion: PDE with non-differentiable diffusion matrix.
"""

import torch
from functools import partial

import nutils.function as fn

from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str, norm, frobenius_prod

def u(x, backend=torch):
    """
    Gaussian analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Gaussian function centered at origin with rapid decay.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1
    return backend.exp(-10 * norm(x, backend=backend)**2)

def grad_u(x, backend=torch):
    """
    Gradient of the Gaussian analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Gradient of the Gaussian function.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    # Transform input to range [-1, 1]
    x = 2 * x - 1

    exp_term = backend.exp(-10 * norm(x, backend=backend)**2)
    grad_x = -20 * x[:, 0] * exp_term
    grad_y = -20 * x[:, 1] * exp_term
    return stack([grad_x, grad_y], dim=1, backend=backend)

def hessian_u(x, backend=torch):
    """
    Hessian matrix of the Gaussian analytical solution.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor or fn.Array
            Hessian matrix of the Gaussian function.
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
        
        exp_term = backend.exp(-10 * norm(x, backend=backend)[0]**2)
        
        hessian = fn.asarray([
            [
                -20 * exp_term + 400 * x_0**2 * exp_term, 
                400 * x_0 * x_1 * exp_term
            ],
            [
                400 * x_0 * x_1 * exp_term, 
                -20 * exp_term + 400 * x_1**2 * exp_term
            ],
        ])
        return hessian

    x_0 = x[:, 0]
    x_1 = x[:, 1]
    exp_term = backend.exp(-10 * norm(x, backend=backend)**2)

    # Construct Hessian matrix
    batch_size = x.shape[0]
    hessian = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    hessian[:, 0, 0] = -20 * exp_term + 400 * x_0**2 * exp_term
    hessian[:, 1, 1] = -20 * exp_term + 400 * x_1**2 * exp_term
    hessian[:, 0, 1] = 400 * x_0 * x_1 * exp_term
    hessian[:, 1, 0] = hessian[:, 0, 1]
    
    return hessian

def diffusion_matrix(x, model=None, backend=torch):
    """
    Diffusion matrix with non-differentiable fractional power component.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2) with values in range [0, 1].
        model: torch.nn.Module, optional
            Model instance (unused in this implementation).
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor or fn.Array
            Diffusion matrix with non-differentiable component.
    """
    # Transform input to range [-1, 1]
    x = 2 * x - 1

    if isinstance(x, fn.Array):
        # Symbolic case using nutils.function
        x_0 = x[0, 0]
        x_1 = x[0, 1]
        
        diffusion = fn.asarray([
            [1, 0],
            [0, (x_0**2 * x_1**2)**(1/3) + 1],
        ])
        return diffusion

    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    x_0 = x[:, 0]
    x_1 = x[:, 1]

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = 1
    diffusion[:, 0, 1] = 0
    diffusion[:, 1, 0] = 0
    diffusion[:, 1, 1] = (x_0**2 * x_1**2)**(1/3) + 1
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

    return frobenius_prod(diffusion_matrix(x, backend=backend), hessian_u(x, backend=backend), backend=backend)

def boundary_condition(x, backend=torch):
    """
    Boundary condition matching the Gaussian analytical solution.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2) with values in range [0, 1].
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Boundary condition values from the Gaussian function.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform input to range [-1, 1]
    x = 2 * x - 1
    return backend.exp(-10 * norm(x, backend=backend)**2)
  
class NonDifferentiableDiffusion(NonDivFormResidual, DirichletBC, WithSolution):
    """
    PDE with non-differentiable diffusion matrix and Gaussian solution.
    
    This class implements a PDE where the diffusion matrix contains fractional
    powers that are not differentiable, creating numerical challenges.
    """
    
    def __init__(self, backend=torch):
        """
        Initialize the non-differentiable diffusion PDE.

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
        NonDivFormResidual.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        """
        String representation of the NonDifferentiableDiffusion class.

        Returns:
            str
                String representation including backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"NonDifferentiableDiffusion(backend={backend_str})"