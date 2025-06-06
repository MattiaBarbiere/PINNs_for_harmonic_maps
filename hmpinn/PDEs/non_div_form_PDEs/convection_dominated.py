from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
import torch
from functools import partial
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str, norm, frobenius_prod
import nutils.function as fn


# The analytical solution
def u(x, backend=torch):
    """
    Solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor with values in the range [0, 1]
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform the input into the range [-1, 1]
    x = 2 * x - 1

    return backend.sin(backend.pi * x[:, 0]) * backend.sin(backend.pi * x[:, 1])

def grad_u(x, backend=torch):
    """
    Gradient of the analytical solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor with values in the range [0, 1]
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    torch.Tensor: Gradient of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    # Transform the input into the range [-1, 1]
    x = 2 * x - 1

    grad_x = backend.pi * backend.cos(backend.pi * x[:, 0]) * backend.sin(backend.pi * x[:, 1])
    grad_y = backend.pi * backend.sin(backend.pi * x[:, 0]) * backend.cos(backend.pi * x[:, 1])
    return stack([grad_x, grad_y], dim = 1, backend=backend)

def hessian_u(x, backend=torch):
    """
    Hessian of the analytical solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor with values in the range [0, 1]
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    torch.Tensor: Hessian of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform the input into the range [-1, 1]
    x = 2 * x - 1

    if isinstance(x, fn.Array):

        # Symbolic case using nutils.function
        x_0 = x[0, 0]
        x_1 = x[0, 1]
        
        # Create the symbolic Hessian matrix
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

# The diffusion matrix
def diffusion_matrix(x, K=1, model=None, backend=torch):
    """
    Diffusion matrix

    Parameters:
    x (torch.Tensor): Input tensor with values in the range [0, 1]
    K (float): Multiplicative constant for the diffusion matrix (default is 1)
    model (torch.nn.Module, optional): Model that is trying to solve the pde. This is used only for PDEs that have
                                        a diffusion matrix that depends on the true solution (default is None).
                                        Not used in this case.
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    torch.Tensor: Diffusion matrix of size (batch_size, 2, 2)
    """
    # Transform the input into the range [-1, 1]
    x = 2 * x - 1

    if isinstance(x, fn.Array):
        # Symbolic case using nutils.function
        
        # Create the symbolic diffusion matrix
        diffusion = fn.asarray([[
            [1, 0],
            [0, backend.arctan(K * (norm(x, backend=backend)[0]**2 - 1)) + 2],
        ]])

        return diffusion

    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = 1
    diffusion[:, 0, 1] = 0
    diffusion[:, 1, 0] = 0
    diffusion[:, 1, 1] = backend.arctan(K * (norm(x, backend=backend)**2 - 1)) + 2
    return diffusion

# The source term
def f(x, K=1, backend=torch):
    """
    Source term f(x) for the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor with values in the range [0, 1]
    K (float): Multiplicative constant for the diffusion matrix (default is 1)
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of f(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return frobenius_prod(diffusion_matrix(x, K=K, backend=backend), hessian_u(x, backend=backend), backend=backend)

def boundary_condition(x, backend=torch):
    """
    Boundary condition u(x)

    Parameters:
    x (torch.Tensor): Input tensor of size (batch_size, 2) with values in the range [0, 1]
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of the boundary condition of size (batch_size, 1)
    """
    # Ensure the backend is set correctly
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform the input into the range [-1, 1]
    x = 2 * x - 1

    return backend.sin(backend.pi * x[:, 0]) * backend.sin(backend.pi * x[:, 1])
  
class ConvectionDominatedNonDF(NonDivFormResidual, DirichletBC, WithSolution):
    """
    SymDiffusion class for the diffusion equation with Dirichlet boundary conditions.
    """
    def __init__(self, K=1, backend=torch):
        self.K = K

        f_partial = partial(f, K=K, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)
        diffusion_matrix_partial = partial(diffusion_matrix, K=K, backend=backend)
        boundary_condition_partial = partial(boundary_condition, backend=backend)

        NonDivFormResidual.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"ConvectionDominatedNonDF(K={self.K}, backend={backend_str})"