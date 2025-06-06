from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
import torch
from functools import partial
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str, frobenius_prod, zeros, ones, all
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

    # Check if x is zero
    zeros_tensor = zeros(x, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    # Compute the value for x != 0
    x1 = x[:, 0]
    x2 = x[:, 1]
    non_zero_values = x1 * x2 * (x1**2 - x2**2) / (x1**2 + x2**2)

    # Combine the results
    result = backend.where(is_zero, zeros_tensor[:, 0], non_zero_values)

    return result

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

    # Check if x is zero
    zeros_tensor = zeros(x, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    # Compute the value for x != 0
    x1 = x[:, 0]
    x2 = x[:, 1]
    grad_x = (x2 * (x1**4 + 4 * x1**2 * x2**2 - x2**4)) / (x1**2 + x2**2)**2
    grad_y = (x1 * (x1**4 - 4 * x1**2 * x2**2 - x2**4)) / (x1**2 + x2**2)**2
    grad_x = backend.where(is_zero, zeros_tensor[:, 0], grad_x)
    grad_y = backend.where(is_zero, zeros_tensor[:, 0], grad_y)
    
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

    # Check if x is zero
    zeros_tensor = zeros(x, backend=backend)
    ones_tensor = ones(x, add_extra_dim=True, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    x1 = x[:, 0]
    x2 = x[:, 1]

    hessian_xx = -4 * x1 * x2**3 * (x1**2 - 3 * x2**2) / (x1**2 + x2**2)**3
    hessian_xy = (x1**6 + 9 * x1**4 * x2**2 - 9 * x2**4 * x1**2 - x2**6) / (x1**2 + x2**2)**3
    hessian_yx = hessian_xy
    hessian_yy = 4 * x1**3 * x2 * (-3 * x1**2 + x2**2) / (x1**2 + x2**2)**3

    # Hessian for x == 0
    batch_size = x.shape[0]
    hessian = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    hessian[:, 0, 0] = backend.where(is_zero, zeros_tensor[:, 0], hessian_xx)
    hessian[:, 0, 1] = backend.where(is_zero, -ones_tensor[:, 0], hessian_xy)
    hessian[:, 1, 0] = backend.where(is_zero, ones_tensor[:, 0], hessian_yx)
    hessian[:, 1, 1] = backend.where(is_zero, zeros_tensor[:, 0], hessian_yy)

    return hessian

# The diffusion matrix
def diffusion_matrix(x, model=None, backend=torch):
    """
    Diffusion matrix

    Parameters:
    x (torch.Tensor): Input tensor with values in the range [0, 1]
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
        b = (x[0, 0]**2 * x[0, 1]**2)**(1/3)
        
        # Create the symbolic diffusion matrix
        diffusion = fn.asarray([[
            [1, b],
            [b, 2],
        ]])

        return diffusion

    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = 1
    diffusion[:, 0, 1] = (x[:, 0]**2 * x[:, 1]**2)**(1/3)
    diffusion[:, 1, 0] = (x[:, 0]**2 * x[:, 1]**2)**(1/3)
    diffusion[:, 1, 1] = 2
    return diffusion

# The source term
def f(x, backend=torch):
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

    return frobenius_prod(diffusion_matrix(x, backend=backend), hessian_u(x, backend=backend))

def boundary_condition(x, backend=torch):
    """
    Boundary condition u(x)

    Parameters:
    x (torch.Tensor): Input tensor of size (batch_size, 2) with values in the range [0, 1]
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of the boundary condition of size (batch_size, 1)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Transform the input into the range [-1, 1]
    x = 2 * x - 1

    # Check if x is zero
    zeros_tensor = zeros(x, backend=backend)
    is_zero = all(x == zeros_tensor, dim=1, backend=backend)

    # Compute the value for x != 0
    x1 = x[:, 0]
    x2 = x[:, 1]
    non_zero_values = x1 * x2 * (x1**2 - x2**2) / (x1**2 + x2**2)

    # Combine the results
    result = backend.where(is_zero, zeros_tensor[:, 0], non_zero_values)

    return result
  
class NonSymHessian(DivFormResidual, DirichletBC, WithSolution):
    """
    SymDiffusion class for the diffusion equation with Dirichlet boundary conditions.
    """
    def __init__(self, backend=torch):
        f_partial = partial(f, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)
        diffusion_matrix_partial = partial(diffusion_matrix, backend=backend)
        boundary_condition_partial = partial(boundary_condition, backend=backend)

        DivFormResidual.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"NonSymHessian(backend={backend_str})"