from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str
from functools import partial
import torch
import nutils.function as fn

# The source term
def f(x, backend=torch):
    """
    Source term f(x) for the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of f(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Extract x and y from the input tensor
    x_val = x[:, 0]
    y_val = x[:, 1]

    # Compute the expression
    result = backend.where(x_val <= 0.5,
                            15 * x_val**4 + 3 * x_val**2 * y_val + 33*x_val + y_val + 3*x_val ** 2 - 6 * y_val ** 2 + 2 * x_val * y_val - 14,
                            -(12 * x_val**3 + 6 * x_val * y_val - 12 * x_val - 2 * y_val**2 + 3 * x_val **3 - 8 * y_val ** 3 + 3 * y_val**2 * x_val - 6))
    return result

# The analytical solution
def u(x, backend=torch):
    """
    Solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return x[:, 0] ** 3 - x[:, 1] ** 2 + x[:, 0] * x[:, 1]


# The boundary value
def boundary_condition(x, backend=torch):
    """
    Parameters:
    x (torch.Tensor): The input tensor of size (batch_size, 2)
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: The boundary condition (default is constant value equal to 0)

    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return (x[:, 0] ** 3 - x[:, 1] ** 2 + x[:, 0] * x[:, 1])

# The diffusion matrix
def diffusion_matrix(x, model=None, backend=torch):
    """
    Diffusion matrix

    Parameters:
    x (torch.Tensor): Input tensor
    model (torch.nn.Module, optional): Model that is trying to solve the pde. This is used only for PDEs that have
                                        a diffusion matrix that depends on the true solution (default is None).
                                        Not used in this case.
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    torch.Tensor: Diffusion matrix of size (batch_size, 2, 2)
    """
    if isinstance(x, fn.Array):
        # Symbolic case using nutils.function
        assert x.shape[-1] == 2, "Input must have shape (..., 2)"

        x0, x1 = x[0, 0], x[0, 1]

        # Build both diffusion matrices
        k1 = fn.asarray([
            [x0**3 + 5,    x0 + x1],
            [x0 + x1,      x1**2 + 7]
        ])

        k2 = fn.asarray([
            [-x0**2 + 2,   -x0 * x1],
            [-x0 * x1,     -x1**3 - 3]
        ])

        # Use function.where with scalar condition
        condition = fn.where(x0 <= 0.5, 1, 0)  # symbolic scalar 1 or 0

        # Use condition to blend matrices
        diffusion = condition * k1 + (1 - condition) * k2

        return diffusion
    
    # Ensure the backend is set correctly
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]

    # Define the first diffusion matrix (k1)
    def k1(x):
        diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
        diffusion[:, 0, 0] = x[:, 0]**3 + 5
        diffusion[:, 0, 1] = x[:, 0] + x[:, 1]
        diffusion[:, 1, 0] = x[:, 1] + x[:, 0]
        diffusion[:, 1, 1] = x[:, 1]**2 + 7
        return diffusion

    # Define the second diffusion matrix (k2)
    def k2(x):
        diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
        diffusion[:, 0, 0] = - x[:, 0]**2 + 2
        diffusion[:, 0, 1] = - x[:, 0] * x[:, 1]
        diffusion[:, 1, 0] = - x[:, 0] * x[:, 1]
        diffusion[:, 1, 1] = - x[:, 1]**3 - 3
        return diffusion

    # Apply the condition
    condition = (x[:, 0] <= 0.5)[:, None, None]
    diffusion = backend.where(condition, k1(x), k2(x))

    return diffusion

def grad_u(x, backend=torch):
    """
    Gradient of the analytical solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    torch.Tensor: Gradient of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    grad_x = 3 * x[:, 0] ** 2 + x[:, 1]
    grad_y = -2 * x[:, 1] + x[:, 0]
    return stack([grad_x, grad_y], dim = 1, backend=backend)
    
class PiecewiseDiffusion(DivFormResidual, DirichletBC, WithSolution):
    """
    Non-divergence form for a PDE with non-constant boundary condition.
    """
    def __init__(self, backend=torch):
        f_partial = partial(f, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)
        diffusion_matrix_partial = partial(diffusion_matrix, backend=backend)
        boundary_condition_partial = partial(boundary_condition, backend=backend)

        DivFormResidual.__init__(self, f_partial, diffusion_matrix_partial , backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial , backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial , backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"PiecewiseDiffusion(backend={backend_str})"