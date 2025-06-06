from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
import torch
from functools import partial
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str
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
    result = 3 * y_val**4 + 9 * y_val**2 + 4 * x_val**2 * y_val + 18 * y_val + 4 * x_val + 4
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

    return x[:,0] **2 + x[:,1] ** 3


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

    return (x[:,0] **2 + x[:,1] ** 3)

# The diffusion matrix
def diffusion_matrix(x, model=None, backend=torch):
    """
    Diffusion matrix

    Parameters:
    x (torch.Tensor): Input tensor
    model (torch.nn.Module, optional): Model that is trying to solve the pde. This is used only for PDEs that have
                                        a diffusion matrix that depends on the true solution (default is None).
                                        Not used in this case.

    Returns:
    torch.Tensor: Diffusion matrix of size (batch_size, 2, 2)
    """
    if isinstance(x, fn.Array):
        # Symbolic diffusion matrix construction
        diffusion = fn.asarray([[
            [x[0, 0] + 2, x[0, 0] * x[0, 1]**2],  # First row
            [x[0, 0] * x[0, 1]**2, x[0, 1] + 3],  # Second row
        ]])

        return diffusion


    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]
    diffusion = backend.empty((batch_size, 2, 2), device=x.device, dtype=x.dtype)
    diffusion[:, 0, 0] = x[:, 0] + 2
    diffusion[:, 0, 1] = x[:, 0] * x[:, 1]**2
    diffusion[:, 1, 0] = x[:, 0] * x[:, 1]**2
    diffusion[:, 1, 1] = x[:, 1] + 3
    return diffusion

def grad_u(x , backend=torch):
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

    grad_x = 2 * x[:, 0]
    grad_y = 3 * x[:, 1] ** 2
    return stack([grad_x, grad_y], dim = 1, backend=backend)
    
class NonConstantBC(DivFormResidual, DirichletBC, WithSolution):
    """
    Non-divergence form for a PDE with non-constant boundary condition.
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
        return f"NonConstantBC(backend={backend_str})"
