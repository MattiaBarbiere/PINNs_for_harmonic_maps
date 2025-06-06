from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
import torch
from functools import partial
from hmpinn.PDEs.utils import check_backend, ensure_backend, stack, ones, backend_to_str

# The functions for Constant Source PDEs
def f(x, const_value=4.0, backend=torch):
    """
    Source term f(x) = const_value

    Parameters:
    x (torch.Tensor): Input tensor of size (batch_size, 2)
    const_value (float): Constant value for the source term
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of f(x) of size (batch_size)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return const_value * ones(x, backend=backend)
    
def boundary_condition(x, const_value=4.0, backend=torch):
    """
    Boundary condition u(x) = x[0]**2 + x[1]**2

    Parameters:
    x (torch.Tensor): Input tensor of size (batch_size, 2)
    const_value (float): Constant value for the boundary condition
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of the boundary condition of size (batch_size, 1)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return const_value / 4.0 * (x[:, 0]**2 + x[:, 1]**2)
    
def u(x, const_value=4.0, backend=torch):
    """
    Analytical solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor of size (batch_size, 2)
    const_value (float): Constant value for the analytical solution
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of u(x) of size (batch_size, 1)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)
    return const_value / 4.0 * (x[:, 0]**2 + x[:, 1]**2)
    
def grad_u(x, const_value=4.0, backend=torch):
    """
    Gradient of the analytical solution u(x).

    Parameters:
    x (torch.Tensor): Input tensor of size (batch_size, 2)
    const_value (float): Constant value for the gradient of the analytical solution
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Gradient of u(x) of size (batch_size, 2)
    """
    return const_value / 2.0 * stack((x[:, 0], x[:, 1]), dim=1, backend=backend)

class ConstantSourceNonDF(NonDivFormResidual, DirichletBC, WithSolution):
    """
    Non-divergence form for a PDE with constant source term.
    """
    def __init__(self, const_value=4.0, backend=torch):
        self.const_value = const_value

        # Use partial to bind const_value to the functions
        f_partial = partial(f, const_value=const_value, backend=backend)
        boundary_condition_partial = partial(boundary_condition, const_value=const_value, backend=backend)
        u_partial = partial(u, const_value=const_value, backend=backend)
        grad_u_partial = partial(grad_u, const_value=const_value, backend=backend)

        # Initialize the parent classes
        NonDivFormResidual.__init__(self, f_partial, backend=backend)
        DirichletBC.__init__(self, boundary_condition_partial, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"ConstantSourceNonDF(const_value={self.const_value}, backend={backend_str})"