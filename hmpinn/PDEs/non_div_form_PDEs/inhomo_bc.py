from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
from functools import partial
import torch
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack, backend_to_str

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

    return x[:,1] * (1 - x[:,1]) * (-2) + x[:,0] * (1 - x[:,0]) * (-2)

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

    return x[:,1] * (1 - x[:,1]) * x[:,0] * (1 - x[:,0]) + 5

# The boundary value
boundary_value = 5

# The gradient of the analytical solution
def grad_u(x, backend=torch):
    """
    Gradient of the analytical solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor

    Returns:
    torch.Tensor: Gradient of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    grad_x = (1 - 2 * x[:, 0]) * (1 - x[:, 1]) * x[:, 1]
    grad_y = (1 - 2 * x[:, 1]) * (1 - x[:, 0]) * x[:, 0]
    return stack([grad_x, grad_y], dim = 1, backend=backend)
    
class InhomoBCNonDF(NonDivFormResidual, ConstantBC, WithSolution):
    """
    Non-divergence form for a PDE with non-constant boundary condition.
    """
    def __init__(self, backend=torch):
        """
        An instance of a Poisson equation when the source term is a polynomial inhomogeneous boundary condition.
        """
        f_partial = partial(f, backend=backend)
        u_partial = partial(u, backend=backend)
        grad_u_partial = partial(grad_u, backend=backend)

        NonDivFormResidual.__init__(self, f_partial, backend=backend)
        ConstantBC.__init__(self, boundary_value, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"InhomoBCNonDF(backend={backend_str})"