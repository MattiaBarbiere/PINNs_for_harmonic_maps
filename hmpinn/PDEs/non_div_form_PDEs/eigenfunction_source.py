from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.solutions.with_solution import WithSolution
import torch
from functools import partial
from hmpinn.PDEs.utils import check_backend, ensure_backend, stack, backend_to_str

# The source term
def f(x, a=1, b=1, backend=torch):
    """
    Parameters:
    x (torch.Tensor): The input tensor of size (batch_size, 2)
    a (int): The domain length in the x-direction (default 1)
    b (int): The domain length in the y-direction (default 1)
    backend (torch or np): The backend library to use (default is torch)

    Returns:
    torch.Tensor: The source term evaluated at x with size (batch_size, 1)

    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return -backend.sin(backend.pi * x[:, 0] / a) * backend.sin(backend.pi * x[:, 1] / b)

# The analytical solution
def u(x, a=1, b=1, backend=torch):
    """
    Solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor
    a (float): Domain length in x-direction (default = 1)
    b (float): Domain length in y-direction (default = 1)
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    float: Value of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return (1 / ((backend.pi / a)**2 + (backend.pi / b)**2)) * backend.sin(backend.pi * x[:, 0] / a) * backend.sin(backend.pi * x[:, 1] / b)

# The gradient of the analytical solution
def grad_u(x, a=1, b=1, backend=torch):
    """
    Gradient of the analytical solution u(x) to the Poisson equation.

    Parameters:
    x (torch.Tensor): Input tensor
    a (float): Domain length in x-direction (default = 1)
    b (float): Domain length in y-direction (default = 1)
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    torch.Tensor: Gradient of u(x)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    C = (1 / ((backend.pi / a)**2 + (backend.pi / b)**2))
    return C * stack([backend.cos(backend.pi * x[:, 0] / a) * backend.sin(backend.pi * x[:, 1] / b) * backend.pi / a,
                            backend.sin(backend.pi * x[:, 0] / a) * backend.cos(backend.pi * x[:, 1] / b) * backend.pi / b], dim=1, backend=backend)

class EigenfunctionSourceNonDF(NonDivFormResidual, ConstantBC, WithSolution):
    def __init__(self, a=1, b=1, backend=torch):
        """
        An instance of a Poisson equation when the source term is an eigenfunction in non divergence form

        Parameters:
        a (float): Domain length in x-direction (default = 1)
        b (float): Domain length in y-direction (default = 1)
        backend (torch): The backend library to use (default is torch)
        """
        self.a = a
        self.b = b

        # Use partial to bind parameters a and b
        f_partial = partial(f, a=a, b=b, backend=backend)
        u_partial = partial(u, a=a, b=b, backend=backend)
        grad_u_partial = partial(grad_u, a=a, b=b, backend=backend)

        NonDivFormResidual.__init__(self, f_partial, backend=backend)
        ConstantBC.__init__(self, backend=backend)
        WithSolution.__init__(self, u_partial, grad_u_partial, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"EigenfunctionSourceNonDF(a={self.a}, b={self.b}, backend={backend_str})"