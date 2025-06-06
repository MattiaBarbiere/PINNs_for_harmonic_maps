from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
import torch
from functools import partial
import math
from hmpinn.PDEs.utils import check_backend, ensure_backend, backend_to_str

# The source term
def f(x, mu_x=0, mu_y=0, std_x=1, std_y=1, backend=torch):
    """
    Parameters:
    x (torch.Tensor): The input tensor of size (batch_size, 2)
    mu_x (int): The x coordinate of the central point of the gaussian bump (default 0)
    mu_y (int): The y coordinate of the central point of the gaussian bump (default 0)
    std_x (int): The standard deviation of the gaussian bump in x direction(default 1)
    std_y (int): The standard deviation of the gaussian bump in y direction(default 1)
    backend (torch or np): The backend library to use (default is torch)

    Returns:
    torch.Tensor: The source term evaluated at x with size (batch_size, 1)

    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    return backend.exp(-((x[:, 0] - mu_x) ** 2) / (2 * std_x ** 2) - ((x[:, 1] - mu_y) ** 2) / (2 * std_y ** 2)) / (std_x * std_y * math.sqrt(2 * math.pi))

class GuassianBumpNonDF(NonDivFormResidual, ConstantBC, WithoutSolution):
    def __init__(self, mu_x=0, mu_y=0, std_x=1, std_y=1, backend=torch):
        """
        An instance of a Poisson equation when the source term is a guassian bump.

        Parameters:
        mu_x (int): The x coordinate of the central point of the gaussian bump (default 0)
        mu_y (int): The y coordinate of the central point of the gaussian bump (default 0)
        std_x (int): The standard deviation of the gaussian bump in x direction(default 1)
        std_y (int): The standard deviation of the gaussian bump in y direction(default 1)
        backend (torch): The backend library to use (default is torch)
        """
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.std_x = std_x
        self.std_y = std_y

        # Use partial to bind parameters a and b
        f_partial = partial(f, mu_x=mu_x, mu_y=mu_y, std_x=std_x, std_y=std_y, backend=backend)

        NonDivFormResidual.__init__(self, f_partial, backend=backend)
        ConstantBC.__init__(self, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"GuassianBumpNonDF(mu_x={self.mu_x}, mu_y={self.mu_y}, std_x={self.std_x}, std_y={self.std_y}, backend={backend_str})"