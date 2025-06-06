from hmpinn.PDEs.PDE_factory import construct_PDE_class
from test_PDEs import test_outputs, test_attributes
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack
import torch
import numpy as np

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

    return (x[:, 0] ** 3 - x[:, 1] ** 2 + x[:, 0] * x[:, 1]).reshape(-1, 1)

# The diffusion matrix
def diffusion_matrix(x, backend=torch):
    """
    Diffusion matrix

    Parameters:
    x (torch.Tensor): Input tensor
    backend (torch or np): Backend library to use (default is torch)

    Returns:
    torch.Tensor: Diffusion matrix of size (batch_size, 2, 2)
    """
    # Ensure the backend is set correctlys
    if check_backend(backend):
        x = ensure_backend(x, backend)

    batch_size = x.shape[0]

    # Define the first diffusion matrix
    def k1(x):
        diffusion = backend.empty((batch_size, 2, 2))
        diffusion[:, 0, 0] = x[:, 0] ** 3 + 5
        diffusion[:, 0, 1] = x[:, 0] + x[:, 1]
        diffusion[:, 1, 0] = x[:, 1] + x[:, 0]
        diffusion[:, 1, 1] = x[:, 1] ** 2 + 7
        return diffusion
    
    # Define the second diffusion matrix
    def k2(x):
        diffusion = backend.empty((batch_size, 2, 2))
        diffusion[:, 0, 0] = - x[:, 0] ** 2 + 2
        diffusion[:, 0, 1] = - x[:, 0] * x[:, 1]
        diffusion[:, 1, 0] = - x[:, 0] * x[:, 1]
        diffusion[:, 1, 1] = - x[:, 1]**3 - 3
        return diffusion
    
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


def test_PDE_factory():

    # PDE will all the functions
    PDE = construct_PDE_class(True, f, diffusion_matrix, boundary_condition, u, grad_u)
    test_outputs(PDE(backend=torch))
    test_attributes(PDE(backend=torch))

    # PDE will all the functions
    PDE = construct_PDE_class(False, f, diffusion_matrix, boundary_condition, u, grad_u)
    test_outputs(PDE(backend=torch))
    test_attributes(PDE(backend=torch))

    # PDE will all the functions
    PDE = construct_PDE_class(True, f, diffusion_matrix, boundary_condition, u, grad_u)
    test_outputs(PDE(backend=np))
    test_attributes(PDE(backend=np))

    # PDE will all the functions
    PDE = construct_PDE_class(False, f, diffusion_matrix, boundary_condition, u, grad_u)
    test_outputs(PDE(backend=np))
    test_attributes(PDE(backend=np))

    # PDE will all the functions
    PDE = construct_PDE_class(True, f)
    test_outputs(PDE(backend=torch))
    test_attributes(PDE(backend=torch))

    # PDE will all the functions
    PDE = construct_PDE_class(False, f)
    test_outputs(PDE(backend=torch))
    test_attributes(PDE(backend=torch))

    # PDE will all the functions
    PDE = construct_PDE_class(True, f)
    test_outputs(PDE(backend=np))
    test_attributes(PDE(backend=np))

    # PDE will all the functions
    PDE = construct_PDE_class(False, f)
    test_outputs(PDE(backend=np))
    test_attributes(PDE(backend=np))




if __name__ == "__main__":
    test_PDE_factory()