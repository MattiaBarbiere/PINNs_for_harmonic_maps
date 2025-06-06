import torch
from hmpinn.differential_operators import Jacobian
from hmpinn.PDEs.utils import backend_to_str

def f_hm(x, backend=torch):
    """
    The source term for the Harmonic map PDE
    Parameters:
        x (torch.Tensor): 2D tensor of shape (batch_size, 2)
    Returns:
        torch.Tensor: 2D tensor of shape (batch_size, 2) with the source term
    """
    return backend.zeros_like(x)

def hm_diffusion_matrix(x, model, backend=torch):
    """
    The diffusion matrix for the Harmonic map PDE
    Parameters:
        x (torch.Tensor): 2D tensor of shape (batch_size, 2)
        model: The model to use for the diffusion matrix. The model must take a 2D tensor of shape (batch_size, 2) and return a 2D tensor of shape (batch_size, 2)
        backend (torch): The backend library to use (default is torch) not that no other backend is supported yet
    Returns:
        torch.Tensor: 2D tensor of shape (batch_size, 2, 2) with the diffusion matrix
    """
    
    if backend_to_str(backend) != "torch":
        raise NotImplementedError("Only torch backend is supported for the diffusion matrix for harmonic maps")
    
    jacobian = Jacobian()(model, x)

    # Compute g_ij
    g_mat = torch.matmul(jacobian.transpose(1, 2), jacobian)
    g_11 = g_mat[:, 0, 0]
    g_12 = g_mat[:, 0, 1]
    g_21 = g_mat[:, 1, 0]
    g_22 = g_mat[:, 1, 1]

    # Check that the matrix is symmetric
    assert torch.allclose(g_12, g_21), f"Diffusion matrix is not symmetric: {g_12} != {g_21}"

    # Compute the normalizing constant and then update the g_ij
    normalize = g_11 + g_22 + 1e-8
    g_11 /= normalize
    g_12 /= normalize
    g_21 /= normalize
    g_22 /= normalize

    # Construct the diffusion matrix
    A = torch.stack(
            (torch.stack((g_22, -g_12), dim=1), 
            torch.stack((-g_21, g_11), dim=1)), 
         dim=1)

    # Return the diffusion matrix stacked twice
    return A.unsqueeze(1).repeat(1, 2, 1, 1)