"""
Harmonic Maps Utilities.

This module provides utility functions for harmonic map PDEs, including
source terms and diffusion matrix computations specific to harmonic maps.

Functions:
    f_hm: Source term function for harmonic map PDEs.
    hm_diffusion_matrix: Diffusion matrix function for harmonic maps.
"""

import torch

from hmpinn.differential_operators import Jacobian
from hmpinn.PDEs.utils import backend_to_str

def f_hm(x, backend=torch):
    """
    Source term function for harmonic map PDEs.
    
    For harmonic maps, the source term is typically zero, representing
    the harmonicity condition.
    
    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        backend: torch, optional
            Backend library to use. Defaults to torch.
    
    Returns:
        torch.Tensor
            Zero tensor of the same shape as input coordinates.
    """
    return backend.zeros_like(x)

def hm_diffusion_matrix(x, model, backend=torch):
    """
    Diffusion matrix function for harmonic map PDEs.
    
    Computes the metric-dependent diffusion matrix for harmonic maps using
    the Jacobian of the model and constructing the appropriate diffusion tensor.
    
    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        model: torch.nn.Module
            Neural network model mapping from domain to target manifold.
            Must output shape (batch_size, 2).
        backend: torch, optional
            Backend library to use. Currently only torch is supported.
    
    Returns:
        torch.Tensor
            Diffusion matrix of shape (batch_size, 2, 2, 2) for harmonic maps.
            
    Raises:
        ValueError: If backend is not torch.
    """
    # Validate backend support
    if backend_to_str(backend) != "torch":
        raise ValueError("Only torch backend is supported for the diffusion matrix for harmonic maps")
    
    # Compute Jacobian of the model
    jacobian = Jacobian()(model, x)

    # Compute metric tensor g_ij = J^T * J
    g_mat = torch.matmul(jacobian.transpose(1, 2), jacobian)
    g_11 = g_mat[:, 0, 0]
    g_12 = g_mat[:, 0, 1]
    g_21 = g_mat[:, 1, 0]
    g_22 = g_mat[:, 1, 1]

    # Verify symmetry of metric tensor
    assert torch.allclose(g_12, g_21), f"Diffusion matrix is not symmetric: {g_12} != {g_21}"

    # Normalize metric components to avoid numerical issues
    normalize = g_11 + g_22 + 1e-8
    g_11 /= normalize
    g_12 /= normalize
    g_21 /= normalize
    g_22 /= normalize

    # Construct diffusion matrix A = [[g_22, -g_12], [-g_21, g_11]]
    A = torch.stack(
            (torch.stack((g_22, -g_12), dim=1), 
            torch.stack((-g_21, g_11), dim=1)), 
         dim=1)

    # Return diffusion matrix repeated for both components
    return A.unsqueeze(1).repeat(1, 2, 1, 1)