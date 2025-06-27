"""
Laplacian Operator.

This module provides functionality to compute the Laplacian (with optional diffusion)
of scalar functions using automatic differentiation.

Classes:
    Laplacian: Computes the Laplacian with optional diffusion matrix.
"""

import torch

from hmpinn.differential_operators.base import BaseDifferentialOperator

class Laplacian(BaseDifferentialOperator):
    """
    Computes the Laplacian with optional diffusion matrix.
    
    This operator computes the divergence of the gradient (Laplacian) of a scalar function,
    optionally weighted by a diffusion matrix for anisotropic diffusion problems.
    """
    
    def __init__(self):
        """
        Initialize the Laplacian operator.
        """
        super().__init__()
    
    def __call__(self, func, x, k=None):
        """
        Compute the Laplacian with optional diffusion of a function.
        
        Computes the Laplacian (sum of second derivatives) of a scalar function,
        optionally applying a diffusion matrix for anisotropic problems.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model to compute the second derivatives of or a tensor of size (batch_size).
            x: torch.Tensor
                The input coordinates of shape (batch_size, input_dim).
            k: callable, optional
                The diffusion matrix function that returns a tensor of shape (batch_size, 2, 2).
                If not provided, uses identity matrix (isotropic diffusion).

        Returns:
            torch.Tensor
                The Laplacian with diffusion of the function at the given coordinates.
        """
        # Prepare the input for differentiation
        y = self.prepare_input(func, x)

        # Compute the first derivatives
        grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # Apply diffusion matrix if provided
        if k is not None:
            # Evaluate diffusion matrix at input coordinates
            diffusion_matrix = k(x, model=func).to(x.device)
            # Apply diffusion matrix: k * grad
            grad = torch.bmm(diffusion_matrix, grad.unsqueeze(-1)).squeeze(-1).to(x.device)

        # Compute the divergence of the gradient
        grad_model_x = torch.autograd.grad(grad[:, 0].sum(), x, create_graph=True)[0][:, 0]
        grad_model_y = torch.autograd.grad(grad[:, 1].sum(), x, create_graph=True)[0][:, 1]

        # Return the Laplacian
        return grad_model_x + grad_model_y