"""
Partial Derivative Operator.

This module provides functionality to compute partial derivatives of functions
with respect to specific coordinate dimensions using automatic differentiation.

Classes:
    PartialDerivative: Computes partial derivatives with respect to specified dimensions.
"""

import torch

from hmpinn.differential_operators.base import BaseDifferentialOperator

class PartialDerivative(BaseDifferentialOperator):
    """
    Computes partial derivatives of functions with respect to coordinate dimensions.
    
    This operator uses automatic differentiation to compute the partial derivative
    of a function with respect to a specified coordinate dimension.
    """
    
    def __init__(self):
        """
        Initialize the PartialDerivative operator.
        """
        super().__init__()

    def __call__(self, func, x, dim):
        """
        Compute the partial derivative of a function at given coordinates.
        
        Uses automatic differentiation to compute the partial derivative with
        respect to the specified dimension.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model to compute the derivative of or a tensor of size (batch_size).
            x: torch.Tensor
                The input coordinates of shape (batch_size, input_dim).
            dim: int
                The dimension to compute the derivative with respect to (0 for x, 1 for y).

        Returns:
            torch.Tensor
                The partial derivative of the function with respect to the specified dimension.
        """
        # Prepare the input for differentiation
        y = self.prepare_input(func, x)

        # Compute the gradient using automatic differentiation
        grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # Extract and return the partial derivative for the specified dimension
        return grad[:, dim]

