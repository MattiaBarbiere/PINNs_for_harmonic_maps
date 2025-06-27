"""
Gradient Operator.

This module provides functionality to compute the gradient (first-order derivatives)
of scalar functions using automatic differentiation.

Classes:
    Gradient: Computes the gradient of scalar functions.
"""

import torch

from hmpinn.differential_operators.base import BaseDifferentialOperator

class Gradient(BaseDifferentialOperator):
    """
    Computes the gradient of scalar functions.
    
    This operator uses automatic differentiation to compute the full gradient
    vector of a scalar function with respect to all input coordinates.
    """
    
    def __init__(self):
        """
        Initialize the Gradient operator.
        """
        super().__init__()
    
    def __call__(self, func, x):
        """
        Compute the gradient of a scalar function at given coordinates.
        
        Uses automatic differentiation to compute the gradient vector containing
        partial derivatives with respect to all input dimensions.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model to compute the derivative of or a tensor of size (batch_size).
            x: torch.Tensor
                The input coordinates of shape (batch_size, input_dim).

        Returns:
            torch.Tensor
                The gradient vector of shape (batch_size, input_dim).
        """
        # Prepare the input for differentiation
        y = self.prepare_input(func, x)

        # Compute the gradient using automatic differentiation
        grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # Return the full gradient vector
        return grad