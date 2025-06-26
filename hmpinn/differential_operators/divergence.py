"""
Divergence Operator.

This module provides functionality to compute the divergence of vector fields
using automatic differentiation.

Classes:
    Divergence: Computes the divergence of 2D vector fields.
"""

import torch
from hmpinn.differential_operators.base import BaseDifferentialOperator

class Divergence(BaseDifferentialOperator):
    """
    Computes the divergence of 2D vector fields.
    
    This operator computes the divergence (sum of diagonal elements of the Jacobian)
    of a 2D vector field using automatic differentiation.
    """
    
    def __init__(self):
        """
        Initialize the Divergence operator.
        """
        super().__init__()
    
    def __call__(self, func, x):
        """
        Compute the divergence of a 2D vector field.
        
        Computes the divergence by taking partial derivatives of each component
        with respect to the corresponding coordinate and summing them.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model representing a 2D vector field with output shape (batch_size, 2).
            x: torch.Tensor
                The input coordinates of shape (batch_size, input_dim).

        Returns:
            torch.Tensor
                The divergence of the vector field at the given coordinates.
                
        Raises:
            ValueError: If the output of the model is not of shape (batch_size, 2).
        """
        # Prepare the input for differentiation
        y = self.prepare_input(func, x)

        # Validate that output represents a 2D vector field
        if y.dim() != 2 or y.size(1) != 2:
            raise ValueError("The output of the model must be a tensor of shape (batch_size, 2)")

        # Compute partial derivatives for divergence calculation
        partial_x = torch.autograd.grad(y[:, 0].sum(), x, create_graph=True)[0][:, 0]
        partial_y = torch.autograd.grad(y[:, 1].sum(), x, create_graph=True)[0][:, 1]

        # Return the divergence
        return partial_x + partial_y