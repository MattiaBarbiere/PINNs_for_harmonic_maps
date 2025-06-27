"""
Jacobian Operator.

This module provides functionality to compute the Jacobian matrix of vector-valued
functions using automatic differentiation.

Classes:
    Jacobian: Computes the Jacobian matrix of vector-valued functions.
"""

import torch

from hmpinn.differential_operators.base import BaseDifferentialOperator

class Jacobian(BaseDifferentialOperator):
    """
    Computes the Jacobian matrix of vector-valued functions.
    
    This operator uses automatic differentiation to compute the Jacobian matrix
    containing all first-order partial derivatives of a vector-valued function.
    """
    
    def __init__(self):
        """
        Initialize the Jacobian operator.
        """
        super().__init__()
    
    def __call__(self, func, x):
        """
        Compute the Jacobian matrix of a vector-valued function.
        
        Uses automatic differentiation to compute the Jacobian matrix containing
        partial derivatives of each output component with respect to each input dimension.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model that returns an output of size (batch_size, 2).
            x: torch.Tensor
                The input coordinates of shape (batch_size, input_dim).

        Returns:
            torch.Tensor
                The Jacobian matrix of shape (batch_size, output_dim, input_dim).
                
        Raises:
            AssertionError: If the output of the model is not of size (batch_size, 2).
        """
        # Prepare the input for differentiation
        y = self.prepare_input(func, x)

        # Validate output dimensions
        assert y.shape[1] == 2, f"Output of the model must be of size (batch_size, 2), but got {y.shape}"
            
        # Initialize Jacobian tensor
        jacobian = torch.zeros(x.shape[0], 2, 2, dtype=x.dtype, device=x.device)

        # Compute Jacobian by differentiating each output component
        for i in range(2):
            # Create gradient output mask for current component
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            
            # Compute gradients for current output component
            grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            jacobian[:, i, :] = grads

        return jacobian