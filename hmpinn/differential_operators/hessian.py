"""
Hessian Operator.

This module provides functionality to compute the Hessian matrix (second-order derivatives)
of scalar and vector-valued functions using automatic differentiation.

Classes:
    Hessian: Computes the Hessian matrix of functions.
"""

import torch

from hmpinn.differential_operators.base import BaseDifferentialOperator

class Hessian(BaseDifferentialOperator):
    """
    Computes the Hessian matrix of scalar and vector-valued functions.
    
    This operator uses automatic differentiation to compute the Hessian matrix
    containing all second-order partial derivatives of a function.
    """
    
    def __init__(self):
        """
        Initialize the Hessian operator.
        """
        super().__init__()
    
    def __call__(self, func, x):
        """
        Compute the Hessian matrix of a function at given coordinates.
        
        Uses automatic differentiation to compute the Hessian matrix containing
        all second-order partial derivatives. Handles both scalar and vector-valued functions.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model to compute the second derivatives of.
            x: torch.Tensor
                The input coordinates of shape (batch_size, input_dim).

        Returns:
            torch.Tensor
                The Hessian matrix. For scalar outputs: shape (batch_size, input_dim, input_dim).
                For vector outputs: shape (batch_size, output_dim, input_dim, input_dim).
        """
        # Prepare the input for differentiation
        y = self.prepare_input(func, x)
        
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        # Handle scalar outputs by adding dimension
        if y.ndim == 1:  # Shape [batch_size]
            y = y.unsqueeze(1)  # Convert to [batch_size, 1]
        
        output_dim = y.shape[1]
        hessians = []
        
        # Compute Hessian for each output dimension
        for i in range(output_dim):
            # Create gradient output mask for current output dimension
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            
            # Compute first derivatives for current output dimension
            first_grads = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute second derivatives for each input dimension
            hess_rows = []
            for j in range(input_dim):
                # Compute second derivatives with respect to all inputs
                second_grads = torch.autograd.grad(
                    outputs=first_grads[:, j],
                    inputs=x,
                    grad_outputs=torch.ones(batch_size, device=x.device),
                    create_graph=True,
                    retain_graph=True 
                )[0]
                hess_rows.append(second_grads)
            
            # Stack rows to form Hessian for current output dimension
            hess_i = torch.stack(hess_rows, dim=1)
            hessians.append(hess_i)
        
        # Stack all output dimensions
        result = torch.stack(hessians, dim=1)
        
        # For scalar outputs, remove output dimension for backward compatibility
        if output_dim == 1:
            result = result.squeeze(1)
        
        return result