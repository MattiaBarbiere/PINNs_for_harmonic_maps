"""
Base Differential Operator.

This module provides the base class for all differential operators in the hmpinn library.
It handles input preparation and device management for differential computations.

Classes:
    BaseDifferentialOperator: Base class for all differential operators.
"""

import torch

class BaseDifferentialOperator:
    """
    Base class for all differential operators.
    
    This class provides common functionality for differential operators including
    input preparation and device management for automatic differentiation.
    """
    
    def __init__(self):
        """
        Initialize the BaseDifferentialOperator.
        """
        pass

    def prepare_input(self, func, x):
        """
        Prepare the input for differential operator computation.
        
        Handles device placement, gradient requirements, and input validation
        for both callable functions and tensor inputs.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model to apply operator to or a tensor of size (batch_size).
            x: torch.Tensor
                The input coordinates of shape (batch_size, input_dim) over which 
                the derivative is computed.

        Returns:
            torch.Tensor
                The prepared output tensor ready for differentiation.
                
        Raises:
            ValueError: If func is neither a callable nor a tensor with matching batch size.
        """
        # Ensure the function is on the same device as x
        if isinstance(func, torch.nn.Module):
            func.to(x.device)
            
        # Ensure x requires gradients for automatic differentiation
        if not x.requires_grad:
            x.requires_grad = True
        
        # Process the input based on its type
        if callable(func):
            # Evaluate the function at input coordinates
            y = func(x)
        elif isinstance(func, torch.Tensor) and func.shape[0] == x.shape[0]:
            # Use the tensor directly if batch sizes match
            y = func
        else:
            raise ValueError("func must be a torch.nn.Module or a torch.Tensor of size (batch_size)")
        
        # Ensure output is on the same device as input
        return y.to(x.device)