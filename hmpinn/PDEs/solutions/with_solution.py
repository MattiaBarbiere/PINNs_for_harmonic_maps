"""
Solution Class with Known Analytical Solution.

This module provides the solution implementation for PDEs with known
analytical solutions and their gradients.

Classes:
    WithSolution: Solution class for PDEs with known analytical solutions.
"""

import torch

from hmpinn.PDEs.solutions.base import BaseSolution
from hmpinn.PDEs.utils import relative_error
from hmpinn.differential_operators.gradient import Gradient

class WithSolution(BaseSolution):
    """
    Solution class for PDEs with known analytical solutions.
    
    This class handles PDEs where both the analytical solution and its
    gradient are known, enabling error computation and validation.
    """
    
    def __init__(self, u, grad_u, backend=torch):
        """
        Initialize with known analytical solution and gradient.

        Parameters:
            u: callable
                The analytical solution function.
            grad_u: callable
                The gradient of the analytical solution function.
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
                
        Raises:
            ValueError: If u or grad_u are not callable.
        """
        super().__init__(backend=backend)
        
        # Validate input functions
        if not callable(u):
            raise ValueError("u must be a callable function.")
        if not callable(grad_u):
            raise ValueError("grad_u must be a callable function.")
            
        self.u_func = u
        self.grad_u_func = grad_u

    @property
    def has_solution(self):
        """
        Check if an analytical solution exists.

        Returns:
            bool
                Always True for this class.
        """
        return True

    def u(self, x):
        """
        Evaluate the analytical solution at given coordinates.

        Parameters:
            x: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor
                The analytical solution values at the given coordinates.
        """
        return self.u_func(x, backend=self.backend)

    def grad_u(self, x):
        """
        Evaluate the gradient of the analytical solution.

        Parameters:
            x: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor
                The gradient values at the given coordinates.
        """
        return self.grad_u_func(x, backend=self.backend)

    def compute_relative_grad_error(self, model, X):
        """
        Compute the relative gradient error between model and analytical solution.

        Parameters:
            model: torch.nn.Module
                The neural network model.
            X: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor
                The relative gradient error.
                
        Raises:
            ValueError: If backend is not torch (required for gradients).
        """
        # Gradient computation requires torch backend
        if self.backend != torch:
            raise ValueError("Relative gradient error can only be computed with torch backend.")
        
        # Compute analytical gradient
        true_grad = self.grad_u(X)

        # Compute model gradient using automatic differentiation
        model.zero_grad()
        model_grad = Gradient()(model(X), X)

        # Return relative error
        return relative_error(model_grad, true_grad)
