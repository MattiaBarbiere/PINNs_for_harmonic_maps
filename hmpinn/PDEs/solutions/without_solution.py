"""
Solution Class without Known Analytical Solution.

This module provides the solution implementation for PDEs without known
analytical solutions, such as complex nonlinear problems.

Classes:
    WithoutSolution: Solution class for PDEs without known analytical solutions.
"""

import torch

from hmpinn.PDEs.solutions.base import BaseSolution

class WithoutSolution(BaseSolution):
    """
    Solution class for PDEs without known analytical solutions.
    
    This class handles PDEs where no analytical solution is available,
    such as complex nonlinear problems or harmonic maps.
    """
    
    def __init__(self, *args, backend=torch):
        """
        Initialize without analytical solution.

        Parameters:
            *args: 
                Unused arguments for interface consistency.
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
        """
        super().__init__(backend=backend)

    @property   
    def has_solution(self):
        """
        Check if an analytical solution exists.

        Returns:
            bool
                Always False for this class.
        """
        return False

    def u(self, x):
        """
        Evaluate the analytical solution (not available).

        Parameters:
            x: torch.Tensor
                The input coordinates (unused).

        Returns:
            None
                Always returns None as no solution is available.
        """
        return None

    def grad_u(self, x):
        """
        Evaluate the gradient of the analytical solution (not available).

        Parameters:
            x: torch.Tensor
                The input coordinates (unused).

        Returns:
            None
                Always returns None as no solution is available.
        """
        return None

    def compute_relative_grad_error(self, model, X):
        """
        Compute the relative gradient error (not available).

        Parameters:
            model: torch.nn.Module
                The neural network model (unused).
            X: torch.Tensor
                The input coordinates (unused).

        Returns:
            None
                Always returns None as no analytical solution is available.
        """
        return None