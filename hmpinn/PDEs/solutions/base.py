"""
Base Solution Class.

This module provides the abstract base class for all PDE solution implementations
in the hmpinn library, defining the common interface for solution handling.

Classes:
    BaseSolution: Abstract base class for PDE solutions.
"""

from abc import ABC, abstractmethod
import torch

from hmpinn.PDEs.utils import check_backend

class BaseSolution(ABC):
    """
    Abstract base class for PDE solutions.
    
    This class defines the common interface for all solution implementations,
    whether analytical solutions are known or unknown.
    """

    def __init__(self, backend=torch):
        """
        Initialize the base solution class.
        
        Parameters:
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
        """
        if check_backend(backend):
            self.backend = backend
    
    @property
    @abstractmethod
    def has_solution(self):
        """
        Check if an analytical solution exists.
        
        Must be implemented by subclasses.
        
        Returns:
            bool
                True if analytical solution is available, False otherwise.
        """
        pass

    @abstractmethod
    def u(self, x):
        """
        Evaluate the analytical solution at given coordinates.
        
        Must be implemented by subclasses.

        Parameters:
            x: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor or None
                The solution values, or None if no solution exists.
        """
        pass

    @abstractmethod
    def grad_u(self, x):
        """
        Evaluate the gradient of the analytical solution.
        
        Must be implemented by subclasses.

        Parameters:
            x: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor or None
                The gradient values, or None if no solution exists.
        """
        pass

    @abstractmethod
    def compute_relative_grad_error(self, model, X):
        """
        Compute the relative gradient error between model and analytical solution.
        
        Must be implemented by subclasses.

        Parameters:
            model: torch.nn.Module
                The neural network model.
            X: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor or None
                The relative gradient error, or None if no solution exists.
        """
        pass

