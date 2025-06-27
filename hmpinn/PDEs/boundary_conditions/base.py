"""
Base Boundary Condition.

This module provides the abstract base class for all boundary conditions
in the hmpinn library PDE framework.

Classes:
    BaseBC: Abstract base class for boundary conditions.
"""

from abc import ABC, abstractmethod
import torch

from hmpinn.PDEs.utils import ensure_backend, check_backend

class BaseBC(ABC):
    """
    Abstract base class for boundary conditions in PDEs.
    
    This class provides the common interface and functionality for all
    boundary condition implementations, including loss computation.
    """
    
    def __init__(self, input, backend=torch):
        """
        Initialize the boundary condition base class.
        
        Parameters:
            input: callable or int or float
                The boundary condition specification (function or constant value).
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
                
        Raises:
            ValueError: If input is neither callable nor numeric.
        """
        # Validate input type
        if not callable(input) and not isinstance(input, (int, float)):
            raise ValueError("The boundary condition must be callable or a number.")
        
        # Store and validate the backend
        if check_backend(backend):
            self.backend = backend

    @abstractmethod
    def BC(self, X_boundary):
        """
        Evaluate the boundary condition at given coordinates.
        
        Must be implemented by all subclasses.
        
        Parameters:
            X_boundary: torch.Tensor
                The coordinates at the boundary.
                
        Returns:
            torch.Tensor
                The boundary condition values.
        """
        pass
    
    @property
    @abstractmethod
    def type_BC(self):
        """
        Get the type of boundary condition.
        
        Must be implemented by all subclasses.
        
        Returns:
            str
                The boundary condition type identifier.
        """
        pass

    def compute_boundary_loss(self, y_boundary, X_boundary):
        """
        Compute the boundary condition loss.
        
        Calculates the mean squared error between the model output
        and the boundary condition values.

        Parameters:
            y_boundary: torch.Tensor
                The model output at boundary coordinates of shape (batch_size, 1).
            X_boundary: torch.Tensor
                The boundary coordinates.

        Returns:
            torch.Tensor
                The computed boundary loss (mean squared error).
                
        Raises:
            ValueError: If boundary condition shape doesn't match output shape.
        """
        # Ensure inputs match the backend
        y_boundary = ensure_backend(y_boundary, self.backend)
        X_boundary = ensure_backend(X_boundary, self.backend) 

        # Compute the boundary condition values
        BC_value = self.BC(X_boundary)
        
        # Reshape BC_value to match output shape if necessary
        if BC_value.ndim == 1:
            BC_value = BC_value.unsqueeze(1)

        # Validate shape compatibility
        if BC_value.shape != y_boundary.shape:
            raise ValueError(f"The shape of the boundary condition value must match the output shape.{BC_value.shape} != {y_boundary.shape}")
        
        # Compute and return mean squared error
        return self.backend.mean((y_boundary - BC_value) ** 2)

