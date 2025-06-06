from abc import ABC, abstractmethod
import torch
from hmpinn.PDEs.utils import ensure_backend, check_backend

class BaseBC(ABC):
    """
    Abstract base class for boundary conditions in PDEs.
    """
    def __init__(self, input, backend=torch):
        """
        Initialize the boundary condition.
        """
        # Check if the boundary condition is callable or float or int
        if not callable(input) and not isinstance(input, (int, float)):
            raise ValueError("The boundary condition must be callable or a number.")
        
        # Store the backend
        if check_backend(backend):
            self.backend = backend

    @abstractmethod
    def BC(self, X_boundary):
        """
        Return the boundary condition function.
        """
        pass
    
    @property
    @abstractmethod
    def type_BC(self):
        """
        Return the type of boundary condition.
        """
        pass

    def compute_boundary_loss(self, y_boundary, X_boundary):
        """
        Compute the boundary loss.

        Parameters:
        y_boundary (torch.Tensor): The output of the model at the boundary.
        X_boundary (torch.Tensor): The input to the model at the boundary.

        Returns:
        torch.Tensor: The computed boundary loss.
        """
        # Make sure the backend is the same as the input
        y_boundary = ensure_backend(y_boundary, self.backend) # shape: (batch_size, 1)
        X_boundary = ensure_backend(X_boundary, self.backend) 

        # Compute the boundary condition value
        BC_value = self.BC(X_boundary) # shape: (batch_size,)
        
        # Adapt the BC_value to the output shape
        if BC_value.ndim == 1:
            BC_value = BC_value.unsqueeze(1)

        # Check if the shape of the BC_value matches the output shape
        if BC_value.shape != y_boundary.shape:
            raise ValueError(f"The shape of the boundary condition value must match the output shape.{BC_value.shape} != {y_boundary.shape}")
        
        return self.backend.mean((y_boundary - BC_value) ** 2)

        