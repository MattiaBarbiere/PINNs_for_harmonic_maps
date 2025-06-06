from hmpinn.PDEs.boundary_conditions.base import BaseBC
import torch
from hmpinn.PDEs.utils import ensure_backend, ones

class ConstantBC(BaseBC):
    """
    Constant boundary condition class for PDEs.
    Inherits from the BoundaryCondition class.
    """
    
    def __init__(self, boundary_condition_value=0, backend=torch):
        """
        Initialize the Constant boundary condition with a constant value.

        Parameters:
        boundary_condition_value (float or int): The boundary condition value. Default is 0.
        backend (torch or np): The backend to use for operations. Default is torch.
        """
        super().__init__(boundary_condition_value, backend=backend)

        # Check if boundary_condition_value is a number
        if not isinstance(boundary_condition_value, (int, float)):
            raise ValueError("The boundary condition value must be a number or None.")

        self.boundary_condition_value = boundary_condition_value

    def BC(self, X_boundary):
        """
        Return the constant boundary condition value.

        Parameters:
        X_boundary (torch.Tensor): The input tensor at the boundary.

        Returns:
        torch.Tensor: The constant boundary condition value.
        """
        # Ensure the input agress with the backend
        X_boundary = ensure_backend(X_boundary, self.backend)

        # Return the constant value for all inputs
        return ones(X_boundary, backend=self.backend, add_extra_dim=False) * self.boundary_condition_value
    
    
    @property
    def type_BC(self):
        """
        Return the type of boundary condition.

        Returns:
        str: The type of boundary condition ('Constant').
        """
        return 'Constant'