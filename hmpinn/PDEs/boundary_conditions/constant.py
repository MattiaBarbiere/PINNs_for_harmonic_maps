"""
Constant Boundary Condition.

This module provides the constant boundary condition implementation for PDEs,
where the solution has a constant value on the boundary.

Classes:
    ConstantBC: Constant boundary condition with uniform values.
"""

import torch

from hmpinn.PDEs.boundary_conditions.base import BaseBC
from hmpinn.PDEs.utils import ensure_backend, ones

class ConstantBC(BaseBC):
    """
    Constant boundary condition class for PDEs.
    
    This boundary condition specifies a constant value for the solution
    on the entire boundary.
    """
    
    def __init__(self, boundary_condition_value=0, backend=torch):
        """
        Initialize the constant boundary condition with a fixed value.

        Parameters:
            boundary_condition_value: float or int, optional
                The constant boundary condition value. Defaults to 0.
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
                
        Raises:
            ValueError: If boundary_condition_value is not a number.
        """
        super().__init__(boundary_condition_value, backend=backend)

        # Validate that boundary condition value is numeric
        if not isinstance(boundary_condition_value, (int, float)):
            raise ValueError("The boundary condition value must be a number or None.")

        self.boundary_condition_value = boundary_condition_value

    def BC(self, X_boundary):
        """
        Evaluate the constant boundary condition at given coordinates.

        Parameters:
            X_boundary: torch.Tensor
                The coordinates at the boundary (used only for shape information).

        Returns:
            torch.Tensor
                The constant boundary condition value for all input coordinates.
        """
        # Ensure input matches the backend
        X_boundary = ensure_backend(X_boundary, self.backend)

        # Return the constant value for all input coordinates
        return ones(X_boundary, backend=self.backend, add_extra_dim=False) * self.boundary_condition_value
    
    @property
    def type_BC(self):
        """
        Get the type of boundary condition.

        Returns:
            str
                The boundary condition type ('Constant').
        """
        return 'Constant'