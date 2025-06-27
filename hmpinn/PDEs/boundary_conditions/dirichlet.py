"""
Dirichlet Boundary Condition.

This module provides the Dirichlet boundary condition implementation for PDEs,
where the solution value is specified on the boundary.

Classes:
    DirichletBC: Dirichlet boundary condition.
"""

import torch

from hmpinn.PDEs.boundary_conditions.base import BaseBC
from hmpinn.PDEs.utils import ensure_backend

class DirichletBC(BaseBC):
    """
    Dirichlet boundary condition class for PDEs.
    
    This boundary condition specifies the value of the solution on the boundary
    using a callable function.
    """
    
    def __init__(self, boundary_condition_function, backend=torch):
        """
        Initialize the Dirichlet boundary condition with a function.

        Parameters:
            boundary_condition_function: callable
                A function that takes coordinates and returns boundary values.
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
                
        Raises:
            ValueError: If boundary_condition_function is not callable.
        """
        super().__init__(boundary_condition_function, backend=backend)

        # Validate that boundary condition function is callable
        if not callable(boundary_condition_function):
            raise ValueError("The boundary condition function g must be callable or None.")
        self.boundary_condition_function = boundary_condition_function

    def BC(self, X_boundary):
        """
        Evaluate the Dirichlet boundary condition at given coordinates.

        Parameters:
            X_boundary: torch.Tensor
                The coordinates at the boundary where to evaluate the condition.

        Returns:
            torch.Tensor
                The boundary condition values at the specified coordinates.
        """
        # Ensure input matches the backend
        X_boundary = ensure_backend(X_boundary, self.backend)

        # Apply the boundary condition function to the input coordinates
        return self.boundary_condition_function(X_boundary, backend=self.backend)

    @property
    def type_BC(self):
        """
        Get the type of boundary condition.

        Returns:
            str
                The boundary condition type ('Dirichlet').
        """
        return 'Dirichlet'