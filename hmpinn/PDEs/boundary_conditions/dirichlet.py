from hmpinn.PDEs.boundary_conditions.base import BaseBC
import torch
from hmpinn.PDEs.utils import ensure_backend

class DirichletBC(BaseBC):
    """
    Dirichlet boundary condition class for PDEs.
    Inherits from the BoundaryCondition class.
    """
    
    def __init__(self, boundary_condition_function, backend=torch):
        """
        Initialize the Dirichlet boundary condition with a constant value.

        Parameters:
        boundary_condition_function (callable): A callable function that takes a tensor x as input and returns the boundary condition value.
        backend (torch or np): The backend to use for operations. Default is torch.
        """
        super().__init__(boundary_condition_function, backend=backend)

        # Check if boundary_condition_function is callable
        if not callable(boundary_condition_function):
            raise ValueError("The boundary condition function g must be callable or None.")
        self.boundary_condition_function = boundary_condition_function

    def BC(self, X_boundary):
        """
        Return the Dirichlet boundary condition value.

        Parameters:
        X_boundary (torch.Tensor): The input tensor at the boundary.

        Returns:
        torch.Tensor: The Dirichlet boundary condition value.
        """
        # Ensure the input agrees with the backend
        X_boundary = ensure_backend(X_boundary, self.backend)

        # Apply the boundary condition function to the input tensor
        return self.boundary_condition_function(X_boundary, backend=self.backend)

    @property
    def type_BC(self):
        """
        Return the type of boundary condition.

        Returns:
        str: The type of boundary condition ('Dirichlet').
        """
        return 'Dirichlet'