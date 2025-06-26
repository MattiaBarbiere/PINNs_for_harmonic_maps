"""
L-Bend Harmonic Map.

This module provides a harmonic map PDE implementation that maps the unit square
to an L-shaped geometry using boundary conditions.

Classes:
    LBendHM: Harmonic map to L-bend boundary.
"""

import torch
from functools import partial

from hmpinn.PDEs.harmonic_maps import hm_diffusion_matrix, f_hm
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import backend_to_str

def L_bend_BC(x, backend=torch):
    """
    Boundary condition function for L-bend mapping.
    
    Maps the boundary of the unit square to an L-shaped domain by applying
    different transformations based on the coordinate values.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2) on unit square boundary.
        backend: torch, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Transformed boundary coordinates representing L-bend shape.
            
    Raises:
        AssertionError: If input coordinates are not on the unit square boundary.
    """
    x_input = x[:, 0]
    y_input = x[:, 1]

    # Validate input is within unit square
    assert backend.all((x_input <= 1) & (x_input >= 0) & (y_input <= 1) & (y_input >= 0)), \
        f"Input must be within the [0,1]x[0,1] square"
    
    # Validate input is on the boundary
    assert backend.all((x_input == 0) | (x_input == 1) | (y_input == 0) | (y_input == 1)), \
        f"Input must be on the boundary of the [0,1]x[0,1] square"    
    
    # Create L-bend transformation based on coordinate values
    # For coordinates where both x > 0.5 and y > 0.5, apply different scaling
    condition = (x_input > 0.5) & (y_input > 0.5)
    y_out = torch.where(condition, 3 - 2 * y_input, 2 * y_input)
    x_out = torch.where(condition, 3 - 2 * x_input, 2 * x_input)

    return backend.stack((x_out, y_out), dim=-1)

class LBendHM(NonDivFormResidual, DirichletBC, WithoutSolution):
    """
    Harmonic map PDE with L-bend boundary conditions.
    
    This class implements a harmonic map that transforms the unit square domain
    into an L-shaped geometry through appropriate boundary conditions.
    """
    
    def __init__(self, backend=torch):
        """
        Initialize the harmonic map with L-bend boundary.

        Parameters:
            backend: torch, optional
                Backend library to use for computations. Defaults to torch.
        """
        # Create partial functions with bound backend
        f_partial = partial(f_hm, backend=backend)

        # Initialize parent classes
        NonDivFormResidual.__init__(self, f_partial, hm_diffusion_matrix, backend=backend)
        DirichletBC.__init__(self, L_bend_BC, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        """
        String representation of the LBendHM class.

        Returns:
            str
                String representation including backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"LBendHM(backend={backend_str})"