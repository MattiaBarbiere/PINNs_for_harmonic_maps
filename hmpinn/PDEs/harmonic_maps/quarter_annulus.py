"""
Quarter Annulus Harmonic Map.

This module provides a harmonic map PDE implementation that maps the unit square
to a quarter annulus geometry using boundary conditions.

Classes:
    QuarterAnnulusHM: Harmonic map to quarter annulus boundary.
"""

import torch
from functools import partial

from hmpinn.PDEs.harmonic_maps import hm_diffusion_matrix, f_hm
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import backend_to_str

def quarter_annulus_BC(x, backend=torch):
    """
    Boundary condition function for quarter annulus mapping.
    
    Maps the boundary of the unit square to a quarter annulus in polar coordinates,
    where the x-coordinate becomes the radial distance and y-coordinate becomes the angle.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2) on unit square boundary.
        backend: torch, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Transformed boundary coordinates representing quarter annulus.
            
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
    
    # Transform x-coordinate to radial distance (range [1, 2])
    r = x_input + 1
    
    # Transform y-coordinate to angle (range [0, π/2])
    theta = y_input * (0.5 * backend.pi)
    
    # Convert polar to Cartesian coordinates
    x_out = r * backend.cos(theta)
    y_out = r * backend.sin(theta)

    return backend.stack((x_out, y_out), dim=-1)

class QuarterAnnulusHM(NonDivFormResidual, DirichletBC, WithoutSolution):
    """
    Harmonic map PDE with quarter annulus boundary conditions.
    
    This class implements a harmonic map that transforms the unit square domain
    into a quarter annulus shape through appropriate boundary conditions.
    """
    
    def __init__(self, backend=torch):
        """
        Initialize the harmonic map with quarter annulus boundary.

        Parameters:
            backend: torch, optional
                Backend library to use for computations. Defaults to torch.
        """
        # Create partial functions with bound backend
        f_partial = partial(f_hm, backend=backend)

        # Initialize parent classes
        NonDivFormResidual.__init__(self, f_partial, hm_diffusion_matrix, backend=backend)
        DirichletBC.__init__(self, quarter_annulus_BC, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        """
        String representation of the QuarterAnnulusHM class.

        Returns:
            str
                String representation including backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"QuarterAnnulusHM(backend={backend_str})"