"""
Harmonic Map with Sinusoidal Boundaries.

This module provides a harmonic map PDE implementation with sinusoidal boundary conditions
that create curved deformations of the unit square boundary.

Classes:
    SinBoundariesHM: Harmonic map with sinusoidal boundary deformations.
"""

import torch
from functools import partial

from hmpinn.PDEs.harmonic_maps import hm_diffusion_matrix, f_hm
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import backend_to_str

def sin_boundaries_BC(x, curvature=0.5, frequency_x=1, frequency_y=1, backend=torch):
    """
    Sinusoidal boundary condition function.
    
    Creates boundary deformations using sinusoidal functions that map
    the unit square boundary to a curved boundary shape.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2).
        curvature: float, optional
            Amplitude of the sinusoidal deformation. Defaults to 0.5.
        frequency_x: int, optional
            Frequency of oscillation in x-direction. Defaults to 1.
        frequency_y: int, optional
            Frequency of oscillation in y-direction. Defaults to 1.
        backend: torch, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Transformed boundary coordinates with sinusoidal deformations.
            
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
    
    # Apply sinusoidal deformations
    x_out = (1 - curvature * torch.sin(frequency_x * torch.pi * y_input)) * x_input - \
            (1 - curvature * torch.sin(frequency_x * torch.pi * y_input)) * (1 - x_input)
    y_out = (1 - curvature * torch.sin(frequency_y * torch.pi * x_input)) * y_input - \
            (1 - curvature * torch.sin(frequency_y * torch.pi * x_input)) * (1 - y_input)
    
    return backend.stack((x_out, y_out), dim=-1)

class SinBoundariesHM(NonDivFormResidual, DirichletBC, WithoutSolution):
    """
    Harmonic map PDE with sinusoidal boundary conditions.
    
    This class implements a harmonic map where the boundary of the unit square
    is deformed using sinusoidal functions, creating curved boundary shapes.
    """
    
    def __init__(self, curvature=0.5, frequency_x=1, frequency_y=1, backend=torch):
        """
        Initialize the harmonic map with sinusoidal boundaries.

        Parameters:
            curvature: float, optional
                Amplitude of sinusoidal boundary deformation. Defaults to 0.5.
            frequency_x: int, optional
                Frequency of oscillation in x-direction. Defaults to 1.
            frequency_y: int, optional
                Frequency of oscillation in y-direction. Defaults to 1.
            backend: torch, optional
                Backend library to use. Defaults to torch.
        """
        # Store boundary parameters
        self.curvature = curvature
        self.frequency_x = frequency_x
        self.frequency_y = frequency_y

        # Create partial functions with bound parameters
        f_partial = partial(f_hm, backend=backend)
        BC_partial = partial(sin_boundaries_BC, curvature=curvature, 
                           frequency_x=frequency_x, frequency_y=frequency_y, backend=backend)

        # Initialize parent classes
        NonDivFormResidual.__init__(self, f_partial, hm_diffusion_matrix, backend=backend)
        DirichletBC.__init__(self, BC_partial, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        """
        String representation of the SinBoundariesHM class.

        Returns:
            str
                String representation including all parameters.
        """
        backend_str = backend_to_str(self.backend)
        return f"SinBoundariesHM(curvature={self.curvature}, frequency_x={self.frequency_x}, frequency_y={self.frequency_y}, backend={backend_str})"