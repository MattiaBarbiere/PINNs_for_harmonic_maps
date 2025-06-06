from hmpinn.PDEs.harmonic_maps import hm_diffusion_matrix, f_hm
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import backend_to_str
import torch
from functools import partial

def sin_boundaries_BC(x, curvature=0.5, frequency_x=1, frequency_y=1, backend=torch):
    """
    Boundary made of sin functions

    Parameters:
        x (torch.Tensor): 2D tensor of shape (batch_size, 2)

    Returns:
        torch.Tensor: 2D tensor of shape (batch_size, 2) with values on the boundary
    """
    x_input = x[:, 0]
    y_input = x[:, 1]

    # Check that the input is on the boundary of the [0,1]x[0,1] square
    assert backend.all((x_input <= 1) & (x_input >= 0) & (y_input <= 1) & (y_input >= 0)), f"Input must be within the [0,1]x[0,1] square"
    assert backend.all((x_input == 0) | (x_input == 1) | (y_input == 0) | (y_input == 1)), f"Input must on the boundary of the [0,1]x[0,1] square"    
    
    x_out = (1 - curvature * torch.sin(frequency_x * torch.pi * y_input)) * x_input - (1 - curvature * torch.sin(frequency_x * torch.pi * y_input)) * (1 - x_input)
    y_out = (1 - curvature * torch.sin(frequency_y * torch.pi * x_input)) * y_input - (1 - curvature * torch.sin(frequency_y * torch.pi * x_input)) * (1 - y_input)
    
    return backend.stack((x_out, y_out), dim=-1)


class SinBoundariesHM(NonDivFormResidual, DirichletBC, WithoutSolution):
    def __init__(self, curvature=0.5, frequency_x=1, frequency_y=1, backend=torch):
        """
        Harmonic map PDE with sin boundaries.

        Parameters:
        backend (torch): The backend library to use (default is torch)
        """
        # Add the parameters to the class
        self.curvature = curvature
        self.frequency_x = frequency_x
        self.frequency_y = frequency_y

        # Use partial to bind parameters a and b
        f_partial = partial(f_hm, backend=backend)
        BC_partial = partial(sin_boundaries_BC, curvature=curvature, frequency_x=frequency_x, frequency_y=frequency_y, backend=backend)

        NonDivFormResidual.__init__(self, f_partial, hm_diffusion_matrix, backend=backend)
        DirichletBC.__init__(self, BC_partial, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"SinBoundariesHM(curvature={self.curvature}, frequency_x={self.frequency_x}, frequency_y={self.frequency_y}, backend={backend_str})"