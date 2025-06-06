from hmpinn.PDEs.harmonic_maps import hm_diffusion_matrix, f_hm
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import backend_to_str
import torch
from functools import partial

def quarter_annulus_BC(x, backend=torch):
    """
    Boundary of quarter annulus

    Parameters:
        x (torch.Tensor): 2D tensor of shape (batch_size, 2)

    Returns:
        torch.Tensor: 2D tensor of shape (batch_size, 2) with values on the boundary of the quarter annulus
    """
    x_input = x[:, 0]
    y_input = x[:, 1]

    # Check that the input is on the boundary of the [0,1]x[0,1] square
    assert backend.all((x_input <= 1) & (x_input >= 0) & (y_input <= 1) & (y_input >= 0)), f"Input must be within the [0,1]x[0,1] square"
    assert backend.all((x_input == 0) | (x_input == 1) | (y_input == 0) | (y_input == 1)), f"Input must on the boundary of the [0,1]x[0,1] square"    
    
    # Make the x coordinate into the range
    r = x_input + 1
    
    # Make the y coordinate into the angle
    theta = y_input * (0.5 * backend.pi)
    
    # Convert polar to Cartesian
    x_out = r * backend.cos(theta)
    y_out = r * backend.sin(theta)

    return backend.stack((x_out, y_out), dim=-1)


class QuarterAnnulusHM(NonDivFormResidual, DirichletBC, WithoutSolution):
    def __init__(self, backend=torch):
        """
        Harmonic map PDE with quarter annulus boundary.

        Parameters:
        backend (torch): The backend library to use (default is torch)
        """
        # Use partial to bind parameters a and b
        f_partial = partial(f_hm, backend=backend)

        NonDivFormResidual.__init__(self, f_partial, hm_diffusion_matrix, backend=backend)
        DirichletBC.__init__(self, quarter_annulus_BC, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"QuarterAnnulusHM(backend={backend_str})"