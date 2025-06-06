from hmpinn.PDEs.harmonic_maps import hm_diffusion_matrix, f_hm
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import backend_to_str
import torch
from functools import partial

def polynomial_boundaries_BC(x, a_left=0.3, a_right=0.1, b_bottom=0.2, b_top=0.4, degree=3, backend=torch):
    """
    Boundary made of asymmetric polynomial functions that deform the unit square

    Parameters:
        x (torch.Tensor): 2D tensor of shape (batch_size, 2)
        a_left (float): Amplitude of left boundary deformation
        a_right (float): Amplitude of right boundary deformation
        b_bottom (float): Amplitude of bottom boundary deformation  
        b_top (float): Amplitude of top boundary deformation
        degree (int): Degree of polynomial (3 for cubic, 4 for quartic)
        backend: Backend library to use

    Returns:
        torch.Tensor: 2D tensor of shape (batch_size, 2) with values on the boundary
    """
    x_input = x[:, 0]
    y_input = x[:, 1]

    # Check that the input is on the boundary of the [0,1]x[0,1] square
    assert backend.all((x_input <= 1) & (x_input >= 0) & (y_input <= 1) & (y_input >= 0)), f"Input must be within the [0,1]x[0,1] square"
    assert backend.all((x_input == 0) | (x_input == 1) | (y_input == 0) | (y_input == 1)), f"Input must on the boundary of the [0,1]x[0,1] square"    
    
    # Different polynomial deformation functions for asymmetry
    if degree == 3:
        # Asymmetric cubic polynomials
        poly_left = y_input**2 * (1 - y_input)  # Different shape for left
        poly_right = y_input * (1 - y_input)**2  # Different shape for right
        poly_bottom = x_input * (1 - x_input)**2  # Different shape for bottom
        poly_top = x_input**2 * (1 - x_input)  # Different shape for top
    if degree == 4:
        # Asymmetric quartic polynomials
        poly_left = y_input**2 * (1 - y_input) * (1 - 0.5*y_input)
        poly_right = y_input * (1 - y_input)**2 * (0.5 + y_input)
        poly_bottom = x_input * (1 - x_input)**2 * (1 - 0.3*x_input)
        poly_top = x_input**2 * (1 - x_input) * (0.7 + x_input)
    else:
        raise ValueError("Degree must be either 3 or 4 for polynomial boundaries")
    
    # Initialize output coordinates
    x_out = x_input.clone()
    y_out = y_input.clone()
    
    # Apply different deformations to each boundary
    # Left boundary (x=0): move inward with left amplitude
    left_mask = (x_input == 0)
    x_out = backend.where(left_mask, x_input + a_left * poly_left, x_out)
    
    # Right boundary (x=1): move inward with right amplitude  
    right_mask = (x_input == 1)
    x_out = backend.where(right_mask, x_input - a_right * poly_right, x_out)
    
    # Bottom boundary (y=0): move inward with bottom amplitude
    bottom_mask = (y_input == 0)
    y_out = backend.where(bottom_mask, y_input + b_bottom * poly_bottom, y_out)
    
    # Top boundary (y=1): move inward with top amplitude
    top_mask = (y_input == 1)
    y_out = backend.where(top_mask, y_input - b_top * poly_top, y_out)
    
    return backend.stack((x_out, y_out), dim=-1)


class PolynomialBoundariesHM(NonDivFormResidual, DirichletBC, WithoutSolution):
    def __init__(self, a_left=0.3, a_right=0.1, b_bottom=0.2, b_top=0.4, degree=3, backend=torch):
        """
        Harmonic map PDE with asymmetric polynomial boundaries.

        Parameters:
        a_left (float): Amplitude of left boundary deformation
        a_right (float): Amplitude of right boundary deformation
        b_bottom (float): Amplitude of bottom boundary deformation
        b_top (float): Amplitude of top boundary deformation
        degree (int): Degree of polynomial deformation
        backend (torch): The backend library to use (default is torch)
        """
        # Add the parameters to the class
        self.a_left = a_left
        self.a_right = a_right
        self.b_bottom = b_bottom
        self.b_top = b_top
        self.degree = degree

        # Use partial to bind parameters
        f_partial = partial(f_hm, backend=backend)
        BC_partial = partial(polynomial_boundaries_BC, a_left=a_left, a_right=a_right, 
                           b_bottom=b_bottom, b_top=b_top, degree=degree, backend=backend)

        NonDivFormResidual.__init__(self, f_partial, hm_diffusion_matrix, backend=backend)
        DirichletBC.__init__(self, BC_partial, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        backend_str = backend_to_str(self.backend)
        return f"PolynomialBoundariesHM(a_left={self.a_left}, a_right={self.a_right}, b_bottom={self.b_bottom}, b_top={self.b_top}, degree={self.degree}, backend={backend_str})"