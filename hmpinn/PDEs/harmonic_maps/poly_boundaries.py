"""
Polynomial Boundaries Harmonic Map.

This module provides a harmonic map PDE implementation with asymmetric polynomial
boundary conditions that create complex deformations of the unit square boundary.

Classes:
    PolynomialBoundariesHM: Harmonic map with asymmetric polynomial boundary deformations.
"""

import torch
from functools import partial

from hmpinn.PDEs.harmonic_maps import hm_diffusion_matrix, f_hm
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import backend_to_str

def polynomial_boundaries_BC(x, a_left=0.3, a_right=0.1, b_bottom=0.2, b_top=0.4, degree=3, backend=torch):
    """
    Asymmetric polynomial boundary condition function.
    
    Creates complex boundary deformations using different polynomial functions
    for each boundary edge, resulting in asymmetric domain shapes.

    Parameters:
        x: torch.Tensor
            Boundary coordinates of shape (batch_size, 2).
        a_left: float, optional
            Amplitude of left boundary deformation. Defaults to 0.3.
        a_right: float, optional
            Amplitude of right boundary deformation. Defaults to 0.1.
        b_bottom: float, optional
            Amplitude of bottom boundary deformation. Defaults to 0.2.
        b_top: float, optional
            Amplitude of top boundary deformation. Defaults to 0.4.
        degree: int, optional
            Degree of polynomial (3 for cubic, 4 for quartic). Defaults to 3.
        backend: torch, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Transformed boundary coordinates with asymmetric polynomial deformations.
            
    Raises:
        AssertionError: If input coordinates are not on the unit square boundary.
        ValueError: If degree is not 3 or 4.
    """
    x_input = x[:, 0]
    y_input = x[:, 1]

    # Validate input is within unit square
    assert backend.all((x_input <= 1) & (x_input >= 0) & (y_input <= 1) & (y_input >= 0)), \
        f"Input must be within the [0,1]x[0,1] square"
    
    # Validate input is on the boundary
    assert backend.all((x_input == 0) | (x_input == 1) | (y_input == 0) | (y_input == 1)), \
        f"Input must be on the boundary of the [0,1]x[0,1] square"    
    
    # Define different polynomial deformation functions for asymmetry
    if degree == 3:
        # Asymmetric cubic polynomials
        poly_left = y_input**2 * (1 - y_input)  # Different shape for left
        poly_right = y_input * (1 - y_input)**2  # Different shape for right
        poly_bottom = x_input * (1 - x_input)**2  # Different shape for bottom
        poly_top = x_input**2 * (1 - x_input)  # Different shape for top
    elif degree == 4:
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
    """
    Harmonic map PDE with asymmetric polynomial boundary conditions.
    
    This class implements a harmonic map where each boundary of the unit square
    is deformed using different polynomial functions, creating complex asymmetric shapes.
    """
    
    def __init__(self, a_left=0.3, a_right=0.1, b_bottom=0.2, b_top=0.4, degree=3, backend=torch):
        """
        Initialize the harmonic map with asymmetric polynomial boundaries.

        Parameters:
            a_left: float, optional
                Amplitude of left boundary deformation. Defaults to 0.3.
            a_right: float, optional
                Amplitude of right boundary deformation. Defaults to 0.1.
            b_bottom: float, optional
                Amplitude of bottom boundary deformation. Defaults to 0.2.
            b_top: float, optional
                Amplitude of top boundary deformation. Defaults to 0.4.
            degree: int, optional
                Degree of polynomial deformation (3 or 4). Defaults to 3.
            backend: torch, optional
                Backend library to use. Defaults to torch.
        """
        # Store boundary parameters
        self.a_left = a_left
        self.a_right = a_right
        self.b_bottom = b_bottom
        self.b_top = b_top
        self.degree = degree

        # Create partial functions with bound parameters
        f_partial = partial(f_hm, backend=backend)
        BC_partial = partial(polynomial_boundaries_BC, a_left=a_left, a_right=a_right, 
                           b_bottom=b_bottom, b_top=b_top, degree=degree, backend=backend)

        # Initialize parent classes
        NonDivFormResidual.__init__(self, f_partial, hm_diffusion_matrix, backend=backend)
        DirichletBC.__init__(self, BC_partial, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        """
        String representation of the PolynomialBoundariesHM class.

        Returns:
            str
                String representation including all parameters.
        """
        backend_str = backend_to_str(self.backend)
        return f"PolynomialBoundariesHM(a_left={self.a_left}, a_right={self.a_right}, b_bottom={self.b_bottom}, b_top={self.b_top}, degree={self.degree}, backend={backend_str})"