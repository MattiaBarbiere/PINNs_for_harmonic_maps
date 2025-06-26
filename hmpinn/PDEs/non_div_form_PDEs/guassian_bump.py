"""
Gaussian Bump Source PDE.

This module provides a non-divergence form PDE implementation with a Gaussian
bump source term and no known analytical solution.

Classes:
    GuassianBumpNonDF: PDE with Gaussian bump source term.
"""

import torch
import math
from functools import partial

from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from hmpinn.PDEs.utils import check_backend, ensure_backend, backend_to_str

def f(x, mu_x=0, mu_y=0, std_x=1, std_y=1, backend=torch):
    """
    Gaussian bump source term function.

    Parameters:
        x: torch.Tensor
            Input coordinates of shape (batch_size, 2).
        mu_x: float, optional
            X-coordinate of Gaussian center. Defaults to 0.
        mu_y: float, optional
            Y-coordinate of Gaussian center. Defaults to 0.
        std_x: float, optional
            Standard deviation in x-direction. Defaults to 1.
        std_y: float, optional
            Standard deviation in y-direction. Defaults to 1.
        backend: torch or np, optional
            Backend library to use. Defaults to torch.

    Returns:
        torch.Tensor
            Normalized Gaussian bump values at the given coordinates.
    """
    # Ensure backend compatibility
    if check_backend(backend):
        x = ensure_backend(x, backend)

    # Compute Gaussian bump with normalization
    exp_term = backend.exp(-((x[:, 0] - mu_x) ** 2) / (2 * std_x ** 2) - ((x[:, 1] - mu_y) ** 2) / (2 * std_y ** 2))
    normalization = std_x * std_y * math.sqrt(2 * math.pi)
    
    return exp_term / normalization

class GuassianBumpNonDF(NonDivFormResidual, ConstantBC, WithoutSolution):
    """
    Non-divergence form PDE with Gaussian bump source term.
    
    This class implements a Poisson-type equation where the source term is
    a Gaussian bump centered at specified coordinates.
    """
    
    def __init__(self, mu_x=0, mu_y=0, std_x=1, std_y=1, backend=torch):
        """
        Initialize the Gaussian bump PDE.

        Parameters:
            mu_x: float, optional
                X-coordinate of Gaussian center. Defaults to 0.
            mu_y: float, optional
                Y-coordinate of Gaussian center. Defaults to 0.
            std_x: float, optional
                Standard deviation in x-direction. Defaults to 1.
            std_y: float, optional
                Standard deviation in y-direction. Defaults to 1.
            backend: torch, optional
                Backend library to use. Defaults to torch.
        """
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.std_x = std_x
        self.std_y = std_y

        # Create partial function with bound parameters
        f_partial = partial(f, mu_x=mu_x, mu_y=mu_y, std_x=std_x, std_y=std_y, backend=backend)

        # Initialize parent classes
        NonDivFormResidual.__init__(self, f_partial, backend=backend)
        ConstantBC.__init__(self, backend=backend)
        WithoutSolution.__init__(self, backend=backend)

    def __repr__(self):
        """
        String representation of the GuassianBumpNonDF class.

        Returns:
            str
                String representation including all parameters and backend information.
        """
        backend_str = backend_to_str(self.backend)
        return f"GuassianBumpNonDF(mu_x={self.mu_x}, mu_y={self.mu_y}, std_x={self.std_x}, std_y={self.std_y}, backend={backend_str})"