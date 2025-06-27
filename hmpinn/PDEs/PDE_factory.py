"""
PDE Factory.

This module provides a factory function for dynamically constructing PDE classes
based on specified components (residual type, boundary conditions, solutions).

Functions:
    construct_PDE_class: Factory function to construct custom PDE classes.
"""

import torch
from functools import partial

from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.solutions.without_solution import WithoutSolution

def construct_PDE_class(is_in_divergence_form, f, diffusion_matrix=None, BC=None, u=None, grad_u=None):
    """
    Factory function to construct a custom PDE class.
    
    Dynamically creates a PDE class by combining the appropriate residual,
    boundary condition, and solution components based on the provided parameters.

    Parameters:
        is_in_divergence_form: bool
            Whether the PDE is in divergence form (True) or non-divergence form (False).
        f: callable
            The source term function.
        diffusion_matrix: callable, optional
            The diffusion matrix function. Uses identity if None.
        BC: callable or float or int, optional
            Boundary condition function or constant value. Uses 0 if None.
        u: callable, optional
            Analytical solution function. Uses WithoutSolution if None.
        grad_u: callable, optional
            Gradient of analytical solution. Required if u is provided.

    Returns:
        class
            Dynamically constructed PDE class inheriting from appropriate components.
            
    Raises:
        ValueError: If BC type is invalid or if u is provided without grad_u.
    """
    # Select residual class based on form
    if is_in_divergence_form:
        residual_class = DivFormResidual
    else:
        residual_class = NonDivFormResidual

    # Determine boundary condition class and value
    if BC is None:
        BC = 0
        boundary_class = ConstantBC
    elif isinstance(BC, (int, float)):
        boundary_class = ConstantBC
    elif callable(BC):
        boundary_class = DirichletBC
    else:
        raise ValueError("BC must be a callable function or a constant value.")

    # Determine solution class
    if u is None:
        solution_class = WithoutSolution
    else:
        solution_class = WithSolution

    # Validate gradient requirement
    if grad_u is None and u is not None:
        raise ValueError("If u is provided, grad_u must also be provided.")

    class PDE(residual_class, solution_class, boundary_class):
        """
        Dynamically constructed PDE class.
        
        This class combines the specified residual, solution, and boundary
        condition components to create a complete PDE specification.
        """
        
        def __init__(self, backend=torch):
            """
            Initialize the constructed PDE class.

            Parameters:
                backend: torch or np, optional
                    Backend library to use. Defaults to torch.
            """
            # Create partial functions with bound backend
            f_partial = partial(f, backend=backend)
            u_partial = partial(u, backend=backend) if u else None
            grad_u_partial = partial(grad_u, backend=backend) if grad_u else None
            diffusion_matrix_partial = partial(diffusion_matrix, backend=backend) if diffusion_matrix else None

            BC_partial = partial(BC, backend=backend) if callable(BC) else BC
            
            # Initialize all parent classes
            residual_class.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
            solution_class.__init__(self, u_partial, grad_u_partial, backend=backend)
            boundary_class.__init__(self, BC_partial, backend=backend)

    return PDE
