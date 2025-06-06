import torch
from hmpinn.PDEs.boundary_conditions.dirichlet import DirichletBC
from hmpinn.PDEs.boundary_conditions.constant import ConstantBC
from hmpinn.PDEs.residuals.div_form_residual import DivFormResidual
from hmpinn.PDEs.residuals.non_div_form_residual import NonDivFormResidual
from hmpinn.PDEs.solutions.with_solution import WithSolution
from hmpinn.PDEs.solutions.without_solution import WithoutSolution
from functools import partial


def construct_PDE_class(is_in_divergence_form, f, diffusion_matrix=None, BC=None, u=None, grad_u=None):
    """
    Construct a PDE object based on the provided parameters.

    Parameters:
    is_in_divergence_form (bool): Flag to indicate if the PDE is in divergence form. Defaults to True.
    f (callable): The source term function.
    diffusion_matrix (callable, optional): The diffusion matrix function. If None uses identity. 
                                            Defaults to None.
    BC (callable or float or int, optional): The boundary condition function or value. If None, uses constant 0.
    u (callable, optional): The solution function. If None uses WithSolution. Defaults to None.
    grad_u (callable, optional): The gradient of the solution function. If None and u is not None, uses WithSolution and computes using autograd.
    backend (torch or np): The backend to use for the arrays or tensors. Defaults to torch.

    Returns:
    object: The constructed PDE object.
    """
       
    if is_in_divergence_form:
        residual_class = DivFormResidual
    else:
        residual_class = NonDivFormResidual

    if BC is None:
        BC = 0
        boundary_class = ConstantBC
    elif isinstance(BC, (int, float)):
        boundary_class = ConstantBC
    elif callable(BC):
        boundary_class = DirichletBC
    else:
        raise ValueError("BC must be a callable function or a constant value.")

    if u is None:
        solution_class = WithoutSolution
    else:
        solution_class = WithSolution

    if grad_u is None and u is not None:
        raise ValueError("If u is provided, grad_u must also be provided.")

    class PDE(residual_class, solution_class, boundary_class):
        def __init__(self, backend=torch):
            f_partial = partial(f, backend=backend)
            u_partial = partial(u, backend=backend) if u else None
            grad_u_partial = partial(grad_u, backend=backend) if grad_u else None
            diffusion_matrix_partial = partial(diffusion_matrix, backend=backend) if diffusion_matrix else None

            BC_partial = partial(BC, backend=backend) if callable(BC) else BC
            
            residual_class.__init__(self, f_partial, diffusion_matrix_partial, backend=backend)
            solution_class.__init__(self, u_partial, grad_u_partial, backend=backend)
            boundary_class.__init__(self, BC_partial, backend=backend)

    return PDE
