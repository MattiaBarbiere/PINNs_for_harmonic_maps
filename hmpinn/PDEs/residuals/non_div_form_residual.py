"""
Non-Divergence Form Residual.

This module provides the non-divergence form residual implementation for PDEs,
using the Hessian operator with diffusion matrix.

Classes:
    NonDivFormResidual: Non-divergence form residual for PDEs.
"""

import torch

from hmpinn.PDEs.residuals.base import BaseResidual
from hmpinn.differential_operators.hessian import Hessian


class NonDivFormResidual(BaseResidual):
    """
    Non-divergence form residual implementation for PDEs.

    This class computes residuals for PDEs in non-divergence form using the
    Hessian operator with diffusion matrices.
    """

    def __init__(self, f, diffusion_matrix=None, backend=torch):
        """
        Initialize the non-divergence form residual.

        Parameters:
            f: callable
                The source term function.
            diffusion_matrix: callable, optional
                The diffusion matrix function. Uses identity if None.
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
        """
        super().__init__(f, diffusion_matrix, backend)

    @property
    def is_in_divergence_form(self):
        """
        Check if the residual is in divergence form.

        Returns:
            bool
                Always False for non-divergence form residuals.
        """
        return False

    def differential_operator(self, func, x):
        """
        Compute the differential operator for non-divergence form residuals.

        Uses the Hessian operator with diffusion matrix via Frobenius product.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model or tensor to apply the operator to.
            x: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor
                The result of applying the Hessian operator with diffusion matrix.

        Raises:
            ValueError: If backend is not torch (required for gradients).
        """
        # Non-divergence form requires torch backend for automatic differentiation
        if self.backend != torch:
            raise ValueError("Backend must be torch for this operation.")

        # Compute Hessian and apply diffusion matrix via element-wise product and sum
        return (Hessian()(func, x) * self.diffusion_matrix(x, model=func)).sum(dim=(-2, -1))

