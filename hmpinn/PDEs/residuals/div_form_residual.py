"""
Divergence Form Residual.

This module provides the divergence form residual implementation for PDEs,
using the Laplacian operator with optional diffusion matrix.

Classes:
    DivFormResidual: Divergence form residual for PDEs.
"""

import torch

from hmpinn.PDEs.residuals.base import BaseResidual
from hmpinn.differential_operators.laplacian import Laplacian


class DivFormResidual(BaseResidual):
    """
    Divergence form residual implementation for PDEs.

    This class computes residuals for PDEs in divergence form using the
    Laplacian operator with optional diffusion matrices.
    """

    def __init__(self, f, diffusion_matrix=None, backend=torch):
        """
        Initialize the divergence form residual.

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
                Always True for divergence form residuals.
        """
        return True

    def differential_operator(self, func, x):
        """
        Compute the differential operator for divergence form residuals.

        Uses the Laplacian operator with optional diffusion matrix.

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model or tensor to apply the operator to.
            x: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor
                The result of applying the Laplacian operator.

        Raises:
            ValueError: If backend is not torch (required for gradients).
        """
        # Divergence form requires torch backend for automatic differentiation
        if self.backend != torch:
            raise ValueError("Backend must be torch for this operation.")

        # Apply Laplacian with diffusion matrix
        return Laplacian()(func, x, k=self.diffusion_matrix)
