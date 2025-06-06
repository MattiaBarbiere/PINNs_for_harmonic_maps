from hmpinn.PDEs.residuals.base import BaseResidual
from hmpinn.differential_operators.hessian import Hessian
import torch


class NonDivFormResidual(BaseResidual):
    def __init__(self, f, diffusion_matrix=None, backend=torch):
        """
        Class for the non divergence form residual of a PDE.

        Parameters:
        f (callable): The source term function.
        diffusion_matrix (callable, optional): The diffusion matrix function. If None uses identity. 
                                                Defaults to None.
        backend (torch or np): The backend to use for operations. Default is torch.
        """
        super().__init__(f, diffusion_matrix, backend)

    @property
    def is_in_divergence_form(self):
        """
        Check if the residual is in divergence form.
        """
        return False

    def differential_operator(self, func, x):
        """
        Compute the differential operator for the non divergence form residual.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model to compute the derivative of or a tensor of size (batch_size)
        x (torch.Tensor): The input to the model

        Returns:
        torch.tensor: The differential operator of the model wrt x (batch_size)
        """
        # If the backend is not torch we cannot perform this
        if self.backend != torch:
            raise ValueError("Backend must be torch for this operation.")

        return (Hessian()(func, x) * self.diffusion_matrix(x, model=func)).sum(dim = (-2, -1))

