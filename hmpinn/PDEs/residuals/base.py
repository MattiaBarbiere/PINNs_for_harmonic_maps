from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn
from functools import partial
from hmpinn.PDEs.utils import relative_error, check_backend, ensure_backend

def default_diffusion_matrix(x, model=None, backend=torch):
    """
    Default diffusion matrix function that returns an identity matrix.

    Parameters:
    x (torch.Tensor): The input tensor.
    model (torch.nn.Module, optional): Model that is trying to solve the pde. This is used only for PDEs that have
                                        a diffusion matrix that depends on the true solution (default is None).
                                        Not used in this case.
    backend (torch or np): The backend to use for operations. Default is torch.

    Returns:
    torch.Tensor or np.ndarray: Identity matrix of size (2, 2) repeated for the batch size.
    """
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    if backend == torch:
        return torch.eye(2, device=x.device).repeat(x.shape[0], 1, 1)
    else:
        return np.repeat(np.eye(2)[np.newaxis, :, :], x.shape[0], axis=0)


class BaseResidual(ABC):
    def __init__(self, f, diffusion_matrix=None, backend=torch):
        """
        Abstract class for the residual of a PDE.

        Parameters:
        f (callable): The source term function.
        diffusion_matrix (callable, optional): The diffusion matrix function. If None uses identity. 
                                                Defaults to None.
        backend (torch or np): The backend to use for operations. Default is torch.
        """
        # Some checks to the inputs
        if not callable(f):
            raise TypeError("f must be a callable function")
        if diffusion_matrix is not None and not callable(diffusion_matrix):
            raise TypeError("diffusion_matrix must be a callable function or None")
        
        self.f = f

        if diffusion_matrix is None:
            self.diffusion_matrix = partial(default_diffusion_matrix, backend=backend)
        else:
            self.diffusion_matrix = partial(diffusion_matrix, backend=backend)

        # Initialize the some variables
        self.relative_residual_error = None         # The relative L^2 residual error of the loss (torch.Tensor)

        if check_backend(backend):
            self.backend = backend

    @property
    @abstractmethod
    def is_in_divergence_form(self):
        """
        Abstract property to check if the residual is in divergence form.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def differential_operator(self, func, x):
        """
        Abstract method to compute the differential operator.
        Must be implemented in subclasses.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model to compute the derivative of or a tensor of size (batch_size)
        x (torch.Tensor): The input to the model

        Returns:
        DifferentialOperator: The differential operator of the model wrt x
        """
        pass

    def compute_residual(self, model, X):
        """
        Compute the residual of the model.

        Returns:
        torch.Tensor: The residual of the model.
        """
        # If the backend is not torch, we cannot perform this opearation
        if self.backend != torch:
            raise ValueError("Backend must be torch to compute the residual")
        
        # Compute the relevant quantities
        differential_val = self.differential_operator(model, X)
        real_source = self.f(X)

        # Compute the relative residual error
        self.relative_residual_error = relative_error(differential_val, real_source) # Relative error of the residual (torch.Tensor)

        return nn.MSELoss()(differential_val, real_source)       
        
        