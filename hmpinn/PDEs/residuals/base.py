"""
Base Residual Class.

This module provides the abstract base class for PDE residuals and utilities
for residual computation in both divergence and non-divergence forms.

Functions:
    default_diffusion_matrix: Default identity diffusion matrix function.

Classes:
    BaseResidual: Abstract base class for PDE residuals.
"""

from abc import ABC, abstractmethod
from functools import partial
import torch
import numpy as np
import torch.nn as nn

from hmpinn.PDEs.utils import relative_error, check_backend, ensure_backend

def default_diffusion_matrix(x, model=None, backend=torch):
    """
    Default diffusion matrix function returning identity matrix.
    
    Provides an identity diffusion matrix for isotropic diffusion problems.

    Parameters:
        x: torch.Tensor or np.ndarray
            Input coordinates of shape (batch_size, input_dim).
        model: torch.nn.Module, optional
            Model for PDEs with solution-dependent diffusion. Not used here.
        backend: torch or np, optional
            Backend library for operations. Defaults to torch.

    Returns:
        torch.Tensor or np.ndarray
            Identity matrix of shape (batch_size, 2, 2).
    """
    if check_backend(backend):
        x = ensure_backend(x, backend)
    
    if backend == torch:
        return torch.eye(2, device=x.device).repeat(x.shape[0], 1, 1)
    else:
        return np.repeat(np.eye(2)[np.newaxis, :, :], x.shape[0], axis=0)

class BaseResidual(ABC):
    """
    Abstract base class for PDE residuals.
    
    This class provides the common interface and functionality for computing
    PDE residuals in both divergence and non-divergence forms.
    
    Attributes:
        f: Source term function.
        diffusion_matrix: Diffusion matrix function.
        backend: Backend library (torch or numpy).
        relative_residual_error: Computed relative residual error.
    """
    
    def __init__(self, f, diffusion_matrix=None, backend=torch):
        """
        Initialize the base residual class.

        Parameters:
            f: callable
                The source term function.
            diffusion_matrix: callable, optional
                The diffusion matrix function. Uses identity if None.
            backend: torch or np, optional
                The backend to use for operations. Defaults to torch.
                
        Raises:
            TypeError: If f is not callable or diffusion_matrix is not callable/None.
        """
        # Validate input functions
        if not callable(f):
            raise TypeError("f must be a callable function")
        if diffusion_matrix is not None and not callable(diffusion_matrix):
            raise TypeError("diffusion_matrix must be a callable function or None")
        
        self.f = f

        # Set up diffusion matrix with appropriate backend
        if diffusion_matrix is None:
            self.diffusion_matrix = partial(default_diffusion_matrix, backend=backend)
        else:
            self.diffusion_matrix = partial(diffusion_matrix, backend=backend)

        # Initialize tracking variables
        self.relative_residual_error = None  # Relative L^2 residual error

        # Validate and store backend
        if check_backend(backend):
            self.backend = backend

    @property
    @abstractmethod
    def is_in_divergence_form(self):
        """
        Check if the residual is in divergence form.
        
        Must be implemented by subclasses.
        
        Returns:
            bool
                True if in divergence form, False otherwise.
        """
        pass

    @abstractmethod
    def differential_operator(self, func, x):
        """
        Compute the differential operator for the residual.
        
        Must be implemented by subclasses to define the specific
        differential operator (Laplacian, Hessian, etc.).

        Parameters:
            func: torch.nn.Module or torch.Tensor
                The model or tensor to apply the operator to.
            x: torch.Tensor
                The input coordinates.

        Returns:
            torch.Tensor
                The result of applying the differential operator.
        """
        pass

    def compute_residual(self, model, X):
        """
        Compute the PDE residual loss.
        
        Evaluates the difference between the differential operator applied
        to the model and the true source term.

        Parameters:
            model: torch.nn.Module
                The neural network model.
            X: torch.Tensor
                Input coordinates for residual evaluation.

        Returns:
            torch.Tensor
                The computed residual loss (mean squared error).
                
        Raises:
            ValueError: If backend is not torch (required for gradients).
        """
        # Residual computation requires torch backend for gradients
        if self.backend != torch:
            raise ValueError("Backend must be torch to compute the residual")
        
        # Compute differential operator and source term
        differential_val = self.differential_operator(model, X)
        real_source = self.f(X)

        # Track relative residual error for analysis
        self.relative_residual_error = relative_error(differential_val, real_source)

        # Return mean squared error loss
        return nn.MSELoss()(differential_val, real_source)

