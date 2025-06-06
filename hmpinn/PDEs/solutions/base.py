from abc import ABC, abstractmethod
import torch
from hmpinn.PDEs.utils import check_backend

class BaseSolution(ABC):
    """Base class for PDE solutions."""

    def __init__(self, backend=torch):
        """
        Initialize the base solution class.
        """
        if check_backend(backend):
            self.backend = backend
    
    @property
    @abstractmethod
    def has_solution(self):
        """
        Abstract property to check if the solution exists.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def u(self, x):
        """
        Abstract method to compute the solution.
        Must be implemented in subclasses.

        Parameters:
        x (torch.Tensor): The input to the model.

        Returns:
        torch.Tensor: The solution at x or None if it does not exist.
        """
        pass

    @abstractmethod
    def grad_u(self, x):
        """
        Abstract method to compute the gradient of the solution.
        Must be implemented in subclasses.

        Parameters:
        x (torch.Tensor): The input to the model.

        Returns:
        torch.Tensor: The gradient of the solution at x or None if it does not exist.
        """
        pass

    @abstractmethod
    def compute_relative_grad_error(self, model, X):
        """
        Abstract method to compute the relative gradient error.
        Must be implemented in subclasses.

        Parameters:
        model (torch.nn.Module): The model to compute the gradient of.
        X (torch.Tensor): The input to the model.

        Returns:
        torch.Tensor: The relative gradient error or None if it does not exist.
        """
        pass

    