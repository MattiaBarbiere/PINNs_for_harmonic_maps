from hmpinn.PDEs.solutions.base import BaseSolution
from hmpinn.PDEs.utils import relative_error
from hmpinn.differential_operators.gradient import Gradient
import torch

class WithSolution(BaseSolution):
    """
    Base class for PDEs with known solutions.
    """
    def __init__(self, u, grad_u, backend=torch):
        """
        Initialize with known solution.

        Parameters:
        u (callable): The analytical solution function.
        grad_u (callable): The gradient of the analytical solution function.
        backend (torch or np): The backend to use (default: torch).
        """
        super().__init__(backend=backend)
        # Check the inputs
        if not callable(u):
            raise ValueError("u must be a callable function.")
        if not callable(grad_u):
            raise ValueError("grad_u must be a callable function.")
        self.u_func = u
        self.grad_u_func = grad_u

    @property
    def has_solution(self):
        """
        Check if the PDE has a known solution.

        Returns:
        bool: True, indicating a known solution.
        """
        return True

    def u(self, x):
        """
        Abstract method to compute the solution.
        Must be implemented in subclasses.

        Parameters:
        x (torch.Tensor): The input to the model.

        Returns:
        torch.Tensor: The solution at x or None if it does not exist.
        """
        return self.u_func(x, backend=self.backend)

    def grad_u(self, x):
        """
        Abstract method to compute the gradient of the solution.
        Must be implemented in subclasses.

        Parameters:
        x (torch.Tensor): The input to the model.

        Returns:
        torch.Tensor: The gradient of the solution at x or None if it does not exist.
        """
        return self.grad_u_func(x, backend=self.backend)

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
        # If the backend is not torch, raise an error
        if self.backend != torch:
            raise ValueError("Relative gradient error can only be computed with torch backend.")
        
        # Compute the true gradient if it exists
        true_grad = self.grad_u(X)

        # Compute the model gradient
        model.zero_grad()
        model_grad = Gradient()(model(X), X)

        # Compute the relative gradient error
        return relative_error(model_grad,true_grad) 
        