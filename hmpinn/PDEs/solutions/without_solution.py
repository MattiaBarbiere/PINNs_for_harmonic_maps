from hmpinn.PDEs.solutions.base import BaseSolution
import torch

class WithoutSolution(BaseSolution):
    """
    Base class for PDEs with unknown analytical solutions.
    """
    def __init__(self, *args, backend=torch):
        """
        Initialize with unknown solution.

        Parameters:
        args: The are not used but are added to keep the interface consistent.
        """
        super().__init__(backend=backend)

    @property   
    def has_solution(self):
        """
        Check if the PDE has a known solution.

        Returns:
        bool: False, indicating no known solution.
        """
        return False

    def u(self, x):
        """
        Abstract method to compute the solution.
        Must be implemented in subclasses.

        Parameters:
        x (torch.Tensor): The input to the model.

        Returns:
        torch.Tensor: The solution at x or None if it does not exist.
        """
        return None

    def grad_u(self, x):
        """
        Abstract method to compute the gradient of the solution.
        Must be implemented in subclasses.

        Parameters:
        x (torch.Tensor): The input to the model.

        Returns:
        torch.Tensor: The gradient of the solution at x or None if it does not exist.
        """
        return None

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
        return None