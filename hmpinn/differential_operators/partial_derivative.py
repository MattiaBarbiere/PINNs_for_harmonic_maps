import torch
from hmpinn.differential_operators.base import BaseDifferentialOperator

class PartialDerivative(BaseDifferentialOperator):
    def __init__(self):
        super().__init__()

    def __call__(self, func, x, dim):
        """
        Compute the partial derivative of a model at x.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model to compute the derivative of or a tensor of size (batch_size)
        x (torch.Tensor): The input to the model
        dim (int): The dimension to compute the derivative with respect to. (0 for x, 1 for y)

        Returns:
        torch.Tensor: The partial derivative of the model wrt x
        """
        
        # Prepare the input to take derivatives
        y = self.prepare_input(func, x)

        # Compute the gradient
        grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # Return the partial derivative
        return grad[:, dim]

    