import torch
from hmpinn.differential_operators.base import BaseDifferentialOperator

class Divergence(BaseDifferentialOperator):
    def __init__(self):
        super().__init__()
    
    def __call__(self, func, x):
        """
        Compute the divergence of a model at x.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model to compute the derivative of or a tensor of size (batch_size)
        x (torch.Tensor): The input to the model

        Returns:
        torch.Tensor: The divergence of the model wrt x
        """
        
        # Prepare the input to take derivatives
        y = self.prepare_input(func, x)

        # Check that y has the correct dimensions
        if y.dim() != 2 or y.size(1) != 2:
            raise ValueError("The output of the model must be a tensor of shape (batch_size, 2)")

        # Compute the second derivatives
        partial_x = torch.autograd.grad(y[:, 0].sum(), x, create_graph=True)[0][:, 0]
        partial_y = torch.autograd.grad(y[:, 1].sum(), x, create_graph=True)[0][:, 1]

        # Return the divergence
        return partial_x + partial_y