import torch
from hmpinn.differential_operators.base import BaseDifferentialOperator

class Jacobian(BaseDifferentialOperator):
    def __init__(self):
        super().__init__()
    
    def __call__(self, func, x):
        """
        Compute the gradient of a model at x.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model that returns an output of size (batch_size, 2)
        x (torch.Tensor): The input to the model

        Returns:
        torch.Tensor: The jacobian of the model wrt x
        """
        
        # Prepare the input to take derivatives
        y = self.prepare_input(func, x)

        assert y.shape[1] == 2, f"Output of the model must be of size (batch_size, 2), but got {y.shape}"
            
        jacobian = torch.zeros(x.shape[0], 2, 2, dtype=x.dtype, device=x.device)

        for i in range(2):  # loop over output components
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            jacobian[:, i, :] = grads

        return jacobian