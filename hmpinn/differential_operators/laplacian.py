import torch
from hmpinn.differential_operators.base import BaseDifferentialOperator

class Laplacian(BaseDifferentialOperator):
    def __init__(self):
        super().__init__()
    
    def __call__(self, func, x, k = None):
        """
        Compute the Laplacian with diffusion of a model at x.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model to compute the second derivative of or a tensor of size (batch_size)
        x (torch.Tensor): The input to the model
        k (function): The diffusion matrix function that returns a torch tensor of size (batch_size, 2,2)
                    If not provided, the diffusion is the identity matrix

        Returns:
        torch.Tensor: The Laplacian with diffusion of the model wrt x
        """
        # Prepare the input to take derivatives
        y = self.prepare_input(func, x)

        # First derivative
        grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # Apply diffusion matrix if provided, otherwise use identity
        if k is not None:
            diffusion_matrix = k(x, model=func).to(x.device)
            grad = torch.bmm(diffusion_matrix, grad.unsqueeze(-1)).squeeze(-1).to(x.device)

        # Compute the second derivatives
        grad_model_x = torch.autograd.grad(grad[:, 0].sum(), x, create_graph=True)[0][:, 0]
        grad_model_y = torch.autograd.grad(grad[:, 1].sum(), x, create_graph=True)[0][:, 1]

        # Return the Laplacian
        return grad_model_x + grad_model_y