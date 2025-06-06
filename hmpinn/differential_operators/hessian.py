import torch
from hmpinn.differential_operators.base import BaseDifferentialOperator


class Hessian(BaseDifferentialOperator):
    def __init__(self):
        super().__init__()
    
    def __call__(self, func, x):
        """
        Compute the hessian of a model at x.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model to compute the derivative of or a tensor
        x (torch.Tensor): The input to the model with shape [batch_size, input_dim]

        Returns:
        torch.Tensor: The hessian of the model wrt x
            If output dim is 1: returns tensor of shape [batch_size, input_dim, input_dim]
            If output dim > 1: returns tensor of shape [batch_size, output_dim, input_dim, input_dim]
        """
        
        # Prepare the input to take derivatives
        y = self.prepare_input(func, x)
        
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        # Handle scalar outputs case
        if y.ndim == 1:  # Shape [batch_size]
            y = y.unsqueeze(1)  # Convert to [batch_size, 1]
        
        output_dim = y.shape[1]
        hessians = []
        
        for i in range(output_dim):
            # First derivatives for this output dimension
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            
            first_grads = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            hess_rows = []
            for j in range(input_dim):
                # For each element of the gradient, compute its gradient wrt inputs
                second_grads = torch.autograd.grad(
                    outputs=first_grads[:, j],
                    inputs=x,
                    grad_outputs=torch.ones(batch_size, device=x.device),
                    create_graph=True,
                    retain_graph=True 
                )[0]
                hess_rows.append(second_grads)
            
            # Stack rows to form Hessian for this output dimension
            hess_i = torch.stack(hess_rows, dim=1)
            hessians.append(hess_i)
        
        # Stack all output dimensions
        result = torch.stack(hessians, dim=1)
        
        # If scalar output, remove the output_dim dimension for backward compatibility
        if output_dim == 1:
            result = result.squeeze(1)
        
        return result

    # def __call__(self, func, x):
    #     """
    #     Compute the hessian of a model at x.

    #     Parameters:
    #     func (torch.nn.Module or torch.tensor): The model to compute the derivative of or a tensor of size (batch_size)
    #     x (torch.Tensor): The input to the model

    #     Returns:
    #     torch.Tensor: The hessian of the model wrt x
    #     """
        
    #     # Prepare the input to take derivatives
    #     y = self.prepare_input(func, x)

    #     # Compute the gradient
    #     grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    #     # Compute the hessian
    #     hess = []
    #     hess.append(torch.autograd.grad(grad[:, 0].sum(), x, create_graph=True)[0])
    #     hess.append(torch.autograd.grad(grad[:, 1].sum(), x, create_graph=True)[0])

    #     # Return the hessian
    #     return torch.stack(hess, dim=1)