import torch

class BaseDifferentialOperator:
    def __init__(self):
        """
        Initialize the DifferentialOperator class.
        """
        pass

    def prepare_input(self, func, x):
        """
        Prepare the input for the model.

        Parameters:
        func (torch.nn.Module or torch.tensor): The model to apply operator to. If tensor size is (batch_size)
        x (torch.Tensor): The input to the model, over which the derivative is computed (batch_size, 2)

        Returns:
        torch.Tensor: The prepared input for the model
        """
        # Make sure the function in on the same device as x
        if isinstance(func, torch.nn.Module):
            func.to(x.device)
            
        # Make sure x requires gradients
        if not x.requires_grad:
            x.requires_grad = True
        
        # Check the type of the input
        if callable(func):
            y = func(x)
        elif isinstance(func, torch.Tensor) and func.shape[0] == x.shape[0]:
            y = func
        else:
            raise ValueError("func must be a torch.nn.Module or a torch.Tensor of size (batch_size)")
        
        return y.to(x.device)