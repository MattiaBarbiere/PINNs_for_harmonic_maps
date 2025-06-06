import torch

def eval_model_and_function(model, func, resolution=100):
    """
    Evaluates the model and the function given as input

    Parameters:
    model: the model to evaluate or path to the model
    func: the function to compare with
    resolution: the resolution of the plot

    Returns:
    None
    """
    # Bring everything to the cpu
    model = model.cpu()
    model.eval()

    # Plotting the functions
    x = torch.linspace(0, 1, resolution).to(torch.device('cpu'))
    y = torch.linspace(0, 1, resolution).to(torch.device('cpu'))
    X, Y = torch.meshgrid(x, y, indexing='xy')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Make sure model is in eval mode
    model.eval()

    # Compute the function values
    F = model(xy)
    F = F.reshape(X.shape).detach().numpy()
    U = func(xy).reshape(X.shape)
    
    # Depending on the type of U, we need to convert it to numpy
    if type(U) == torch.Tensor:
        U = U.detach().numpy()

    return X, Y, F, U