"""
Plotting Utilities.

This module provides utility functions for model evaluation and data preparation
used in plotting and visualization tasks throughout the hmpinn library.

Functions:
    eval_model_and_function: Evaluate model and reference function on a grid.
"""

import torch

def eval_model_and_function(model, func, resolution=100):
    """
    Evaluate the model and reference function on a regular grid.
    
    Creates a uniform grid over the unit square [0,1]x[0,1] and evaluates
    both the neural network model and a reference function for comparison.

    Parameters:
        model: torch.nn.Module or str
            The neural network model to evaluate, or path to a saved model.
        func: callable
            The reference function to compare against the model.
        resolution: int, optional
            Number of grid points in each dimension. Defaults to 100.

    Returns:
        tuple
            A tuple containing:
            - X: numpy.ndarray - X coordinates of the grid
            - Y: numpy.ndarray - Y coordinates of the grid  
            - F: numpy.ndarray - Model outputs on the grid
            - U: numpy.ndarray - Reference function values on the grid
    """
    # Ensure model is on CPU and in evaluation mode
    model = model.cpu()
    model.eval()

    # Create uniform grid over unit square
    x = torch.linspace(0, 1, resolution).to(torch.device('cpu'))
    y = torch.linspace(0, 1, resolution).to(torch.device('cpu'))
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # Flatten grid coordinates for evaluation
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Evaluate model on grid points
    with torch.no_grad():
        F = model(xy)
    F = F.reshape(X.shape).detach().numpy()
    
    # Evaluate reference function on grid points
    U = func(xy).reshape(X.shape)
    
    # Convert to numpy if needed
    if isinstance(U, torch.Tensor):
        U = U.detach().numpy()

    return X, Y, F, U