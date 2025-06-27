"""
Model Utilities.

This module provides utility functions for neural network models in the hmpinn library,
including parameter counting and activation function selection.

Functions:
    count_parameters: Count the number of trainable parameters in a model.
    get_activation_function: Get activation function instance by name.
"""

import torch.nn as nn

def count_parameters(model):
    """
    Count the number of trainable parameters in a neural network model.
    
    Parameters:
        model: torch.nn.Module
            The neural network model to count parameters for.
    
    Returns:
        int
            The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_activation_function(activation_function: str):
    """
    Get an activation function instance by name.
    
    Creates and returns an instance of the specified activation function
    for use in neural network layers.
    
    Parameters:
        activation_function: str
            The name of the activation function. Supported values are:
            "tanh", "relu", "sigmoid", "prelu", "gelu".
    
    Returns:
        torch.nn.Module
            An instance of the requested activation function.
            
    Raises:
        ValueError: If the activation function name is not recognized.
    """
    if activation_function == "tanh":
        activation = nn.Tanh()
    elif activation_function == "relu":
        activation = nn.ReLU()
    elif activation_function == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_function == "prelu":
        activation = nn.PReLU()
    elif activation_function == "gelu":
        activation = nn.GELU()
    else:
        raise ValueError("The activation function is not recognized")
    
    return activation