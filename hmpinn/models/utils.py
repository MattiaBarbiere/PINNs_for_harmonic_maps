import torch.nn as nn

#Function to count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_activation_function(activation_function: str):
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