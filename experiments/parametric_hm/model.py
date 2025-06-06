import torch.nn as nn
from functools import partial
import torch

# The PDE I use is the sin boundary PDE with the curvature being a parameter.
from hmpinn.PDEs.harmonic_maps.sin_boundaries import SinBoundariesHM

class ParametricHmModel(nn.Module):
    def __init__(self,
                 PDE_class=SinBoundariesHM,
                 param_space=(0, 0.6),
                 nodes_hidden_layers = [64, 64, 64, 64, 64, 64]):
        super().__init__()

        # Add the PDE class to the model
        self.PDE_class = PDE_class

        # The parameter space
        self.param_space = param_space

        # We add another input node for the parameter
        self.nodes = [3] + nodes_hidden_layers + [2]

        # Count the number of layers
        self.numb_layers = len(self.nodes)

        # Start constructing the network
        self.network = nn.Sequential()

        # All layers
        for i in range(0, self.numb_layers-1):
            self.network.add_module("layer_" + str(i), nn.Linear(self.nodes[i], self.nodes[i+1]))

            #Add all the activations except for the output layer
            if i != self.numb_layers-2:
                self.network.add_module(f"gelu_" + str(i), nn.GELU())

        #Initialise
        self.initialise_weights()
        
    #Forward pass
    def forward(self, x, param=None):
        if param is None:
            # Sample a parameter from the parameter space
            param = torch.rand(1,) * (self.param_space[1] - self.param_space[0]) + self.param_space[0]
            param = param.to(x.device)
            param.requires_grad = True
        elif isinstance(param, (int, float)):
            assert self.param_space[0] <= param <= self.param_space[1], f"Parameter {param} is out of the range {self.param_space}"
            param = torch.tensor([param], dtype=x.dtype, device=x.device, requires_grad=True)

        # The pde to the class
        self.PDE = self.PDE_class(curvature=param)

        # Copy the parameter to make a tensor of size (batch_size, 1)
        param = param.unsqueeze(0).expand(x.shape[0], 1)
        
        # Concatenate the parameter to the input
        x = torch.cat((x, param), dim=-1)
        return self.network(x)
    
    @property
    def has_embedding_layer(self):
        return False
    
    #Initialize the weights of the network as suggested by Xavier Glorot Yoshua Bengio (2010)
    def initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)