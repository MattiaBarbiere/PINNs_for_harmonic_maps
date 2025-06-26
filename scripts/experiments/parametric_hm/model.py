"""
Parametric Harmonic Map Model.

This module provides a neural network model for learning parametric harmonic maps,
where the PDE parameters (such as curvature) are treated as additional inputs
to the network, enabling the model to learn across parameter spaces.

Classes:
    ParametricHmModel: Neural network for parametric harmonic map problems.
"""

import torch
import torch.nn as nn
from functools import partial

from hmpinn.PDEs.harmonic_maps.sin_boundaries import SinBoundariesHM

class ParametricHmModel(nn.Module):
    """
    Neural network model for parametric harmonic map problems.
    
    This model extends standard harmonic map neural networks by including
    PDE parameters as additional inputs, allowing the network to learn
    solutions across a continuous parameter space.
    
    Attributes:
        PDE_class: Class used to instantiate PDE problems.
        param_space: Valid range for the parameter values.
        nodes: List of layer sizes including parameter input.
        numb_layers: Total number of layers in the network.
        network: The main neural network architecture.
    """
    
    def __init__(self,
                 PDE_class=SinBoundariesHM,
                 param_space=(0, 0.6),
                 nodes_hidden_layers=[64, 64, 64, 64, 64, 64]):
        """
        Initialize the parametric harmonic map model.

        Parameters:
            PDE_class: class, optional
                The PDE class to use for generating problem instances.
                Defaults to SinBoundariesHM.
            param_space: tuple of float, optional
                Range of valid parameter values as (min, max).
                Defaults to (0, 0.6).
            nodes_hidden_layers: list of int, optional
                Number of nodes in each hidden layer.
                Defaults to [64, 64, 64, 64, 64, 64].
        """
        super().__init__()

        # Store PDE configuration
        self.PDE_class = PDE_class
        self.param_space = param_space

        # Configure network architecture (3 inputs: x, y, parameter)
        self.nodes = [3] + nodes_hidden_layers + [2]  # Output dim is 2 for harmonic maps
        self.numb_layers = len(self.nodes)

        # Construct the neural network
        self.network = nn.Sequential()

        # Add all layers with GELU activation (except output layer)
        for i in range(0, self.numb_layers-1):
            self.network.add_module("layer_" + str(i), nn.Linear(self.nodes[i], self.nodes[i+1]))

            # Add activation function to all layers except the output layer
            if i != self.numb_layers-2:
                self.network.add_module(f"gelu_" + str(i), nn.GELU())

        # Initialize network weights
        self.initialise_weights()
        
    def forward(self, x, param=None):
        """
        Forward pass through the parametric harmonic map model.
        
        Handles parameter sampling, validation, and concatenation with
        spatial coordinates before passing through the network.

        Parameters:
            x: torch.Tensor
                Input spatial coordinates of shape (batch_size, 2).
            param: float, int, or None, optional
                Parameter value for the PDE. If None, samples randomly
                from the parameter space. Defaults to None.

        Returns:
            torch.Tensor
                Model output of shape (batch_size, 2) representing the
                harmonic map transformation.
                
        Raises:
            AssertionError: If parameter is outside the valid range.
        """
        # Handle parameter input
        if param is None:
            # Sample a random parameter from the parameter space
            param = torch.rand(1,) * (self.param_space[1] - self.param_space[0]) + self.param_space[0]
            param = param.to(x.device)
            param.requires_grad = True
        elif isinstance(param, (int, float)):
            # Validate parameter is within allowed range
            assert self.param_space[0] <= param <= self.param_space[1], \
                f"Parameter {param} is out of the range {self.param_space}"
            param = torch.tensor([param], dtype=x.dtype, device=x.device, requires_grad=True)

        # Create PDE instance with the given parameter
        self.PDE = self.PDE_class(curvature=param)

        # Expand parameter to match batch size
        param = param.unsqueeze(0).expand(x.shape[0], 1)
        
        # Concatenate spatial coordinates with parameter
        x_augmented = torch.cat((x, param), dim=-1)
        
        return self.network(x_augmented)
    
    @property
    def has_embedding_layer(self):
        """
        Indicates whether the model has an embedding layer.
        
        Returns:
            bool
                Always False for parametric models as they don't use embedding layers.
        """
        return False
    
    def initialise_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        Applies Xavier Glorot uniform initialization to all linear layers
        and sets biases to zero for improved training stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)