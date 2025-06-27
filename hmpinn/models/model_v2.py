"""
Model V2 Implementation.

This module provides a simplified neural network model implementation (V2)
without embedding layers and with configurable activation functions.

Classes:
    ModelV2: Simplified neural network model without embedding layers.
"""

from torch import nn

from hmpinn.models.utils import get_activation_function
from hmpinn.models.base import BaseModel

#Neural Network Model
class ModelV2(BaseModel):
    """
    Simplified neural network model without embedding layers (Version 2).
    
    This model provides a straightforward feedforward neural network
    with configurable activation functions. It does not support embedding
    layers, making it suitable for general PDE problems without special
    boundary condition requirements.
    
    Attributes:
        PDE: The PDE problem instance.
        network: The main neural network.
        nodes: List of layer sizes.
        numb_layers: Number of layers in the network.
    """
    
    def __init__(self,
                 PDE,
                 nodes_hidden_layers=[64, 64, 64, 64, 64, 64], 
                 activation_function="gelu",
                 has_embedding_layer=False,
                 output_dim=1):
        """
        Initialize the ModelV2 with specified architecture and activation function.
        
        Parameters:
            PDE: The PDE problem instance to solve.
            nodes_hidden_layers: list of int, optional
                Number of nodes in each hidden layer. Defaults to [64, 64, 64, 64, 64, 64].
            activation_function: str, optional
                Name of the activation function to use. Defaults to "gelu".
            has_embedding_layer: bool, optional
                Must be False for ModelV2. Included for interface compatibility.
            output_dim: int, optional
                Dimension of the output. Defaults to 1.
                
        Raises:
            ValueError: If has_embedding_layer is True (not supported in V2).
        """
        super().__init__(PDE)

        # ModelV2 does not support embedding layers
        if has_embedding_layer:
            raise ValueError("Embedding layers are not supported in model_v2. Please use model_v1 instead.")

        # Configure network architecture (input size is 2D coordinates)
        self.nodes = [2] + nodes_hidden_layers + [output_dim]

        # Count total layers
        self.numb_layers = len(self.nodes)

        # Initialize the network
        self.network = nn.Sequential()

        # Get the activation function instance
        activation = get_activation_function(activation_function)
        
        # Add all layers with specified activation function (except output layer)
        for i in range(0, self.numb_layers-1):
            self.network.add_module("layer_" + str(i), nn.Linear(self.nodes[i], self.nodes[i+1]))

            # Add activation function to all layers except the output layer
            if i != self.numb_layers-2:
                self.network.add_module(f"{activation_function}_" + str(i), activation)
        
        # Initialize network weights
        self.initialise_weights()
        
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Performs a standard feedforward pass without any special
        boundary condition handling.
        
        Parameters:
            x: torch.Tensor
                Input coordinates of shape (batch_size, 2).
        
        Returns:
            torch.Tensor
                Model output of shape (batch_size, output_dim).
        """
        # Standard forward pass without embedding layer
        return self.network(x)
    
    @property
    def has_embedding_layer(self):
        """
        Indicates whether the model has an embedding layer.
        
        Returns:
            bool
                Always False for ModelV2 as it never uses embedding layers.
        """
        return False
