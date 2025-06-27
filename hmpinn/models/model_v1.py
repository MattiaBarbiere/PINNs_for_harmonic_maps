"""
Model V1 Implementation.

This module provides an improved neural network model implementation (V1)
that always uses embedding layers for enhanced boundary condition handling.

Classes:
    ModelV1: Neural network model with mandatory embedding layer.
"""

import torch
from torch import nn

from hmpinn.embedding import Embedding_layer
from hmpinn.models.base import BaseModel

#Neural Network Model
class ModelV1(BaseModel):
    """
    Neural network model with mandatory embedding layer (Version 1).
    
    This model always includes an embedding layer and is designed specifically
    for problems with constant boundary conditions. It provides improved
    boundary condition satisfaction compared to standard approaches.
    
    Attributes:
        PDE: The PDE problem instance.
        embedding_block: The embedding layer block.
        network: The main neural network.
        nodes: List of layer sizes.
        numb_layers: Number of layers in the network.
        embedding_size_per_dim: Size of embedding per input dimension.
        output_dim: Dimension of the output.
    """
    
    def __init__(self,
                 PDE,
                 nodes_hidden_layers=[64, 64, 64, 64, 64, 64],
                 embedding_size_per_dim=2,
                 output_dim=1):
        """
        Initialize the ModelV1 with embedding layer and specified architecture.
        
        Parameters:
            PDE: The PDE problem instance to solve.
            nodes_hidden_layers: list of int, optional
                Number of nodes in each hidden layer. Defaults to [64, 64, 64, 64, 64, 64].
            embedding_size_per_dim: int, optional
                Size of embedding per input dimension. Defaults to 2.
            output_dim: int, optional
                Dimension of the output. Defaults to 1.
                
        Raises:
            ValueError: If PDE does not have constant boundary conditions.
        """
        super().__init__(PDE)
        self.embedding_size_per_dim = embedding_size_per_dim
        self.output_dim = output_dim

        # Validate boundary condition compatibility
        if self.has_embedding_layer and not PDE.type_BC == "Constant":
            raise ValueError("Embedding layers are only supported with constant boundary conditions. Use another model or remove the embedding layer.")

        # Create the embedding block with embedding layer
        self.embedding_block = nn.Sequential(Embedding_layer(self.embedding_size_per_dim))

        # Configure network architecture
        self.nodes = [self.embedding_size_per_dim**2] + nodes_hidden_layers + [self.output_dim]
        self.numb_layers = len(self.nodes)

        # Construct the main network
        self.network = nn.Sequential()
        
        # Add all layers with Tanh activations (except output layer)
        for i in range(0, self.numb_layers-1):
            self.network.add_module("layer_" + str(i), nn.Linear(self.nodes[i], self.nodes[i+1]))

            # Add activation function to all layers except the output layer
            if i != self.numb_layers-2:
                self.network.add_module("tanh_" + str(i), nn.Tanh())
        
        # Initialize network weights
        self.initialise_weights()
        
    def forward(self, x):
        """
        Forward pass through the neural network with embedding layer.
        
        Applies embedding transformation and handles boundary condition
        correction during evaluation mode.
        
        Parameters:
            x: torch.Tensor
                Input coordinates of shape (batch_size, 2).
        
        Returns:
            torch.Tensor
                Model output of shape (batch_size, output_dim).
        """
        # During training: standard forward pass with embedding
        if self.training:
            x = self.embedding_block(x)
            return self.network(x)
              
        # During evaluation: apply boundary condition correction
        else:
            x = self.embedding_block(x)
            # Apply boundary condition correction by subtracting network value at origin
            # and adding the actual boundary condition value
            return self.network(x) - \
                    self.network(torch.zeros_like(x)) + \
                    self.PDE.BC(x).reshape(-1, 1)

    @property
    def has_embedding_layer(self):
        """
        Indicates whether the model has an embedding layer.
        
        Returns:
            bool
                Always True for ModelV1 as it always uses embedding layers.
        """
        return True