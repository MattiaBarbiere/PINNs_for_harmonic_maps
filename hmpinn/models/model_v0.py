"""
Model V0 Implementation.

This module provides the original neural network model implementation (V0)
with optional embedding layers and boundary condition handling.

Classes:
    ModelV0: Original neural network model with optional embedding layer.
"""

import torch
from torch import nn

from hmpinn.embedding import Embedding_layer
from hmpinn.models.base import BaseModel

#Neural Network Model
class ModelV0(BaseModel):
    """
    Original neural network model implementation with optional embedding layer.
    
    This model supports both standard feedforward networks and networks with
    embedding layers for improved boundary condition handling. When using
    embedding layers, only constant boundary conditions are supported.
    
    Attributes:
        PDE: The PDE problem instance.
        embedding_block: The embedding layer or identity layer.
        network: The main neural network.
        nodes: List of layer sizes.
        numb_layers: Number of layers in the network.
    """
    
    def __init__(self,
                 PDE,
                 nodes_hidden_layers=[128, 256, 128], 
                 has_embedding_layer=True, 
                 embedding_size_per_dim=2,
                 output_dim=1):
        """
        Initialize the ModelV0 with specified architecture parameters.
        
        Parameters:
            PDE: The PDE problem instance to solve.
            nodes_hidden_layers: list of int, optional
                Number of nodes in each hidden layer. Defaults to [128, 256, 128].
            has_embedding_layer: bool, optional
                Whether to include an embedding layer. Defaults to True.
            embedding_size_per_dim: int, optional
                Size of embedding per input dimension. Defaults to 2.
            output_dim: int, optional
                Dimension of the output. Defaults to 1.
                
        Raises:
            ValueError: If embedding layer is used with non-constant boundary conditions.
        """
        super().__init__(PDE)
        self._has_embedding_layer = has_embedding_layer
        self.embedding_size_per_dim = embedding_size_per_dim
        self.output_dim = output_dim

        # Validate embedding layer compatibility with boundary conditions
        if self.has_embedding_layer and not PDE.type_BC == "Constant":
            raise ValueError("Embedding layers are only supported with constant boundary conditions. Use another model or remove the embedding layer.")

        # Initialize the embedding block
        self.embedding_block = nn.Sequential()

        # Configure network architecture based on embedding layer presence
        if self.has_embedding_layer:
            self.embedding_block.add_module("embedding", Embedding_layer(self.embedding_size_per_dim))
            # Input size is embedding dimension squared
            self.nodes = [self.embedding_size_per_dim**2] + nodes_hidden_layers + [output_dim]
        else:
            self.embedding_block.add_module("no_embedding", nn.Identity())
            # Input size is 2D coordinates
            self.nodes = [2] + nodes_hidden_layers + [output_dim]

        # Count total layers
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
        
    #Forward pass
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Handles both training and evaluation modes, with special boundary
        condition treatment when using embedding layers during evaluation.
        
        Parameters:
            x: torch.Tensor
                Input coordinates of shape (batch_size, 2).
        
        Returns:
            torch.Tensor
                Model output of shape (batch_size, output_dim).
        """
        # Handle embedding layer case
        if self.has_embedding_layer:
            # During training: standard forward pass
            if self.training:
                return self.network(self.embedding_block(x))
            
            # During evaluation: apply boundary condition correction
            else:
                embedding_values = self.embedding_block(x)

                # Apply boundary condition correction by subtracting network value at origin
                # and adding the actual boundary condition value
                return self.network(embedding_values) - \
                    self.network(torch.zeros_like(embedding_values)) + \
                    self.PDE.BC(x).reshape(-1, 1)
        
        # Standard forward pass without embedding layer
        return self.network(x)
    
    @property
    def has_embedding_layer(self):
        """
        Indicates whether the model has an embedding layer.
        
        Returns:
            bool
                True if the model uses an embedding layer, False otherwise.
        """
        return self._has_embedding_layer
