"""
Embedding Layer for Neural Networks.

This module provides an embedding layer implementation that transforms 2D coordinates
into high-dimensional representations using trigonometric basis functions.

Classes:
    Embedding_layer: Embedding layer using sine basis functions.
"""

import torch
import torch.nn as nn
import itertools

class Embedding_layer(nn.Module):
    """
    Embedding layer that transforms 2D coordinates into high-dimensional representations.
    
    This layer uses trigonometric basis functions (sine functions) to create embeddings
    that help neural networks better approximate functions with complex boundary conditions.
    The embedding is based on eigenfunction representations of the Laplacian operator.
    
    Attributes:
        num_embedding_per_dim: Number of embedding functions per input dimension.
        frequencies: Tensor containing frequency pairs for the embedding functions.
    """
    
    def __init__(self, num_embedding_per_dim=2):
        """
        Initialize the embedding layer.
        
        Parameters:
            num_embedding_per_dim: int, optional
                Number of embedding functions per input dimension. The total output
                dimension will be num_embedding_per_dim^2. Defaults to 2.
        """
        super().__init__()
        self.num_embedding_per_dim = num_embedding_per_dim

        # Compute the frequency pairs when initialized
        self.frequencies = self.frequency_pairs()

    def frequency_pairs(self):
        """
        Generate frequency pairs for the embedding functions.
        
        Creates all combinations of frequency values from 1 to num_embedding_per_dim
        for both x and y coordinates, forming a grid of frequency pairs.
        
        Returns:
            torch.Tensor
                Tensor of shape (num_embedding_per_dim^2, 2) containing frequency pairs.
        """
        # Frequency values per dimension (starting from 1)
        freq_vals = [float(i+1) for i in range(self.num_embedding_per_dim)]

        # Return tensor of all frequency combinations
        return torch.tensor(list(itertools.product(freq_vals, freq_vals)))

    def forward(self, x):
        """
        Forward pass through the embedding layer.
        
        Transforms 2D input coordinates into high-dimensional embeddings using
        sine basis functions with different frequency pairs.
        
        Parameters:
            x: torch.Tensor
                Input coordinates of shape (batch_size, 2).
        
        Returns:
            torch.Tensor
                Embedded representation of shape (batch_size, num_embedding_per_dim^2).
        """
        # Extract x and y coordinates while maintaining 2D tensor structure
        x_0 = x[:, 0:1]  # Shape: (batch_size, 1)
        x_1 = x[:, -1:]  # Shape: (batch_size, 1)

        # Extract frequency values and move to same device as input
        ms = self.frequencies[:, 0:1].to(x.device)  # x-direction frequencies
        ns = self.frequencies[:, -1:].to(x.device)  # y-direction frequencies

        # Compute sine embeddings: sin(π * m * x) * sin(π * n * y)
        # Uses matrix multiplication for efficient computation across all frequency pairs
        return torch.sin(x_0.matmul(ms.T) * torch.pi) * torch.sin(x_1.matmul(ns.T) * torch.pi)
    
    def extra_repr(self):
        """
        Extra representation string for the module.
        
        Returns:
            str
                String representation showing the number of embeddings per dimension.
        """
        return f"num_embedding_per_dim={self.num_embedding_per_dim}"