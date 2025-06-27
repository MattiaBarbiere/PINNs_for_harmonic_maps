"""
Base Model Class.

This module provides the abstract base class for all neural network models
in the hmpinn library. It defines the common interface and shared functionality.

Classes:
    BaseModel: Abstract base class for all neural network models.
"""

from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all neural network models in the hmpinn library.
    
    This class provides a common interface and shared functionality for all
    model implementations, including weight initialization and PDE integration.
    
    Attributes:
        PDE: The PDE problem instance associated with this model.
    """
    
    def __init__(self, PDE):
        """
        Initialize the BaseModel with a PDE problem.
        
        Parameters:
            PDE: The PDE problem instance that this model will solve.
        """
        super().__init__()
        self.PDE = PDE

    @property
    @abstractmethod
    def has_embedding_layer(self):
        """
        Indicates whether the model has an embedding layer.
        
        Returns:
            bool
                True if the model uses an embedding layer, False otherwise.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the neural network model.
        
        This method must be implemented by all subclasses to define
        the forward computation of the model.
        
        Parameters:
            x: torch.Tensor
                Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor
                Output tensor from the model.
        """
        pass

    def initialise_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        Applies Xavier Glorot uniform initialization to all linear layers
        and sets biases to zero, as suggested by Glorot & Bengio (2010).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)