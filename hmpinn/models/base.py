from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Base class for all models. This class should not be instantiated directly.
    It provides a template for the forward method and the loss computation.
    """
    def __init__(self, PDE):
        super().__init__()
        self.PDE = PDE

    @property
    @abstractmethod
    def has_embedding_layer(self):
        """
        Indicates whether the model has an embedding layer or not.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model. This method should be implemented by subclasses.
        """
        pass

    #Initialize the weights of the network as suggested by Xavier Glorot Yoshua Bengio (2010)
    def initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)