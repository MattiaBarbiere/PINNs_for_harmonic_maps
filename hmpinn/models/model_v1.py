from hmpinn.embedding import Embedding_layer
from hmpinn.models.base import BaseModel
import torch
from torch import nn

#Neural Network Model
class ModelV1(BaseModel):
    def __init__(self,
                 PDE,
                 nodes_hidden_layers = [64, 64, 64, 64, 64, 64],
                 embedding_size_per_dim = 2,
                 output_dim = 1):
        super().__init__(PDE)
        self.embedding_size_per_dim = embedding_size_per_dim
        self.output_dim = output_dim

        # If we have an embedding layer, we need to have a constant boundary condition for the embedding layer to be useful
        if self.has_embedding_layer and not PDE.type_BC == "Constant":
            raise ValueError("Embedding layers are only supported with constant boundary conditions. Use another model or remove the embedding layer.")

        #Create the embedding block (embedding layer + layer normalization)
        self.embedding_block = nn.Sequential(Embedding_layer(self.embedding_size_per_dim))

        #Create a list with sizes of the layers
        self.nodes = [self.embedding_size_per_dim**2] + nodes_hidden_layers + [self.output_dim]
        self.numb_layers = len(self.nodes)

        #Start constructing the network
        self.network = nn.Sequential()
        
        #All layers
        for i in range(0, self.numb_layers-1):
            self.network.add_module("layer_" + str(i), nn.Linear(self.nodes[i], self.nodes[i+1]))

            #Add all the activations except for the output layer
            if i != self.numb_layers-2:
                self.network.add_module("tanh_" + str(i), nn.Tanh())
        
        #Initialise
        self.initialise_weights()
        
    #Forward pass
    def forward(self, x):
        #If we are training, we need to compute the embedding layer
        if self.training:
            x = self.embedding_block(x)
            return self.network(x)
              
        #If we are not training, we can use the embedding layer and subtract the value of the network at zero
        else:
            x = self.embedding_block(x)
            return self.network(x) - \
                    self.network(torch.zeros_like(x)) + \
                    self.PDE.BC(x).reshape(-1, 1)

    @property
    def has_embedding_layer(self):
        return True