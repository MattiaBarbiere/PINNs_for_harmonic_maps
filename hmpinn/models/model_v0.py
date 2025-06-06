from hmpinn.embedding import Embedding_layer
from hmpinn.models.base import BaseModel
import torch
from torch import nn

#Neural Network Model
class ModelV0(BaseModel):
    def __init__(self,
                 PDE,
                 nodes_hidden_layers = [128, 256, 128], 
                 has_embedding_layer = True, 
                 embedding_size_per_dim = 2,
                 output_dim = 1):
        super().__init__(PDE)
        self._has_embedding_layer = has_embedding_layer
        self.embedding_size_per_dim = embedding_size_per_dim
        self.output_dim = output_dim

        # If we have an embedding layer, we need to have a constant boundary condition for the embedding layer to be useful
        if self.has_embedding_layer and not PDE.type_BC == "Constant":
            raise ValueError("Embedding layers are only supported with constant boundary conditions. Use another model or remove the embedding layer.")

        # Init the embedding block
        self.embedding_block = nn.Sequential()

        #Embedding layer or identity layer
        if self.has_embedding_layer:
            self.embedding_block.add_module("embedding", Embedding_layer(self.embedding_size_per_dim))

            #If we choose to include the embedding layer, the first layer will have the size of the embedding
            self.nodes = [self.embedding_size_per_dim**2] + nodes_hidden_layers + [output_dim]
        else:
            self.embedding_block.add_module("no_embedding", nn.Identity())
            self.nodes = [2] + nodes_hidden_layers + [output_dim]


        #Count the number of layers
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
        #If we have an embedding layer
        if self.has_embedding_layer:
            #If we are training, we need to compute the embedding layer
            if self.training:
                return self.network(self.embedding_block(x))
            
            #If we are not training, we can use the embedding layer and subtract the value of the network at zero
            else:
                embedding_values = self.embedding_block(x)

                # add warning the boundary value must be a constant
                return self.network(embedding_values) - \
                    self.network(torch.zeros_like(embedding_values)) + \
                    self.PDE.BC(x).reshape(-1, 1)
        
        #If we do not have the embedding layer we just return the network
        return self.network(x)
    
    @property
    def has_embedding_layer(self):
        """
        Indicates whether the model has an embedding layer or not.
        """
        return self._has_embedding_layer
