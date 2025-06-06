from torch import nn
from hmpinn.models.utils import get_activation_function
from hmpinn.models.base import BaseModel

#Neural Network Model
class ModelV2(BaseModel):
    def __init__(self,
                 PDE,
                 nodes_hidden_layers = [64, 64, 64, 64, 64, 64], 
                 activation_function = "gelu",
                 has_embedding_layer = False,
                 output_dim = 1):
        super().__init__(PDE)

        # Model v2 does not support embedding layers
        if has_embedding_layer:
            raise ValueError("Embedding layers are not supported in model_v2. Please use model_v1 instead.")

        # Embedding layer or identity layer
        self.nodes = [2] + nodes_hidden_layers + [output_dim]

        # Count the number of layers
        self.numb_layers = len(self.nodes)

        # Start constructing the network
        self.network = nn.Sequential()

        # Get the activate function
        activation = get_activation_function(activation_function)
        
        #All layers
        for i in range(0, self.numb_layers-1):
            self.network.add_module("layer_" + str(i), nn.Linear(self.nodes[i], self.nodes[i+1]))

            #Add all the activations except for the output layer
            if i != self.numb_layers-2:
                self.network.add_module(f"{activation_function}_" + str(i), activation)
        
        #Initialise
        self.initialise_weights()
        
    #Forward pass
    def forward(self, x):
        #If we do not have the embedding layer we just return the network
        return self.network(x)
    
    @property
    def has_embedding_layer(self):
        return False
