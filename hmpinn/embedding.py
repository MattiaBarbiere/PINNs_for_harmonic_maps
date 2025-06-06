import torch
import torch.nn as nn
import itertools

class Embedding_layer(nn.Module):
    """ Embedding layer """
    def __init__(self, num_embedding_per_dim = 2):
        super().__init__()
        self.num_embedding_per_dim = num_embedding_per_dim

        #Compute the frequencies pairs when it is initialized
        self.frequencies = self.frequency_pairs()

    def frequency_pairs(self):
        # # Frequency values per dim
        freq_vals = [float(i+1) for i in range(self.num_embedding_per_dim)]

        # #Returns a tensor of dim (num_embedding_per_dim, 2)
        return torch.tensor(list(itertools.product(freq_vals, freq_vals)))


    def forward(self, x):
        #Extract the first and second coordinate from the input (while keeping a 2 dim tensor)
        x_0 = x[:,0:1]
        x_1 = x[:, -1:]

        #Extract the values of the frequencies from the input (while keeping a 2 dim tensor)
        ms = self.frequencies[:,0:1].to(x.device)
        ns = self.frequencies[:, -1:].to(x.device)

        #Return the eigenfunctions values   
        return torch.sin(x_0.matmul(ms.T) * torch.pi) * torch.sin(x_1.matmul(ns.T) * torch.pi)
    
    def extra_repr(self):
        return f"num_embedding_per_dim={self.num_embedding_per_dim}"