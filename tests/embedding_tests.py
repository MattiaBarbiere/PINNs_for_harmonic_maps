from hmpinn.core.embedding import Embedding_layer
import torch

#Algorithm for test the embedding layer from Kast et al. 2023
def test_embedding_layer(numb_embeddings_per_dim = 2, mesh_resolution = 10):
    """
    Test the embedding layer

    Parameters:
    numb_embeddings (int): The number of embeddings to test
    mesh_resolution (int): The resolution of the mesh

    Returns:
    bool: True if the test passes, False otherwise
    list: List of collision points
    """
    #Create the embedding layer
    embedding = Embedding_layer(numb_embeddings_per_dim)

    #Tolerance
    tol = 1/(mesh_resolution)  #As suggested in the paper
    embedding_tol = 0.1 * tol

    #Create a mesh
    x = torch.linspace(0.1, 0.9, mesh_resolution)
    y = torch.linspace(0.1, 0.9, mesh_resolution)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

    #Compute the embedding
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            #Check for collision
            val_i = embedding(xy[i].reshape(1,2))
            val_j = embedding(xy[j].reshape(1,2))
            if torch.norm(val_i - val_j) < embedding_tol and torch.norm(xy[i] - xy[j]) > tol:
                return False, [xy[i], xy[j]]

    return True, []

if __name__ == "__main__":
    assert test_embedding_layer()[0], "The embedding layer is incorrect"
    x = torch.tensor([[0.1,0.1], [0.3, 0.5]])
    print(x)
    emb = Embedding_layer(2)
    print(emb(x))
    print(emb.frequencies)
    print(torch.sin(torch.tensor([0.1 * torch.pi])))
    print("The embedding layer is passes the test")