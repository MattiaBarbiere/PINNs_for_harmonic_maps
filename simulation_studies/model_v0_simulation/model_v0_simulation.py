#hydra imports
import hydra
from omegaconf import DictConfig

# project imports
from hmpinn.models.model_v0 import ModelV0
from hmpinn.ML_utils import train
from hmpinn.utils import get_PDE_object
# torch import
import torch


@hydra.main(version_base=None, config_path="config_files", config_name="config_3.yaml")
def main(cfg: DictConfig, total_points=300_000):
    # Generate an instance of the Poisson equation depending on the input
    
    poisson_eq = get_PDE_object(cfg.poisson_equation)

    # Create the model
    model = ModelV0(PDE=poisson_eq, 
                     embedding_size_per_dim=cfg.embeddings_per_dim, 
                     embedding_layer=cfg.embedding_layer, 
                     nodes_hidden_layers=cfg.hidden_layers)
    
    # Compute the number of epochs so that the total number of data points is the same
    batch_size = 100 # This remains fixed
    numb_batchs = cfg.numb_batches
    n_epochs = total_points // (batch_size * numb_batchs)

    # Train the model
    errors, grad_errors, loss = train(model, numb_batchs=numb_batchs, batch_size=batch_size, n_epochs=n_epochs, save_BC_loss=False)

    # Save the errors and the model
    torch.save(model.state_dict(), "model.pt")

    # Save the errors and losses
    torch.save(errors, "errors.pt")
    torch.save(grad_errors, "grad_errors.pt")
    torch.save(loss, "loss.pt")

    

if __name__ == "__main__":
    main()